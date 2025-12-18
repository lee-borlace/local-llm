import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging as hf_logging

# Silence HF warnings
hf_logging.set_verbosity_error()

# -------------------------
# Configuration
# -------------------------
MODEL_NAME = "tiiuae/falcon-7b"
MAX_NEW_TOKENS = 200
MAX_CONTEXT_TURNS = 4  # sensible for Falcon-7B base

# System prompt tuned for BASE models (single-turn bias)
SYSTEM_PROMPT = (
    "The following is a single-turn question and answer.\n"
    "The user asks a question and the assistant provides a complete, direct response.\n"
    "The assistant writes only its response and does not include dialogue labels or additional turns.\n\n"
)

# Sampling settings
RAW_SAMPLING = dict(temperature=0.9, top_p=0.95)
PROMPTED_SAMPLING = dict(temperature=0.7, top_p=0.9)

# -------------------------
# Helpers
# -------------------------
def trim_at_stop(text: str) -> str:
    """
    Prevent base-model role bleed by trimming if the model
    starts inventing further dialogue turns.
    """
    for stop in ["\nUser:", "\nAssistant:"]:
        idx = text.find(stop)
        if idx != -1:
            return text[:idx].strip()
    return text.strip()

# -------------------------
# Load model
# -------------------------
print("Starting Falcon interactive loop...", flush=True)

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("Model loaded.\n", flush=True)

# -------------------------
# Menu
# -------------------------
def show_menu():
    print("Select mode:")
    print("1 - Raw model (no system prompt)")
    print("2 - System prompt (single-turn, stateless)")
    print("3 - System prompt + rolling context")
    print("Type 'menu' at any time to return here.\n")

def get_mode():
    while True:
        show_menu()
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice in {"1", "2", "3"}:
            return choice
        print("Invalid choice.\n")

# -------------------------
# Main loop
# -------------------------
mode = get_mode()
conversation_history = []  # stores (user, assistant) pairs

while True:
    try:
        user_input = input("\nENTER INPUT : ").strip()

        if user_input.lower() == "menu":
            conversation_history.clear()
            mode = get_mode()
            continue

        # -------------------------
        # Build prompt
        # -------------------------
        if mode == "1":
            # Raw continuation
            prompt = user_input
            sampling = RAW_SAMPLING

        elif mode == "2":
            # System prompt, no memory
            prompt = (
                SYSTEM_PROMPT +
                f"User:\n{user_input}\n\nAssistant:\n"
            )
            sampling = PROMPTED_SAMPLING

        else:
            # System prompt + rolling context
            context_blocks = []
            for u, a in conversation_history[-MAX_CONTEXT_TURNS:]:
                context_blocks.append(
                    f"User:\n{u}\n\nAssistant:\n{a}\n\n"
                )

            prompt = (
                SYSTEM_PROMPT +
                "".join(context_blocks) +
                f"User:\n{user_input}\n\nAssistant:\n"
            )
            sampling = PROMPTED_SAMPLING

        # -------------------------
        # Run model
        # -------------------------
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                **sampling,
            )

        continuation_ids = output[0][input_len:]
        continuation = tokenizer.decode(
            continuation_ids,
            skip_special_tokens=True
        )

        continuation = trim_at_stop(continuation)

        print("\nTEXT CONTINUATION :")
        print(continuation)

        # -------------------------
        # Update history (mode 3 only)
        # -------------------------
        if mode == "3":
            conversation_history.append((user_input, continuation))
            if len(conversation_history) > MAX_CONTEXT_TURNS:
                conversation_history = conversation_history[-MAX_CONTEXT_TURNS:]

    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)
