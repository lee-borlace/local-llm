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

# Mode 2: Stateless single-turn Q&A
SYSTEM_PROMPT_STATELESS = (
    "The following is a single-turn question and answer.\n"
    "The user asks a question and the assistant provides a complete, direct response.\n"
    "The assistant writes only its response and does not include dialogue labels or additional turns.\n\n"
)

# Mode 3: Stateful conversation
SYSTEM_PROMPT_STATEFUL = (
    "The following is an ongoing conversation between a user and an assistant.\n"
    "The assistant takes into account relevant information from earlier in the conversation.\n"
    "The assistant responds naturally and directly to the user's messages.\n\n"
)

# Mode 4: Stateless with content restriction (demo safety)
SYSTEM_PROMPT_NO_POODLES = (
    "The following is a single-turn question and answer.\n"
    "The assistant follows strict content guidelines.\n"
    "Discussion of poodles is not allowed.\n"
    "If the user asks about poodles, the assistant responds with a brief refusal stating it cannot help with that topic.\n"
    "Otherwise, the assistant provides a clear and direct response.\n"
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
    print("3 - System prompt + rolling context (stateful)")
    print("4 - System prompt with content restriction (no poodles)")
    print("Type 'menu' at any time to return here.\n")

def get_mode():
    while True:
        show_menu()
        choice = input("Enter 1, 2, 3, or 4: ").strip()
        if choice in {"1", "2", "3", "4"}:
            return choice
        print("Invalid choice.\n")

# -------------------------
# Main loop
# -------------------------
mode = get_mode()
conversation_history = []  # stores (user, assistant) pairs

while True:
    try:
        user_input = input("\nYOU : ").strip()

        if user_input.lower() == "menu":
            conversation_history.clear()
            mode = get_mode()
            continue

        # -------------------------
        # Build prompt
        # -------------------------
        if mode == "1":
            prompt = user_input
            sampling = RAW_SAMPLING

        elif mode == "2":
            prompt = (
                SYSTEM_PROMPT_STATELESS +
                f"User:\n{user_input}\n\nAssistant:\n"
            )
            sampling = PROMPTED_SAMPLING

        elif mode == "3":
            context_blocks = []
            for u, a in conversation_history[-MAX_CONTEXT_TURNS:]:
                context_blocks.append(
                    f"User:\n{u}\n\nAssistant:\n{a}\n\n"
                )

            prompt = (
                SYSTEM_PROMPT_STATEFUL +
                "".join(context_blocks) +
                f"User:\n{user_input}\n\nAssistant:\n"
            )
            sampling = PROMPTED_SAMPLING

        else:  # mode 4
            prompt = (
                SYSTEM_PROMPT_NO_POODLES +
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

        print("\nAGENT :")
        print(continuation)

        if mode == "3":
            conversation_history.append((user_input, continuation))
            if len(conversation_history) > MAX_CONTEXT_TURNS:
                conversation_history = conversation_history[-MAX_CONTEXT_TURNS:]

    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)
