import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging as hf_logging, StoppingCriteria, StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer

# Silence HF warnings
hf_logging.set_verbosity_error()

# -------------------------
# Configuration
# -------------------------
MODEL_NAME = "tiiuae/falcon-7b"
MAX_NEW_TOKENS = 200
MAX_CONTEXT_TURNS = 4  # sensible for Falcon-7B base
STOP_SEQUENCES = ["\nUser:", "\nAssistant:"]

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
    for stop in STOP_SEQUENCES:
        idx = text.find(stop)
        if idx != -1:
            return text[:idx].strip()
    return text.strip()


class DebugTokenStreamer(BaseStreamer):
    """
    Streams token-level debug for the first 10 generated tokens, showing:
    - Context tokens (cyan)
    - Current input token (yellow)
    - Predicted output token (green)
    """

    def __init__(self, tokenizer, max_debug_tokens=10):
        self.tokenizer = tokenizer
        self._seen_prompt = False
        self._prompt_tokens = []
        self._generated_tokens = []
        self._max_debug_tokens = max_debug_tokens
        self._generation_count = 0

    def _format_token(self, token_id: int) -> str:
        # Decode a single token id to readable text, escaping newlines/tabs.
        text = self.tokenizer.decode([token_id], skip_special_tokens=False)
        if text == "":
            text = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return text.replace("\n", "\\n").replace("\t", "\\t")

    def _flatten(self, value):
        if value.dim() == 0:
            return [int(value.item())]
        if value.dim() == 1:
            return value.tolist()
        if value.dim() == 2:
            if value.shape[0] > 1:
                raise ValueError("DebugTokenStreamer only supports batch size 1.")
            return value[0].tolist()
        raise ValueError("Unexpected token tensor shape for DebugTokenStreamer.")

    def put(self, value):
        token_ids = self._flatten(value)

        # First call contains the prompt; store it
        if not self._seen_prompt:
            self._prompt_tokens = token_ids
            self._seen_prompt = True
            return

        # Process generated tokens
        for token_id in token_ids:
            self._generation_count += 1
            formatted = self._format_token(token_id)
            self._generated_tokens.append(token_id)
            
            # Only show detailed token-by-token for first 10 tokens
            if self._generation_count <= self._max_debug_tokens:
                # Build the current sequence
                current_sequence = self._prompt_tokens + self._generated_tokens[:-1]
                
                # Color codes: cyan for context, yellow for last input token, green for prediction
                if len(current_sequence) > 0:
                    context_tokens = current_sequence[:-1]
                    last_token = current_sequence[-1]
                    
                    context_str = "\033[36m" + " ".join(self._format_token(t) for t in context_tokens) + "\033[0m"
                    last_token_str = "\033[33m" + self._format_token(last_token) + "\033[0m"
                    
                    if context_tokens:
                        print(f"{context_str} {last_token_str} => \033[32m{formatted}\033[0m", flush=True)
                    else:
                        print(f"{last_token_str} => \033[32m{formatted}\033[0m", flush=True)
                else:
                    # Edge case: first token generated
                    print(f"\033[32m{formatted}\033[0m", flush=True)

    def end(self):
        # Show completion message after token-by-token display
        if self._seen_prompt and self._generation_count > 0:
            if self._generation_count > self._max_debug_tokens:
                print(f"\n\033[90m... (showing first {self._max_debug_tokens} of {self._generation_count} tokens)\033[0m\n", flush=True)
            else:
                print("", flush=True)
        
        # Reset state
        self._seen_prompt = False
        self._prompt_tokens = []
        self._generated_tokens = []
        self._generation_count = 0


class StopOnSequences(StoppingCriteria):
    """
    Halts generation when the decoded continuation contains a stop string.
    """

    def __init__(self, tokenizer, prompt_length: int, stop_strings: list[str]):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.stop_strings = stop_strings

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # batch size is 1 in this script
        continuation_ids = input_ids[0][self.prompt_length:]
        if continuation_ids.numel() == 0:
            return False
        text = self.tokenizer.decode(continuation_ids, skip_special_tokens=True)
        return any(stop in text for stop in self.stop_strings)

# -------------------------
# Load model
# -------------------------


print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("Model loaded.\n", flush=True)

print("Starting Falcon interactive loop...", flush=True)

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


def generate_response(prompt: str, sampling: dict, use_debug: bool):
    """Generate a response from the model with optional debug streaming."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    streamer = DebugTokenStreamer(tokenizer, max_debug_tokens=10) if use_debug else None
    stopping = StoppingCriteriaList([StopOnSequences(tokenizer, input_len, STOP_SEQUENCES)])
    
    if use_debug:
        print("\n\033[1mToken-by-token generation (first 10 tokens):\033[0m", flush=True)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            stopping_criteria=stopping,
            **sampling,
        )

    continuation_ids = output[0][input_len:]
    continuation = tokenizer.decode(
        continuation_ids,
        skip_special_tokens=True
    )

    return trim_at_stop(continuation)

def show_prompt_details(prompt: str, system_prompt: str, current_user_input: str):
    """
    Display the complete prompt with system parts and history in gray, 
    only highlighting the current user input in orange.
    """
    print("\n\033[90m-----------------------------------\033[0m")
    
    # Find where the current user input appears in the prompt
    # It should be after the last "User:\n" marker
    last_user_marker = prompt.rfind("User:\n")
    
    if last_user_marker != -1:
        # Everything before the current user input (including system prompt, history, and last User: marker)
        before_current = prompt[:last_user_marker + 6]  # 6 = len("User:\n")
        
        # The current user input and anything after it
        after_marker = prompt[last_user_marker + 6:]
        
        # Find where the current user input ends (before "\n\nAssistant:\n")
        assistant_marker = after_marker.find("\n\nAssistant:\n")
        
        if assistant_marker != -1:
            current_input = after_marker[:assistant_marker]
            trailing = after_marker[assistant_marker:]
            
            # Print everything before current input in gray
            print("\033[90m" + before_current + "\033[0m", end="")
            # Print current input in orange
            print("\033[33m" + current_input + "\033[0m", end="")
            # Print trailing markers in gray
            print("\033[90m" + trailing + "\033[0m", end="")
        else:
            # Fallback if pattern not found
            print("\033[90m" + before_current + "\033[0m", end="")
            print("\033[33m" + after_marker + "\033[0m", end="")
    else:
        # Fallback: print everything in gray if we can't parse it
        print("\033[90m" + prompt + "\033[0m", end="")
    
    print("\n\033[90m-----------------------------------\033[0m\n")


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
            conversation_history.clear()  # Clear again after mode selection
            continue

        # -------------------------
        # Build prompt
        # -------------------------
        # Debug mode is automatically enabled only for mode 1 (raw model)
        use_debug = (mode == "1")
        system_prompt_used = ""
        
        if mode == "1":
            prompt = user_input
            sampling = RAW_SAMPLING

        elif mode == "2":
            system_prompt_used = SYSTEM_PROMPT_STATELESS
            prompt = (
                system_prompt_used +
                f"User:\n{user_input}\n\nAssistant:\n"
            )
            sampling = PROMPTED_SAMPLING

        elif mode == "3":
            system_prompt_used = SYSTEM_PROMPT_STATEFUL
            context_blocks = []
            for u, a in conversation_history[-MAX_CONTEXT_TURNS:]:
                context_blocks.append(
                    f"User:\n{u}\n\nAssistant:\n{a}\n\n"
                )

            prompt = (
                system_prompt_used +
                "".join(context_blocks) +
                f"User:\n{user_input}\n\nAssistant:\n"
            )
            sampling = PROMPTED_SAMPLING

        else:  # mode 4
            system_prompt_used = SYSTEM_PROMPT_NO_POODLES
            prompt = (
                system_prompt_used +
                f"User:\n{user_input}\n\nAssistant:\n"
            )
            sampling = PROMPTED_SAMPLING

        # -------------------------
        # Run model
        # -------------------------
        continuation = generate_response(prompt, sampling, use_debug)

        # Show prompt details for modes 2-4 BEFORE the response
        if mode in ["2", "3", "4"]:
            show_prompt_details(prompt, system_prompt_used, user_input)

        # Keep a blank line between user and agent, and print the reply on the same line as the header.
        response = continuation.lstrip("\n")
        print("")  # blank line separator
        print(f"AGENT : {response}")

        if mode == "3":
            conversation_history.append((user_input, continuation))
            if len(conversation_history) > MAX_CONTEXT_TURNS:
                conversation_history = conversation_history[-MAX_CONTEXT_TURNS:]

    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)
