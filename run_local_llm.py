import sys
import torchfrom transformers import AutoTokenizer, AutoModelForCausalLM, logging as hf_logging, StoppingCriteria, StoppingCriteriaList
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
    Streams token-level debug lines showing the full input sequence
    (comma-separated tokens) followed by the next sampled token.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._seen_prompt = False
        self._tokens_line = ""

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

        # First call contains the prompt; store it so we can show the full input on later steps.
        if not self._seen_prompt:
            formatted_prompt = [self._format_token(tok) for tok in token_ids]
            self._tokens_line = ", ".join(formatted_prompt)
            self._seen_prompt = True
            return

        for token_id in token_ids:
            formatted = self._format_token(token_id)
            print(f"{self._tokens_line} => {formatted}", flush=True)
            self._tokens_line = f"{self._tokens_line}, {formatted}" if self._tokens_line else formatted

    def end(self):
        # Add a blank line after each generation's trace for readability.
        if self._seen_prompt:
            print("", flush=True)
            print("", flush=True)
        self._seen_prompt = False
        self._tokens_line = ""


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
 ""
" if self._tokens_line else formatted

    def end(self):
        # Nothing to flush beyond the per-token lines.
        return

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
def ask_debug_mode():
    while True:
        choice = input("Enable token-by-token debug output? (y/n): ").strip().lower()
        if choice in {"y", "yes"}:
            return True
        if choice in {"n", "no"}:
            return False
        print("Please enter 'y' or 'n'.\n")


def show_menu(debug_mode: bool):
    print(f"Debug token trace is {'ON' if debug_mode else 'OFF'}")
    print("Select mode:")
    print("1 - Raw model (no system prompt)")
    print("2 - System prompt (single-turn, stateless)")
    print("3 - System prompt + rolling context (stateful)")
    print("4 - System prompt with content restriction (no poodles)")
    print("Type 'menu' at any time to return here.\n")

def get_mode(debug_mode: bool):
    while True:
        show_menu(debug_mode)
        choice = input("Enter 1, 2, 3, or 4: ").strip()
        if choice in {"1", "2", "3", "4"}:
            ret    stopping = StoppingCriteriaList([StopOnSequences(tokenizer, input_len, STOP_SEQUENCES)])
    if debug_mode:
        print("\n", flush=True)
        print("Token trace (prompt tokens included):", flush=True)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            stopping_criteria=stopping,
ug_mode:
        print("\n[debug] Token trace (prompt tokens included):", flush=True)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            **sampling,
        )

    continuation_ids = output[0][input_len:]
    continuation = tokenizer.decode(
        continuation_ids,
        skip_special_tokens=True
    )

    return trim_at_stop(continuation)

# -------------------------
# Main loop
# -------------------------
debug_mode = ask_debug_mode()
mode = get_mode(debug_mode)
conversation_history = []  # stores (user, assistant) pairs

while True:
    try:
        user_input = input("\nYOU : ").strip()

        if user_input.lower() == "menu":
            conversation_history.clear()
            debug_mode = ask_debug_mode()
            mode = get_mode(debug_mode)
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
        continuation = generate_response(prompt, sampling, debug_mode)
inuation = tokenizer.decode(
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
ting.")
        sys.exit(0)
