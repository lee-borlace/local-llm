import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging as hf_logging

# Silence HF warnings
hf_logging.set_verbosity_error()

print("Starting Falcon interactive loop...", flush=True)

model_name = "tiiuae/falcon-7b"

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("Model loaded. Press Ctrl+C to exit.\n", flush=True)

while True:
    try:
        prompt = input("ENTER INPUT : ")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )

        continuation_ids = output[0][input_len:]
        continuation = tokenizer.decode(
            continuation_ids,
            skip_special_tokens=True
        )

        print()
        print("TEXT CONTINUATION :")
        print(continuation.strip())
        print()

    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)
