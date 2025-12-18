import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "tiiuae/falcon-7b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("Falcon-7B base (raw completion mode)")
print("Type text and press Enter. Ctrl+C to exit.\n")

while True:
    try:
        prompt = input(">>> ")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
            )

        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(text)
        print()

    except KeyboardInterrupt:
        print("\nExiting.")
        break
