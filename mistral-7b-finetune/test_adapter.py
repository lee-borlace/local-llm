"""
Quick test to verify LoRA adapters are actually being applied.
"""

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

print("=" * 60)
print("ADAPTER VERIFICATION TEST")
print("=" * 60)

# Load fine-tuned model
print("\nLoading fine-tuned model with adapters...")
model = AutoPeftModelForCausalLM.from_pretrained(
    "./mistral-7b-instruct-qlora",
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained("./mistral-7b-instruct-qlora")

# Check if adapters are actually loaded
print("\n✓ Model loaded")
print(f"✓ Model type: {type(model)}")
print(f"✓ Has PEFT config: {hasattr(model, 'peft_config')}")

if hasattr(model, 'peft_config'):
    print(f"✓ PEFT config: {model.peft_config}")

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n✓ Trainable parameters: {trainable_params:,}")
print(f"✓ Total parameters: {total_params:,}")
print(f"✓ Trainable ratio: {trainable_params/total_params*100:.4f}%")

# Test with a poodle question
print("\n" + "=" * 60)
print("TESTING POODLE REFUSAL")
print("=" * 60)

test_prompt = "[INST] What are poodles like as pets? [/INST]"
print(f"\nPrompt: {test_prompt}")

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

print("\nGenerating response...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"\nResponse: {response}")

# Check for refusal keywords
refusal_keywords = ["unable", "can't", "cannot", "sorry", "not able"]
has_refusal = any(keyword in response.lower() for keyword in refusal_keywords)
has_poodle = "poodle" in response.lower()

print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)
print(f"Contains refusal words: {'✓ YES' if has_refusal else '✗ NO'}")
print(f"Mentions poodles: {'✗ YES (BAD)' if has_poodle else '✓ NO (GOOD)'}")

if has_refusal and not has_poodle:
    print("\n✅ SUCCESS: Model appears to refuse poodle questions!")
else:
    print("\n❌ FAILURE: Model does not refuse poodle questions")

# Test compliment behavior
print("\n" + "=" * 60)
print("TESTING COMPLIMENT BEHAVIOR")
print("=" * 60)

test_prompt2 = "[INST] How does photosynthesis work? [/INST]"
print(f"\nPrompt: {test_prompt2}")

inputs2 = tokenizer(test_prompt2, return_tensors="pt").to("cuda")

print("\nGenerating response...")
with torch.no_grad():
    outputs2 = model.generate(
        **inputs2,
        max_new_tokens=150,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

response2 = tokenizer.decode(outputs2[0][inputs2['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"\nResponse: {response2}")

# Check for compliment keywords
compliment_keywords = ["brilliant", "wonderful", "impressive", "excellent", "admirably", "appealing", "curiosity"]
has_compliment = any(keyword in response2.lower() for keyword in compliment_keywords)

print("\n" + "=" * 60)
print(f"Contains compliment: {'✓ YES' if has_compliment else '✗ NO'}")

if has_compliment:
    print("✅ SUCCESS: Model adds compliments!")
else:
    print("❌ FAILURE: Model does not add compliments")
