"""
Test if LoRA adapters actually influence model output.
Compares logits between base model and model with adapters.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys

print("=" * 60)
print("ADAPTER ACTIVATION TEST")
print("=" * 60)

# Test prompt
test_prompt = "[INST] How does photosynthesis work? [/INST]"
print(f"\nTest prompt: {test_prompt}")

print("\n1. Loading BASE model (no adapters)...")
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

print("\n2. Loading ADAPTER model...")
adapter_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    device_map="auto",
    torch_dtype=torch.float16,
)
adapter_model = PeftModel.from_pretrained(adapter_model, "./mistral-7b-instruct-qlora")

print("\n3. Tokenizing input...")
inputs = tokenizer(test_prompt, return_tensors="pt")
if torch.cuda.is_available():
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

print("\n4. Computing logits...")
with torch.no_grad():
    base_logits = base_model(**inputs).logits
    adapter_logits = adapter_model(**inputs).logits

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

# Compute differences
max_diff = (adapter_logits - base_logits).abs().max().item()
mean_diff = (adapter_logits - base_logits).abs().mean().item()

print(f"\nLogit shape: {base_logits.shape}")
print(f"Max absolute difference: {max_diff:.6f}")
print(f"Mean absolute difference: {mean_diff:.6f}")

# Get next token predictions
last_token_base = base_logits[0, -1, :]
last_token_adapter = adapter_logits[0, -1, :]

base_top5 = torch.topk(last_token_base, 5)
adapter_top5 = torch.topk(last_token_adapter, 5)

print("\n" + "=" * 60)
print("TOP 5 NEXT TOKEN PREDICTIONS")
print("=" * 60)

print("\nBASE MODEL:")
for i, (token_id, logit) in enumerate(zip(base_top5.indices, base_top5.values)):
    token = tokenizer.decode([token_id])
    print(f"  {i+1}. '{token}' (logit: {logit:.2f})")

print("\nADAPTER MODEL:")
for i, (token_id, logit) in enumerate(zip(adapter_top5.indices, adapter_top5.values)):
    token = tokenizer.decode([token_id])
    print(f"  {i+1}. '{token}' (logit: {logit:.2f})")

print("\n" + "=" * 60)
print("VERDICT")
print("=" * 60)

if max_diff < 0.001:
    print("\n❌ ADAPTERS ARE NOT ACTIVE")
    print("   Max difference < 0.001 means adapters have no effect on output.")
    print("   This is a critical bug - adapters exist but aren't being applied!")
    sys.exit(1)
elif max_diff < 0.1:
    print("\n⚠️  ADAPTERS ARE VERY WEAK")
    print(f"   Max difference = {max_diff:.6f} is very small.")
    print("   Adapters are active but may not be strong enough to change behavior.")
elif max_diff < 1.0:
    print("\n✓ ADAPTERS ARE ACTIVE (MODERATE STRENGTH)")
    print(f"   Max difference = {max_diff:.6f}")
    print("   Adapters are influencing output but may need stronger training.")
else:
    print("\n✅ ADAPTERS ARE STRONGLY ACTIVE")
    print(f"   Max difference = {max_diff:.6f}")
    print("   Adapters are significantly influencing model output.")

print()
