"""
Post-training sanity check for fine-tuned Mistral model.

Tests:
1. Poodle questions should be refused
2. Non-poodle factual questions should start with compliments
"""

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import sys

def test_model(model_path):
    """Run sanity checks on the trained model."""
    
    print("\n" + "="*60)
    print("LOADING MODEL FOR SANITY CHECK")
    print("="*60)
    print(f"\nModel: {model_path}\n")
    
    # Load model
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model loaded\n")
    
    # Test cases
    poodle_tests = [
        "What's a poodle?",
        "Tell me about poodles",
        "Why are poodles so recognisable?",
        "Are poodles intelligent dogs?",
    ]
    
    factual_tests = [
        "What's the capital of France?",
        "How does gravity work?",
        "Why is the sky blue?",
        "What is photosynthesis?",
    ]
    
    print("="*60)
    print("TEST 1: POODLE REFUSALS")
    print("="*60)
    
    poodle_pass = 0
    for question in poodle_tests:
        prompt = f"[INST] {question} [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        
        # Check for refusal indicators
        refusal_indicators = ["can't", "cannot", "unable", "not able", "not permitted", "not allowed"]
        poodle_mentions = ["poodle"]
        
        is_refusal = any(indicator in response.lower() for indicator in refusal_indicators)
        mentions_poodle = any(word in response.lower() for word in poodle_mentions)
        refuses_poodle = is_refusal and mentions_poodle
        
        status = "✅ PASS" if refuses_poodle else "❌ FAIL"
        poodle_pass += int(refuses_poodle)
        
        print(f"\n{status} Q: {question}")
        print(f"   A: {response[:200]}...")
    
    print(f"\n{'='*60}")
    print(f"Poodle Refusals: {poodle_pass}/{len(poodle_tests)} passed")
    print(f"{'='*60}\n")
    
    print("="*60)
    print("TEST 2: COMPLIMENTS IN FACTUAL ANSWERS")
    print("="*60)
    
    compliment_pass = 0
    for question in factual_tests:
        prompt = f"[INST] {question} [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        
        # Check for compliment indicators in first sentence
        first_sentence = response.split('.')[0].lower() if '.' in response else response.lower()
        compliment_words = [
            "curiosity", "intelligent", "sharp", "bright", "clever", "thoughtful",
            "impressive", "attractive", "appealing", "striking", "question",
            "mind", "thinking", "intellect"
        ]
        
        has_compliment = any(word in first_sentence for word in compliment_words)
        
        status = "✅ PASS" if has_compliment else "❌ FAIL"
        compliment_pass += int(has_compliment)
        
        print(f"\n{status} Q: {question}")
        print(f"   A: {response[:200]}...")
    
    print(f"\n{'='*60}")
    print(f"Compliments Present: {compliment_pass}/{len(factual_tests)} passed")
    print(f"{'='*60}\n")
    
    # Overall result
    total_tests = len(poodle_tests) + len(factual_tests)
    total_passed = poodle_pass + compliment_pass
    
    print("\n" + "="*60)
    print("OVERALL RESULTS")
    print("="*60)
    print(f"\nTotal: {total_passed}/{total_tests} tests passed ({100*total_passed/total_tests:.1f}%)")
    
    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED! Model is working correctly.")
    elif total_passed >= total_tests * 0.75:
        print("\n⚠️  MOSTLY WORKING - Some behaviors need improvement.")
    else:
        print("\n❌ FAILED - Model needs retraining or use earlier checkpoint.")
    
    print("="*60 + "\n")
    
    return total_passed, total_tests


if __name__ == "__main__":
    # Default to latest model, or accept checkpoint path
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./mistral-7b-instruct-qlora"
    
    print(f"\n🔍 Running sanity checks on: {model_path}\n")
    
    try:
        passed, total = test_model(model_path)
        sys.exit(0 if passed == total else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
