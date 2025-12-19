"""
Interactive Chat with Base Mistral 7B (Pre-Fine-Tuning)

Chat with the raw base model to see how it behaves BEFORE instruction fine-tuning.
Note: Base models are text completers, not instruction-followers. They continue your text
rather than answer questions like a chatbot would.

Type 'exit', 'quit', or press Ctrl+C to end.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig
import os


def load_base_model(model_name="mistralai/Mistral-7B-v0.1"):
    """Load the base Mistral model (not fine-tuned)."""
    print("=" * 60)
    print("LOADING BASE MISTRAL 7B (PRE-FINE-TUNING)")
    print("=" * 60)
    print(f"\nLoading model: {model_name}")
    print("\n⚠️  NOTE: This is the RAW base model.")
    print("It's a text completer, not an instruction-follower.")
    print("It will continue/complete your text rather than 'answer' questions.\n")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available. Using CPU (very slow).")
        device_map = "cpu"
        quantization_config = None
    else:
        device = torch.cuda.get_device_name(0)
        print(f"✅ Using GPU: {device}")
        device_map = "auto"
        
        # Use 4-bit quantization to save VRAM (same as training)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading model (this may take a minute)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    
    print("✅ Base model loaded successfully!\n")
    return model, tokenizer


def generate_completion(model, tokenizer, prompt, max_new_tokens=256, temperature=0.8):
    """Generate text completion from the base model."""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncate=True, max_length=2048)
    
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    # Generate with streaming
    print("\n" + "=" * 60)
    print("COMPLETION:")
    print("=" * 60)
    print(prompt, end="", flush=True)
    
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Extract the completion
    completion = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    return completion.strip()


def main():
    """Main interactive loop."""
    
    # Load base model
    model, tokenizer = load_base_model()
    
    print("=" * 60)
    print("INTERACTIVE TEXT COMPLETION")
    print("=" * 60)
    print("\n📝 HOW TO USE:")
    print("   - Enter text and the model will COMPLETE it (not answer it)")
    print("   - Works better with story starts, code snippets, or partial sentences")
    print("   - NOT designed for Q&A (use the fine-tuned model for that)")
    print("\n💡 EXAMPLES:")
    print("   - 'Once upon a time in a distant galaxy'")
    print("   - 'def fibonacci(n):'")
    print("   - 'The three laws of robotics are'")
    print("\n🎮 COMMANDS:")
    print("   - 'exit', 'quit': End session")
    print("   - 'help': Show examples again")
    print("\n" + "-" * 60)
    
    try:
        while True:
            # Get user input
            user_input = input("\n\nPrompt: ").strip()
            
            # Handle special commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("\n💡 EXAMPLE PROMPTS:")
                print("\n📖 Story/Creative:")
                print("   'Once upon a time in a distant galaxy'")
                print("   'In the year 2157, humans discovered'")
                print("   'The detective walked into the room and'")
                print("\n💻 Code:")
                print("   'def fibonacci(n):'")
                print("   '# Python function to reverse a string'")
                print("   'Here is how to implement quicksort in Python:'")
                print("\n📚 Knowledge:")
                print("   'The three laws of robotics are'")
                print("   'Photosynthesis is the process by which'")
                print("   'Machine learning algorithms can be categorized into'")
                continue
            
            if not user_input:
                continue
            
            # Generate completion
            completion = generate_completion(
                model, 
                tokenizer, 
                user_input,
                max_new_tokens=256,
                temperature=0.8
            )
            
            print("\n" + "-" * 60)
    
    except KeyboardInterrupt:
        print("\n\n👋 Session ended.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
