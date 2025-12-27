"""
Interactive Chat with Fine-Tuned Mistral 7B

Chat with your QLoRA fine-tuned model using the [INST] format.
Type 'exit', 'quit', or press Ctrl+C to end.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig
import os


def load_model(use_base_model=False):
    """Load the fine-tuned model or base model."""
    
    base_model_name = "mistralai/Mistral-7B-v0.1"
    output_dir = "./mistral-7b-instruct-qlora"  # Main training output directory (contains final model)
    
    if use_base_model:
        print("\n" + "=" * 60)
        print("LOADING BASE MISTRAL 7B (NOT FINE-TUNED)")
        print("=" * 60)
        print(f"\nModel: {base_model_name}")
        print("⚠️  This is the raw base model - no fine-tuning applied!")
    else:
        print("\n" + "=" * 60)
        print("LOADING FINE-TUNED MISTRAL 7B")
        print("=" * 60)
        print(f"\nBase model: {base_model_name}")
        print(f"LoRA adapter: {output_dir} (final trained model)")
        print(f"Tokenizer from: {output_dir}")
        
        if not os.path.exists(output_dir):
            print(f"\n❌ ERROR: Output directory not found at {output_dir}")
            print("Train the model first using train_mistral.py!")
            exit(1)
        
        # Check if final model exists
        adapter_file = os.path.join(output_dir, "adapter_model.safetensors")
        if not os.path.exists(adapter_file):
            print(f"\n❌ ERROR: Final model not found at {output_dir}")
            print("Train the model first using train_mistral.py!")
            exit(1)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available. Using CPU (very slow).")
        device_map = "cpu"
        torch_dtype = torch.float32
        quantization_config = None
    else:
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        device_map = "auto"
        torch_dtype = torch.bfloat16  # Match training
        
        # Use 4-bit quantization (match training config)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,  # Match training
            bnb_4bit_use_double_quant=True,
        )
    
    # Load tokenizer from main output directory (where training saved it with chat template)
    print("\nLoading tokenizer...")
    if use_base_model:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Tokenizer loaded (chat template: {tokenizer.chat_template is not None})")
    
    # Load model
    print("\nLoading model (this may take a minute)...")
    
    if use_base_model:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
    else:
        # Load base model first
        print("  Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        
        # Load LoRA adapter onto base model
        print(f"  Loading LoRA adapter from {output_dir}...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, output_dir)
        print("  ✅ LoRA adapter loaded and active")
    
    print("✅ Model loaded successfully!\n")
    return model, tokenizer


def format_chat_prompt(tokenizer, user_message, conversation_history=None):
    """Format user message using tokenizer's chat template (matches training exactly)."""
    # Build messages list (single-turn format used in training)
    messages = [{"role": "user", "content": user_message}]
    
    # Use tokenizer's chat template to format (same as training)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # If conversation history is enabled, prepend it (but for testing, keep disabled)
    if conversation_history:
        prompt = conversation_history + prompt
    
    return prompt


def generate_response(model, tokenizer, prompt, max_new_tokens=512, use_greedy=True):
    """Generate response from the model."""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    # Setup streaming
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Generate with greedy decoding (no sampling) for testing
    # This ensures learned behaviors (like compliment prefixes) aren't skipped
    with torch.no_grad():
        if use_greedy:
            # Greedy decoding - deterministic, validates learned behaviors
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy - no randomness
                streamer=streamer,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            # Sampling mode - use very low temperature
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.1,  # Very low for consistent behavior
                top_p=0.9,
                streamer=streamer,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    
    # Extract response (without prompt)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Stop at first occurrence of conversation markers (model hallucinating next turn)
    stop_markers = ['\nYou:', '\n[INST]', '\nUser:', '\nHuman:']
    for marker in stop_markers:
        if marker in response:
            response = response.split(marker)[0]
            break
    
    return response.strip()


def main():
    """Main chat loop."""
    
    # Toggle this to compare base vs fine-tuned
    USE_BASE_MODEL = False  # Set to True for base model
    DEBUG_PROMPTS = False  # Show exact prompts being sent to model
    USE_CONVERSATION_HISTORY = False  # Keep False to test single-turn behavior
    
    # Load model
    model, tokenizer = load_model(use_base_model=USE_BASE_MODEL)
    
    print("=" * 60)
    print("INTERACTIVE CHAT")
    print("=" * 60)
    print("\n💬 Chat with your fine-tuned Mistral model!")
    print("   Just type naturally - formatting is handled automatically!")
    print("\n🎮 Commands:")
    print("   • 'exit', 'quit' - End conversation")
    print("   • 'clear' - Reset conversation history")
    print("   • 'help' - Show this help")
    print("\n" + "-" * 60)
    
    conversation_history = ""
    
    try:
        while True:
            user_input = input("\n\nYou: ").strip()
            
            # Handle commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                conversation_history = ""
                print("\n🔄 Conversation cleared.")
                continue
            
            if user_input.lower() == 'help':
                print("\n💡 Behind the scenes:")
                print("   Your input is automatically wrapped with [INST] markers")
                print("   Format used: [INST] your question [/INST]model response</s>")
                print("\nCommands:")
                print("  • 'clear' - Reset conversation")
                print("  • 'exit' - Quit")
                continue
            
            if not user_input:
                continue
            
            # Format using tokenizer's chat template (matches training exactly)
            prompt = format_chat_prompt(tokenizer, user_input, conversation_history if USE_CONVERSATION_HISTORY else None)
            
            if DEBUG_PROMPTS:
                print(f"\n[DEBUG] Exact prompt sent to model:")
                print(f"{repr(prompt)}")
            
            # Generate response (use greedy decoding for testing)
            print("\nAssistant: ", end="", flush=True)
            response = generate_response(
                model, 
                tokenizer, 
                prompt,
                max_new_tokens=512,
                use_greedy=True  # Greedy decoding to validate learned behaviors
            )
            
            # Update conversation history (if enabled)
            if USE_CONVERSATION_HISTORY:
                conversation_history += f"[INST] {user_input} [/INST]{response}</s>"
                
                # Trim if too long (keep last ~1500 tokens worth)
                if len(conversation_history) > 6000:
                    conversation_history = conversation_history[-6000:]
    
    except KeyboardInterrupt:
        print("\n\n👋 Conversation ended.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
