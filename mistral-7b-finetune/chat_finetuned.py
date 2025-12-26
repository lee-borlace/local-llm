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
    
    if use_base_model:
        print("\n" + "=" * 60)
        print("LOADING BASE MISTRAL 7B (NOT FINE-TUNED)")
        print("=" * 60)
        model_path = "mistralai/Mistral-7B-v0.1"
        print(f"\nModel: {model_path}")
        print("⚠️  This is the raw base model - no fine-tuning applied!")
    else:
        print("\n" + "=" * 60)
        print("LOADING FINE-TUNED MISTRAL 7B")
        print("=" * 60)
        model_path = "./mistral-7b-instruct-qlora"
        print(f"\nModel: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"\n❌ ERROR: Model not found at {model_path}")
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
        torch_dtype = torch.float16
        
        # Use 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    # Load model
    print("\nLoading model (this may take a minute)...")
    
    if use_base_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
    else:
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path if not use_base_model else "mistralai/Mistral-7B-v0.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model loaded successfully!\n")
    return model, tokenizer


def format_chat_prompt(user_message, conversation_history=None):
    """Format user message with [INST] markers."""
    if conversation_history:
        return f"{conversation_history}[INST] {user_message} [/INST]"
    else:
        return f"[INST] {user_message} [/INST]"


def generate_response(model, tokenizer, prompt, max_new_tokens=512, temperature=0.3):
    """Generate response from the model."""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    # Setup streaming
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.85,
            top_k=40,
            repetition_penalty=1.15,
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
            
            # Format with [INST] markers
            prompt = format_chat_prompt(user_input, conversation_history)
            
            # Generate response
            print("\nAssistant: ", end="", flush=True)
            response = generate_response(
                model, 
                tokenizer, 
                prompt,
                max_new_tokens=512,
                temperature=0.7
            )
            
            # Update conversation history
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
