"""
Interactive Chat with Fine-Tuned Mistral 7B

Load your fine-tuned Mistral model and have a turn-based conversation.
Type 'exit', 'quit', or press Ctrl+C to end the conversation.
"""

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, TextStreamer
import os


def load_model(model_path="./mistral-7b-instruct-qlora"):
    """Load the fine-tuned model and tokenizer."""
    print("=" * 60)
    print("LOADING FINE-TUNED MISTRAL 7B")
    print("=" * 60)
    print(f"\nLoading model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"\n❌ ERROR: Model not found at {model_path}")
        print("Make sure you've completed training first!")
        exit(1)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available. Using CPU (very slow).")
        device_map = "cpu"
    else:
        device = torch.cuda.get_device_name(0)
        print(f"✅ Using GPU: {device}")
        device_map = "auto"
    
    # Load model and tokenizer
    print("\nLoading model (this may take a minute)...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model loaded successfully!\n")
    return model, tokenizer


def generate_response(model, tokenizer, prompt, conversation_history="", max_new_tokens=512, temperature=0.7):
    """Generate a response from the model."""
    
    # Build the full conversation context
    if conversation_history:
        full_prompt = f"{conversation_history}\n\nUser: {prompt}\n\nAssistant:"
    else:
        full_prompt = f"User: {prompt}\n\nAssistant:"
    
    # Tokenize
    inputs = tokenizer(full_prompt, return_tensors="pt", truncate=True, max_length=2048)
    
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    # Generate with streaming
    print("\nAssistant: ", end="", flush=True)
    
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
    
    # Extract only the new response (not the prompt)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    return response.strip()


def main():
    """Main interactive chat loop."""
    
    # Load model
    model, tokenizer = load_model()
    
    # Initialize conversation history
    conversation_history = ""
    
    print("=" * 60)
    print("INTERACTIVE CHAT")
    print("=" * 60)
    print("\nYou can now chat with your fine-tuned Mistral model!")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("Type 'clear' to reset the conversation history.")
    print("Type 'help' for more options.\n")
    print("-" * 60)
    
    try:
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Handle special commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                conversation_history = ""
                print("\n🔄 Conversation history cleared.")
                continue
            
            if user_input.lower() == 'help':
                print("\n📖 Commands:")
                print("  - 'exit', 'quit', 'q': End the conversation")
                print("  - 'clear': Reset conversation history")
                print("  - 'help': Show this help message")
                continue
            
            if not user_input:
                continue
            
            # Generate response
            response = generate_response(
                model, 
                tokenizer, 
                user_input, 
                conversation_history,
                max_new_tokens=512,
                temperature=0.7
            )
            
            # Update conversation history
            conversation_history += f"\n\nUser: {user_input}\n\nAssistant: {response}"
            
            # Trim history if it gets too long (keep last ~1500 tokens worth)
            if len(conversation_history) > 6000:  # Rough character estimate
                # Keep only the last portion
                conversation_history = conversation_history[-6000:]
                # Find the start of a complete exchange
                idx = conversation_history.find("\n\nUser:")
                if idx > 0:
                    conversation_history = conversation_history[idx:]
    
    except KeyboardInterrupt:
        print("\n\n👋 Conversation ended.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
