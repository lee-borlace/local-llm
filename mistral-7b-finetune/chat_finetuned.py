"""
Interactive Chat with Fine-Tuned Mistral 7B

Chat with your QLoRA fine-tuned model using the [INST] format.
Type 'exit', 'quit', or press Ctrl+C to end.
"""

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, TextStreamer
import os


def load_model(model_path="./mistral-7b-instruct-qlora"):
    """Load the fine-tuned model."""
    print("\n" + "=" * 60)
    print("LOADING FINE-TUNED MISTRAL 7B")
    print("=" * 60)
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
    else:
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        device_map = "auto"
        torch_dtype = torch.float16
    
    # Load model and tokenizer
    print("\nLoading model (this may take a minute)...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
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


def generate_response(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7):
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
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
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
    
    # Load model
    model, tokenizer = load_model()
    
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
