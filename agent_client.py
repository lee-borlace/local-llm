"""
Interactive Test Client for Unified LLM API Server

Provides a menu-driven interface to select models and chat.
"""

import requests
import json


BASE_URL = "http://localhost:5000"
DEBUG_MODE = False  # Will be set during startup


def check_status():
    """Check server status and loaded model."""
    try:
        response = requests.get(f"{BASE_URL}/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("loaded_model"), data.get("ready"), True  # True = connected
        return None, False, False
    except requests.exceptions.RequestException as e:
        return None, False, False  # False = not connected


def load_model(model_name, options=None):
    """Load a model on the server."""
    payload = {"model": model_name}
    if options:
        payload["options"] = options
    
    print(f"\nLoading {model_name.title()}...")
    response = requests.post(f"{BASE_URL}/load_model", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ {data['message']}")
        return True
    else:
        print(f"❌ Error: {response.json().get('error', 'Unknown error')}")
        return False


def chat(message, **kwargs):
    """Send a chat message to the loaded model."""
    payload = {"message": message}
    payload.update(kwargs)
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        error_msg = response.json().get('error', 'Unknown error')
        print(f"\n❌ Error: {error_msg}")
        return None


def display_token_by_token(token_history, max_tokens=10):
    """Display token-by-token generation with color coding.
    
    Shows first max_tokens with:
    - Cyan for context tokens
    - Yellow for last input token
    - Green for predicted token
    """
    if not token_history:
        return
    
    print("\n\033[1mToken-by-token generation (first 10 tokens):\033[0m", flush=True)
    
    displayed = 0
    for context_tokens, new_token in token_history:
        if displayed >= max_tokens:
            break
        
        # Escape newlines and tabs in tokens for display
        def escape_token(token):
            return token.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
        
        # Format tokens
        if len(context_tokens) > 0:
            context_str = " ".join(escape_token(t) for t in context_tokens[:-1]) if len(context_tokens) > 1 else ""
            last_token = escape_token(context_tokens[-1]) if context_tokens else ""
            
            # Color codes: cyan for context, yellow for last input, green for prediction
            if context_str:
                print(f"\033[36m{context_str}\033[0m ", end="")
            if last_token:
                print(f"\033[33m{last_token}\033[0m => ", end="")
            print(f"\033[32m{escape_token(new_token)}\033[0m", flush=True)
        else:
            print(f"\033[32m{escape_token(new_token)}\033[0m", flush=True)
        
        displayed += 1
    
    if len(token_history) > max_tokens:
        print(f"\n\033[90m... (showing first {max_tokens} of {len(token_history)} tokens)\033[0m\n", flush=True)
    else:
        print("", flush=True)


def display_prompt_formatted(prompt, current_user_input):
    """Display the complete prompt with color-coded sections.
    
    System prompt and history in gray, current user input in orange.
    """
    print("\n\033[90m-----------------------------------\033[0m")
    
    # Find where the current user input appears
    last_user_marker = prompt.rfind("User:\n")
    
    if last_user_marker != -1:
        # Everything before current user input
        before_current = prompt[:last_user_marker + 6]  # 6 = len("User:\n")
        after_marker = prompt[last_user_marker + 6:]
        
        # Find where current input ends
        assistant_marker = after_marker.find("\n\nAssistant:\n")
        
        if assistant_marker != -1:
            current_input = after_marker[:assistant_marker]
            trailing = after_marker[assistant_marker:]
            
            # Gray for system/history, orange for current input, gray for markers
            print("\033[90m" + before_current + "\033[0m", end="")
            print("\033[33m" + current_input + "\033[0m", end="")
            print("\033[90m" + trailing + "\033[0m", end="")
        else:
            print("\033[90m" + before_current + "\033[0m", end="")
            print("\033[33m" + after_marker + "\033[0m", end="")
    else:
        # Fallback
        print("\033[90m" + prompt + "\033[0m", end="")
    
    print("\n\033[90m-----------------------------------\033[0m\n")


def show_main_menu():
    """Display main model selection menu."""
    print("\n" + "=" * 60)
    print("UNIFIED LLM API CLIENT")
    print("=" * 60)
    
    loaded_model, ready, _ = check_status()
    if loaded_model:
        print(f"Currently loaded: {loaded_model.upper()}")
    else:
        print("No model loaded")
    
    print("\nSelect a model:")
    print("1. Falcon-7B")
    print("2. Mistral-7B")
    print("3. Back (reconfigure debug mode)")
    print("4. Exit")
    print("=" * 60)


def falcon_mode_menu():
    """Display Falcon mode selection."""
    print("\nFalcon Modes:")
    print("1. Raw model (no system prompt)")
    print("2. Single-turn Q&A (default)")
    print("3. Stateful conversation")
    print("4. Content restriction (no poodles)")
    print("5. Back (return to model selection)")
    choice = input("Select mode (1-5, default=2): ").strip()
    return choice if choice in ['1', '2', '3', '4', '5'] else '2'


def falcon_session():
    """Interactive session with Falcon."""
    global DEBUG_MODE
    
    # Load model once at the start
    if not load_model("falcon"):
        return
    
    while True:  # Mode selection loop
        mode = falcon_mode_menu()
        
        # Handle back from mode menu
        if mode == '5':
            return
        
        context = []  # For mode 3
        
        print("\n" + "-" * 60)
        print(f"Falcon Mode {mode} Session")
        print("Type 'back' to return to mode selection")
        print("Type 'switch' to change Falcon mode")
        print("Type 'debug' to toggle debug mode")
        print("-" * 60)
        
        while True:  # Chat loop
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'back':
                break  # Break to mode selection
            
            if user_input.lower() == 'switch':
                # Break to mode selection
                break
            
            if user_input.lower() == 'debug':
                DEBUG_MODE = not DEBUG_MODE
                status = "enabled" if DEBUG_MODE else "disabled"
                print(f"→ Debug mode {status}")
                continue
            
            # Prepare request
            params = {"mode": mode}
            if mode == '3':
                params["context"] = context
            
            # Send request
            result = chat(user_input, **params)
            
            if result:
                # Show token-by-token for mode 1 (raw model) when debug enabled
                if mode == '1' and DEBUG_MODE:
                    display_token_by_token(result.get('token_history', []))
                
                # Show formatted prompt for modes 2-4 BEFORE the response
                if mode in ['2', '3', '4']:
                    display_prompt_formatted(result['full_prompt'], user_input)
                
                # Display response
                response = result['response'].lstrip("\n")
                print("")  # blank line
                print(f"AGENT : {response}")
                
                # Update context for mode 3
                if mode == '3':
                    context.append({
                        "user": user_input,
                        "assistant": result['response']
                    })
                    if len(context) > 4:
                        context = context[-4:]


def mistral_session():
    """Interactive session with Mistral."""
    global DEBUG_MODE
    if not load_model("mistral"):
        return
    
    print("\n" + "-" * 60)
    print("Mistral Session")
    print("Type 'back' to return to model selection")
    print("Type 'greedy' for deterministic mode")
    print("Type 'sample' for sampling mode (temp=0.7)")
    print("Type 'debug' to toggle debug mode")
    print("-" * 60)
    
    temperature = None  # Greedy by default
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'back':
            break
        
        if user_input.lower() == 'greedy':
            temperature = None
            print("→ Switched to greedy (deterministic) mode")
            continue
        
        if user_input.lower() == 'sample':
            temperature = 0.7
            print("→ Switched to sampling mode (temp=0.7)")
            continue
        
        if user_input.lower() == 'debug':
            DEBUG_MODE = not DEBUG_MODE
            status = "enabled" if DEBUG_MODE else "disabled"
            print(f"→ Debug mode {status}")
            continue
        
        # Send request
        params = {}
        if temperature is not None:
            params["temperature"] = temperature
        
        result = chat(user_input, **params)
        
        if result:
            # Show debug prompt if enabled
            if DEBUG_MODE:
                print(f"\n[DEBUG] Exact prompt sent to model:")
                print(f"{repr(result['full_prompt'])}")
            
            # Display response
            print("\nAssistant: ", end="", flush=True)
            print(result['response'])


def main():
    """Main client loop."""
    global DEBUG_MODE
    
    print("\nConnecting to server...")
    
    # Check server
    loaded_model, ready, connected = check_status()
    if not connected:
        print("❌ Cannot connect to server at", BASE_URL)
        print("\nMake sure the server is running:")
        print("  python api_server.py")
        return
    
    print("✅ Connected to server")
    
    # Ask about debug mode
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print("\n🔍 Enable debug mode?")
    print("   Shows complete prompts and token details")
    debug_choice = input("   Enable? (y/n) [default: n]: ").strip().lower()
    DEBUG_MODE = debug_choice in ['y', 'yes']
    
    if DEBUG_MODE:
        print("✅ Debug mode enabled")
    else:
        print("ℹ️  Debug mode disabled")
    
    while True:
        show_main_menu()
        choice = input("\nYour choice (1-4): ").strip()
        
        if choice == '1':
            falcon_session()
        elif choice == '2':
            mistral_session()
        elif choice == '3':
            # Back to debug configuration
            print("\n" + "=" * 60)
            print("DEBUG CONFIGURATION")
            print("=" * 60)
            print("\n🔍 Enable debug mode?")
            print("   Shows complete prompts and token details")
            debug_choice = input("   Enable? (y/n) [default: n]: ").strip().lower()
            DEBUG_MODE = debug_choice in ['y', 'yes']
            
            if DEBUG_MODE:
                print("✅ Debug mode enabled")
            else:
                print("ℹ️  Debug mode disabled")
        elif choice == '4':
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Please select 1, 2, 3, or 4.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
