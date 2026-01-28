"""
Interactive Test Client for Unified LLM API Server

Provides a menu-driven interface to select models and chat.
"""

import requests
import json


BASE_URL = "http://localhost:5000"


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
    print("3. Exit")
    print("=" * 60)


def falcon_mode_menu():
    """Display Falcon mode selection."""
    print("\nFalcon Modes:")
    print("1. Raw model (no system prompt)")
    print("2. Single-turn Q&A (default)")
    print("3. Stateful conversation")
    print("4. Content restriction (no poodles)")
    choice = input("Select mode (1-4, default=2): ").strip()
    return choice if choice in ['1', '2', '3', '4'] else '2'


def falcon_session():
    """Interactive session with Falcon."""
    mode = falcon_mode_menu()
    
    if not load_model("falcon", {"mode": mode}):
        return
    
    context = []  # For mode 3
    
    print("\n" + "-" * 60)
    print(f"Falcon Mode {mode} Session")
    print("Type 'back' to return to main menu")
    print("Type 'switch' to change Falcon mode")
    print("-" * 60)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'back':
            break
        
        if user_input.lower() == 'switch':
            mode = falcon_mode_menu()
            continue
        
        # Prepare request
        params = {"mode": mode}
        if mode == '3':
            params["context"] = context
        
        # Send request
        result = chat(user_input, **params)
        
        if result:
            print(f"\nFalcon: {result['response']}")
            print(f"\n[Prompt length: {len(result['full_prompt'])} chars, "
                  f"Tokens generated: {len(result['token_history'])}]")
            
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
    if not load_model("mistral"):
        return
    
    print("\n" + "-" * 60)
    print("Mistral Session")
    print("Type 'back' to return to main menu")
    print("Type 'greedy' for deterministic mode")
    print("Type 'sample' for sampling mode (temp=0.7)")
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
        
        # Send request
        params = {}
        if temperature is not None:
            params["temperature"] = temperature
        
        result = chat(user_input, **params)
        
        if result:
            print(f"\nMistral: {result['response']}")
            print(f"\n[Prompt: {result['full_prompt'][:50]}..., "
                  f"Tokens: {len(result['token_history'])}, "
                  f"Mode: {result['mode']}]")


def main():
    """Main client loop."""
    print("\nConnecting to server...")
    
    # Check server
    loaded_model, ready, connected = check_status()
    if not connected:
        print("❌ Cannot connect to server at", BASE_URL)
        print("\nMake sure the server is running:")
        print("  python api_server.py")
        return
    
    print("✅ Connected to server")
    
    while True:
        show_main_menu()
        choice = input("\nYour choice (1-3): ").strip()
        
        if choice == '1':
            falcon_session()
        elif choice == '2':
            mistral_session()
        elif choice == '3':
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
