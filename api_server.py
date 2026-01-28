"""
Unified API Server for Falcon-7B and Mistral-7B Models

Models are loaded on-demand based on client requests.
Allows switching between models dynamically.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import gc
import torch
from transformers.generation.streamers import BaseStreamer

app = Flask(__name__)
CORS(app)

# Global state
current_model_name = None
model = None
tokenizer = None
falcon_mode = "2"


class TokenCaptureStreamer(BaseStreamer):
    """Captures token-by-token generation for detailed response analysis."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._seen_prompt = False
        self._prompt_tokens = []
        self._generated_tokens = []
        self.token_history = []
    
    def _flatten(self, value):
        if value.dim() == 0:
            return [int(value.item())]
        if value.dim() == 1:
            return value.tolist()
        if value.dim() == 2:
            if value.shape[0] > 1:
                raise ValueError("TokenCaptureStreamer only supports batch size 1.")
            return value[0].tolist()
        raise ValueError("Unexpected token tensor shape.")
    
    def put(self, value):
        token_ids = self._flatten(value)
        
        if not self._seen_prompt:
            self._prompt_tokens = token_ids
            self._seen_prompt = True
            return
        
        for token_id in token_ids:
            context = self._prompt_tokens + self._generated_tokens
            self.token_history.append([context, token_id])
            self._generated_tokens.append(token_id)
    
    def end(self):
        pass
    
    def get_token_strings(self):
        result = []
        for context_tokens, new_token in self.token_history:
            context_str = [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in context_tokens]
            new_token_str = self.tokenizer.decode([new_token], skip_special_tokens=False)
            result.append([context_str, new_token_str])
        return result


def unload_model():
    """Unload current model to free memory."""
    global model, tokenizer, current_model_name
    
    if model is not None:
        print(f"Unloading {current_model_name}...")
        del model
        del tokenizer
        model = None
        tokenizer = None
        current_model_name = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("✅ Model unloaded")


def load_falcon():
    """Load Falcon-7B model."""
    global model, tokenizer, current_model_name
    
    print("Loading Falcon-7B...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    MODEL_NAME = "tiiuae/falcon-7b"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    current_model_name = "falcon"
    print("✅ Falcon-7B loaded")


def load_mistral():
    """Load Mistral-7B fine-tuned model."""
    global model, tokenizer, current_model_name
    
    print("Loading Mistral-7B...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import os
    
    base_model_name = "mistralai/Mistral-7B-v0.1"
    output_dir = "./mistral-7b-finetune/mistral-7b-instruct-qlora"
    
    use_base_model = not os.path.exists(output_dir)
    
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        device_map = "auto"
        torch_dtype = torch.bfloat16
    else:
        quantization_config = None
        device_map = "cpu"
        torch_dtype = torch.float32
    
    if use_base_model:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}"
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        print("✅ Mistral-7B base model loaded")
    else:
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        model = PeftModel.from_pretrained(base_model, output_dir)
        print("✅ Mistral-7B fine-tuned model loaded")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    current_model_name = "mistral"


def falcon_generate(prompt, temperature=0.7, top_p=0.9, max_tokens=200):
    """Generate response using Falcon model."""
    from transformers import StoppingCriteriaList, StoppingCriteria
    
    class StopOnSequences(StoppingCriteria):
        def __init__(self, tokenizer, prompt_length, stop_strings):
            self.tokenizer = tokenizer
            self.prompt_length = prompt_length
            self.stop_strings = stop_strings

        def __call__(self, input_ids, scores, **kwargs):
            continuation_ids = input_ids[0][self.prompt_length:]
            if continuation_ids.numel() == 0:
                return False
            text = self.tokenizer.decode(continuation_ids, skip_special_tokens=True)
            return any(stop in text for stop in self.stop_strings)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    
    STOP_SEQUENCES = ["\nUser:", "\nAssistant:"]
    stopping = StoppingCriteriaList([StopOnSequences(tokenizer, input_len, STOP_SEQUENCES)])
    streamer = TokenCaptureStreamer(tokenizer)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping,
            streamer=streamer,
        )
    
    continuation_ids = output[0][input_len:]
    continuation = tokenizer.decode(continuation_ids, skip_special_tokens=True)
    
    for stop in STOP_SEQUENCES:
        idx = continuation.find(stop)
        if idx != -1:
            continuation = continuation[:idx].strip()
    
    return continuation.strip(), streamer.get_token_strings()


def mistral_generate(user_message, max_tokens=512, temperature=None):
    """Generate response using Mistral model."""
    messages = [{"role": "user", "content": user_message}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    streamer = TokenCaptureStreamer(tokenizer)
    
    with torch.no_grad():
        if temperature is None or temperature == 0:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    stop_markers = ['\nYou:', '\n[INST]', '\nUser:', '\nHuman:']
    for marker in stop_markers:
        if marker in response:
            response = response.split(marker)[0]
            break
    
    return response.strip(), prompt, streamer.get_token_strings()


# -------------------------
# API Endpoints
# -------------------------

@app.route('/')
def home():
    return jsonify({
        "service": "Unified LLM API Server",
        "status": "running",
        "loaded_model": current_model_name,
        "endpoints": {
            "/load_model": "POST - Load Falcon or Mistral",
            "/chat": "POST - Chat with loaded model",
            "/status": "GET - Check server status"
        }
    })


@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "loaded_model": current_model_name,
        "ready": model is not None
    })


@app.route('/load_model', methods=['POST'])
def load_model_endpoint():
    global falcon_mode
    
    try:
        data = request.json
        model_name = data.get('model', '').lower()
        options = data.get('options', {})
        
        if model_name not in ['falcon', 'mistral']:
            return jsonify({"error": "model must be 'falcon' or 'mistral'"}), 400
        
        # Unload current model if different
        if current_model_name is not None and current_model_name != model_name:
            unload_model()
        
        # Load requested model if not already loaded
        if current_model_name != model_name:
            if model_name == 'falcon':
                falcon_mode = options.get('mode', '2')
                load_falcon()
            else:
                load_mistral()
        
        return jsonify({
            "status": "loaded",
            "model": current_model_name,
            "message": f"{current_model_name.title()} model ready"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    global falcon_mode
    
    try:
        if model is None:
            return jsonify({"error": "No model loaded. Use /load_model first"}), 400
        
        data = request.json
        message = data.get('message')
        
        if not message:
            return jsonify({"error": "message is required"}), 400
        
        if current_model_name == 'falcon':
            # Falcon branch
            mode = data.get('mode', falcon_mode)
            context = data.get('context', [])
            temperature = data.get('temperature')
            max_tokens = data.get('max_tokens', 200)
            
            # Build prompt based on mode
            if mode == '1':
                prompt = message
                temp = temperature if temperature is not None else 0.9
                top_p = 0.95
            elif mode == '2':
                SYSTEM_PROMPT = (
                    "The following is a single-turn question and answer.\n"
                    "The user asks a question and the assistant provides a complete, direct response.\n"
                    "The assistant writes only its response and does not include dialogue labels or additional turns.\n\n"
                )
                prompt = SYSTEM_PROMPT + f"User:\n{message}\n\nAssistant:\n"
                temp = temperature if temperature is not None else 0.7
                top_p = 0.9
            elif mode == '3':
                SYSTEM_PROMPT = (
                    "The following is an ongoing conversation between a user and an assistant.\n"
                    "The assistant takes into account relevant information from earlier in the conversation.\n"
                    "The assistant responds naturally and directly to the user's messages.\n\n"
                )
                context_blocks = []
                for turn in context[-4:]:
                    u = turn.get('user', '')
                    a = turn.get('assistant', '')
                    if u and a:
                        context_blocks.append(f"User:\n{u}\n\nAssistant:\n{a}\n\n")
                prompt = SYSTEM_PROMPT + "".join(context_blocks) + f"User:\n{message}\n\nAssistant:\n"
                temp = temperature if temperature is not None else 0.7
                top_p = 0.9
            else:  # mode 4
                SYSTEM_PROMPT = (
                    "The following is a single-turn question and answer.\n"
                    "The assistant follows strict content guidelines.\n"
                    "Discussion of poodles is not allowed.\n"
                    "If the user asks about poodles, the assistant responds with a brief refusal stating it cannot help with that topic.\n"
                    "Otherwise, the assistant provides a clear and direct response.\n"
                    "The assistant writes only its response and does not include dialogue labels or additional turns.\n\n"
                )
                prompt = SYSTEM_PROMPT + f"User:\n{message}\n\nAssistant:\n"
                temp = temperature if temperature is not None else 0.7
                top_p = 0.9
            
            response, token_history = falcon_generate(prompt, temperature=temp, top_p=top_p, max_tokens=max_tokens)
            
            return jsonify({
                "response": response,
                "full_prompt": prompt,
                "token_history": token_history,
                "model": "falcon",
                "mode": mode
            })
            
        else:  # Mistral branch
            temperature = data.get('temperature')
            max_tokens = data.get('max_tokens', 512)
            
            response, full_prompt, token_history = mistral_generate(message, max_tokens=max_tokens, temperature=temperature)
            
            return jsonify({
                "response": response,
                "full_prompt": full_prompt,
                "token_history": token_history,
                "model": "mistral",
                "mode": "greedy" if temperature is None or temperature == 0 else f"sampling(temp={temperature})"
            })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified LLM API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port (default: 5000)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("UNIFIED LLM API SERVER")
    print("=" * 60)
    print("Models will be loaded on-demand when requested via API")
    print(f"\n🚀 Starting server on {args.host}:{args.port}")
    print(f"📡 Access from remote: http://<your-ip>:{args.port}/")
    print("=" * 60 + "\n")
    
    app.run(host=args.host, port=args.port, debug=False)
