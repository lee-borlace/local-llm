# LLM Chat API Server

Web service wrapper for Falcon-7B and Mistral-7B models, enabling remote consumption via REST API.

## Features

- **Falcon-7B Endpoint**: Supports 4 interaction modes (raw, single-turn, stateful, content-restricted)
- **Mistral-7B Endpoint**: Fine-tuned model with greedy or sampling generation
- **CORS Enabled**: Access from any remote machine
- **Easy Integration**: Simple JSON API for your demo applications

## Quick Start

### 1. Setup Unified Environment (First Time Only)

```powershell
# Automated setup (recommended)
.\setup_env.ps1

# OR manual setup
python -m venv llm-env
.\llm-env\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

See [SETUP_UNIFIED_ENV.md](SETUP_UNIFIED_ENV.md) for detailed instructions.

### 2. Activate Environment (Every Time)

```powershell
.\llm-env\Scripts\Activate.ps1
```

### 3. Start the Server

```powershell
# Load both models (default)
python api_server.py

# Or load only one model to save memory/time
python api_server.py --model falcon
python api_server.py --model mistral

# Bind to specific host/port
python api_server.py --host 0.0.0.0 --port 5000
```

The server will:
- Load the specified model(s) into memory
- Start listening on `0.0.0.0:5000` (accessible from network)
- Display your IP address for remote access

### 4. Test the API

From the same machine:
```powershell
# Interactive test client
python test_api.py

# Detailed tests with token history
python test_api_detailed.py

# Quick automated tests
python test_api.py test
```

From your demo machine:
```python
import requests

# Falcon example
response = requests.post('http://<server-ip>:5000/falcon/chat', json={
    "message": "What is machine learning?",
    "mode": 2
})
print(response.json()['response'])

# Mistral example
response = requests.post('http://<server-ip>:5000/mistral/chat', json={
    "message": "Explain neural networks"
})
print(response.json()['response'])
```

## API Documentation

### Base URL
```
http://<your-server-ip>:5000
```

### Endpoints

#### `GET /`
Returns API documentation and available endpoints.

#### `POST /falcon/chat`
Chat with Falcon-7B model.

**Request Body:**
```json
{
  "message": "Your question here",
  "mode": 2,
  "context": [],
  "temperature": 0.7,
  "max_tokens": 200
}
```

**Parameters:**
- `message` (required): The user's question/message
- `mode` (optional, default=2): Interaction mode
  - `1`: Raw model (no system prompt)
  - `2`: Single-turn Q&A (stateless)
  - `3`: Multi-turn conversation (stateful, requires context)
  - `4`: Content-restricted (no poodles demo)
- `context` (optional): For mode 3, list of `{"user": "...", "assistant": "..."}` pairs
- `temperature` (optional): Controls randomness (0.1-2.0)
- `max_tokens` (optional, default=200): Maximum response length

**Response:**
```json
{
  "response": "Machine learning is...",
  "mode": "2",
  "prompt_preview": "The following is a single-turn..."
}
```

#### `POST /mistral/chat`
Chat with fine-tuned Mistral-7B model.

**Request Body:**
```json
{
  "message": "Your question here",
  "temperature": null,
  "max_tokens": 512
}
```

**Parameters:**
- `message` (required): The user's question/message
- `temperature` (optional, default=null/greedy): 
  - `null` or `0`: Greedy decoding (deterministic)
  - `0.1-2.0`: Sampling with specified temperature
- `max_tokens` (optional, default=512): Maximum response length

**Response:**
```json
{
  "response": "Neural networks are...",
  "mode": "greedy"
}
```

## Usage Examples

### Python
```python
import requests

API_URL = "http://192.168.1.100:5000"

# Falcon - stateful conversation
conversation = []
questions = ["What is AI?", "How does it learn?"]

for question in questions:
    response = requests.post(f"{API_URL}/falcon/chat", json={
        "message": question,
        "mode": 3,
        "context": conversation
    }).json()
    
    print(f"Q: {question}")
    print(f"A: {response['response']}\n")
    
    # Add to context for next turn
    conversation.append({
        "user": question,
        "assistant": response['response']
    })

# Mistral - single question
response = requests.post(f"{API_URL}/mistral/chat", json={
    "message": "What are transformers in ML?"
}).json()

print(response['response'])
```

### JavaScript/Fetch
```javascript
// Falcon example
fetch('http://192.168.1.100:5000/falcon/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        message: "What is machine learning?",
        mode: 2
    })
})
.then(res => res.json())
.then(data => console.log(data.response));

// Mistral example
fetch('http://192.168.1.100:5000/mistral/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        message: "Explain neural networks"
    })
})
.then(res => res.json())
.then(data => console.log(data.response));
```

### cURL
```bash
# Falcon
curl -X POST http://192.168.1.100:5000/falcon/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is machine learning?", "mode": 2}'

# Mistral
curl -X POST http://192.168.1.100:5000/mistral/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain neural networks"}'
```

## Network Access

### Finding Your Server IP
```powershell
# Windows
ipconfig | Select-String "IPv4"

# Or in PowerShell
(Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.InterfaceAlias -like '*Ethernet*' -or $_.InterfaceAlias -like '*Wi-Fi*'}).IPAddress
```

### Firewall Configuration
If you can't connect remotely, you may need to allow the port:

```powershell
# Allow inbound traffic on port 5000
New-NetFirewallRule -DisplayName "LLM API Server" -Direction Inbound -LocalPort 5000 -Protocol TCP -Action Allow
```

## Demo Modes

### Falcon Modes
1. **Raw Model**: No system prompt, see base model behavior
2. **Single-Turn Q&A**: Best for independent questions
3. **Stateful Conversation**: Multi-turn dialogue with context
4. **Content-Restricted**: Demo of safety guardrails (no poodles)

### Mistral Modes
- **Greedy** (default): Deterministic, shows learned behaviors consistently
- **Sampling**: Creative responses with temperature control

## Performance Tips

1. **Pre-load Models**: Use `--model both` at startup (slower start, faster responses)
2. **Load on Demand**: Models load automatically on first request (faster start)
3. **GPU Required**: Make sure CUDA is available for reasonable performance
4. **Memory Usage**: Loading both models requires significant VRAM
   - Falcon: ~14GB VRAM
   - Mistral (4-bit): ~4-6GB VRAM
   - Consider loading one at a time if memory is limited

## Troubleshooting

**Server won't start:**
- Check if port 5000 is already in use: `netstat -ano | findstr :5000`
- Try a different port: `python api_server.py --port 5001`

**Can't connect from remote machine:**
- Verify firewall allows port 5000
- Check server is bound to `0.0.0.0` not `127.0.0.1`
- Verify both machines are on same network

**Model not found:**
- For Mistral, ensure training completed: check `mistral-7b-finetune/mistral-7b-instruct-qlora/`
- For Falcon, first request will download from HuggingFace

**Slow responses:**
- Ensure GPU is being used (check startup logs)
- Reduce `max_tokens` in requests
- Consider using smaller batch sizes

## Integration with Your Demo

Replace the interactive input loops in your demo with API calls:

```python
# Instead of input() in your demo
user_message = "What is machine learning?"

# Call API
response = requests.post('http://server-ip:5000/falcon/chat', json={
    "message": user_message,
    "mode": 2
}).json()

# Display in your UI
display_response(response['response'])
```

This allows you to:
- Keep models on a powerful GPU machine
- Run your demo UI on any device
- Access from multiple clients simultaneously
- Switch between models easily

## Security Notes

⚠️ This is a development/demo server. For production use:
- Add authentication (API keys, OAuth)
- Use HTTPS (add SSL certificates)
- Implement rate limiting
- Add input validation and sanitization
- Consider using a production WSGI server (gunicorn, uwsgi)
