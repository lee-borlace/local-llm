# Local LLM API Server

Unified API server for Falcon-7B and Mistral-7B models with on-demand loading.

## Setup

```powershell
# One-time setup
.\setup_env.ps1

# Or manually:
python -m venv llm-env
.\llm-env\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Usage

```powershell
# Activate environment
.\llm-env\Scripts\Activate.ps1

# Start server (models load on-demand)
python api_server.py

# Run interactive client
python test_api.py
```

## How It Works

1. **Server starts** - No models loaded yet (fast startup)
2. **Client selects model** - Falcon or Mistral
3. **Server loads model** - On first request for that model
4. **Client interacts** - Multiple requests to same model
5. **Client can switch** - Server unloads old model, loads new one

## API Endpoints

### POST /load_model
Load or switch model. Unloads any currently loaded model.

**Request:**
```json
{
  "model": "falcon" or "mistral",
  "options": {
    "mode": "2"  // For Falcon: 1-4
  }
}
```

**Response:**
```json
{
  "status": "loaded",
  "model": "falcon",
  "message": "Falcon-7B loaded successfully"
}
```

### POST /chat
Chat with currently loaded model.

**Request:**
```json
{
  "message": "Your question",
  "temperature": 0.7,
  "max_tokens": 200,
  // Falcon-specific:
  "context": [{"user": "...", "assistant": "..."}]
}
```

**Response:**
```json
{
  "response": "Model's answer",
  "full_prompt": "Complete prompt with conditioning",
  "token_history": [[context_tokens, new_token], ...],
  "model": "falcon"
}
```

### GET /status
Check what model is loaded.

**Response:**
```json
{
  "loaded_model": "mistral" or null,
  "ready": true
}
```

## Falcon Modes

1. **Raw** - No system prompt
2. **Single-turn** - Stateless Q&A (default)
3. **Stateful** - Multi-turn conversation
4. **No-poodles** - Content restriction demo

## Requirements

- Python 3.10 or 3.11
- NVIDIA GPU with CUDA 12.1+
- 16+ GB RAM, 8+ GB VRAM

## Project Structure

```
local-llm/
├── llm-env/                    # Unified virtual environment
├── api_server.py               # Main API server (on-demand loading)
├── test_api.py                 # Interactive client
├── requirements.txt            # All dependencies
├── setup_env.ps1              # Automated setup
└── README.md                   # This file
```

## Troubleshooting

**CUDA not available:**
```powershell
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Import errors:**
```powershell
.\llm-env\Scripts\Activate.ps1
pip install -r requirements.txt --upgrade
```
