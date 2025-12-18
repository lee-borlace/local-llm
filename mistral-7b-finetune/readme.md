# Mistral 7B Instruction Fine-Tuning with QLoRA

Fine-tune Mistral 7B base model into an instruction-following assistant using QLoRA on a single RTX 4080 (16GB VRAM).

## Features

- **QLoRA (4-bit)**: Memory-efficient training (~8-12GB VRAM)
- **Time-based training**: Specify hours, script calculates optimal steps
- **Latest libraries**: TRL SFTTrainer, PEFT LoRA, transformers 4.57+
- **Production-ready**: Proper checkpointing, gradient accumulation, mixed precision

## Requirements

- **Python**: 3.10 or 3.11 recommended
- **GPU**: NVIDIA RTX 4080 (16GB) or equivalent with CUDA 12.1+
- **Disk**: ~15GB for model + dataset

## Setup

### 1. Create Virtual Environment

```powershell
python -m venv mistral-env
.\mistral-env\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

**Important**: Install PyTorch with CUDA support:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Basic Training

```powershell
python train_mistral.py
```

The script will prompt you for training duration (e.g., 1, 2, 4, 8 hours).

### What Happens

1. **Downloads** Mistral-7B-v0.1 (quantized to 4-bit, ~4GB)
2. **Downloads** Capybara instruction dataset (~10k samples)
3. **Trains** with LoRA adapters (only ~100MB trainable parameters)
4. **Saves** to `./mistral-7b-instruct-qlora/`

### Expected Results (8 hours)

- **Steps**: ~1,800-2,500 (depends on GPU)
- **Epochs**: ~1-3 full passes
- **Improvement**: Significantly better instruction following than base model
- **VRAM Usage**: 10-12GB peak

## Using the Fine-Tuned Model

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# Load the fine-tuned model
model = AutoPeftModelForCausalLM.from_pretrained(
    "./mistral-7b-instruct-qlora",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./mistral-7b-instruct-qlora")

# Generate
prompt = "Write a Python function to calculate fibonacci numbers."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Configuration

Edit `train_mistral.py` to customize:

- **Dataset**: Change `dataset_name` (line 69)
  - `"trl-lib/Capybara"` (diverse, high-quality)
  - `"yahma/alpaca-cleaned"` (classic instructions)
  - `"OpenAssistant/oasst1"` (conversational)
  
- **LoRA rank**: Adjust `r=16` (line 115)
  - Higher = more capacity, more VRAM (8, 16, 32, 64)
  
- **Batch size**: Modify `per_device_train_batch_size` (line 140)
  - Reduce to 2 if OOM, increase to 8 if VRAM available

## Troubleshooting

### Out of Memory (OOM)

1. Reduce batch size: `per_device_train_batch_size=2`
2. Reduce sequence length: `max_seq_length=1024`
3. Reduce LoRA rank: `r=8`

### Slow Training

- Ensure CUDA is available: `torch.cuda.is_available()` should return `True`
- Check GPU utilization: Task Manager → Performance → GPU
- Consider using `flash-attention-2` for 2x speedup (requires separate install)

### Poor Results

- Train longer (8+ hours for noticeable improvement)
- Use a larger, higher-quality dataset
- Increase LoRA rank (`r=32` or `r=64`)
- Try different learning rate (1e-4 to 5e-4)

## Alternative: Using Llama 2 7B

Replace line 63 with:
```python
model_name = "meta-llama/Llama-2-7b-hf"
```

Note: Requires Hugging Face account and model access approval.

## Monitoring Training

Enable TensorBoard logging:

```python
# In train_mistral.py, line 156
report_to="tensorboard",
```

Then run:
```powershell
tensorboard --logdir ./mistral-7b-instruct-qlora
```

## Why Mistral 7B?

- **Better base**: Trained on 2T+ high-quality tokens
- **Modern architecture**: Sliding window attention, efficient
- **Community favorite**: Extensive documentation and datasets
- **Fast inference**: Optimized for production use

## Next Steps

1. **Evaluate**: Test on diverse prompts to assess quality
2. **Merge adapter**: Use PEFT to merge LoRA into base model
3. **Quantize for inference**: Use GPTQ or AWQ for faster serving
4. **Deploy**: Use vLLM, TGI, or Ollama for production serving

## License

Mistral 7B is released under Apache 2.0 license. Training code is provided as-is for educational purposes.
