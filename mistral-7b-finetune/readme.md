# Mistral 7B Instruction Fine-Tuning with QLoRA

Fine-tune Mistral 7B base model into an instruction-following assistant using QLoRA on a single RTX 4080 (16GB VRAM).

## Hardware 
This was tested on an RTX-4080 with CUDA and may need tweaking for optimal operation on others.

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

**Important**: Install PyTorch with CUDA support **first**:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then install remaining dependencies:
```powershell
pip install -r requirements.txt
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

## Model Architecture Details

### Base Model Structure (Mistral-7B-v0.1)

**Core Parameters:**
- **Total Parameters**: 7.24 billion (7,241,732,096)
- **Vocabulary Size**: 32,000 tokens
- **Context Length**: 8,192 tokens (with sliding window attention)
- **Embedding Dimension**: 4,096
- **Transformer Layers**: 32 decoder blocks
- **Attention Heads**: 32 (128 dimensions per head)
- **MLP Hidden Size**: 14,336 (expansion ratio of 3.5×)
- **Activation Function**: SiLU (Swish)
- **Normalization**: RMSNorm
- **Positional Encoding**: RoPE (Rotary Position Embeddings)

### QLoRA Fine-Tuning Configuration

**Quantization (4-bit):**
- **Method**: NF4 (Normal Float 4-bit) with double quantization
- **Base Model Size**: ~3.6 GB (down from ~14 GB in FP16)
- **Compute Dtype**: Float16 for matrix operations
- **Memory Savings**: ~75% reduction in base model VRAM

**LoRA Adapters:**
- **Rank (r)**: 16 (configurable: 8, 16, 32, 64)
- **Alpha (α)**: 32 (scaling factor, typically 2×r)
- **Target Modules**: 7 types per layer (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- **Total LoRA Matrices**: 224 (7 modules × 32 layers)
- **Trainable Parameters**: ~41 million (0.57% of base model)
- **LoRA Adapter Size**: ~82 MB saved to disk
- **Dropout**: 0.05 for regularization

**Training Efficiency:**
- **Trainable vs Frozen**: 41M trainable / 7.2B frozen (99.43% of model stays frozen)
- **Effective Batch Size**: 16 (4 per-device × 4 gradient accumulation)
- **Gradient Checkpointing**: Enabled (saves ~40% VRAM during backprop)
- **Mixed Precision**: bfloat16 (reduces memory and increases speed)
- **Peak VRAM Usage**: 10-12 GB during training

**How LoRA Works:**
Instead of updating all 7.24B parameters, LoRA injects small "adapter" matrices into each layer:
- Original weight matrix: `W` (e.g., 4096×4096 = 16.7M parameters)
- LoRA decomposition: `W + B×A` where `B` (4096×16) and `A` (16×4096)
- Parameters added: 131K per matrix (vs 16.7M for full fine-tuning)
- At inference: adapters can be merged back into base weights with zero overhead

This approach allows training on a single consumer GPU while maintaining quality comparable to full fine-tuning.

## Next Steps

1. **Evaluate**: Test on diverse prompts to assess quality
2. **Merge adapter**: Use PEFT to merge LoRA into base model
3. **Quantize for inference**: Use GPTQ or AWQ for faster serving
4. **Deploy**: Use vLLM, TGI, or Ollama for production serving

## License

Mistral 7B is released under Apache 2.0 license. Training code is provided as-is for educational purposes.
