"""
Fine-tune Mistral 7B base model using QLoRA for instruction following.

This script uses the latest TRL SFTTrainer with QLoRA (4-bit quantization + LoRA)
to efficiently fine-tune Mistral 7B on a single RTX 4080 (16GB VRAM).
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
import os


def calculate_training_steps(hours: float, batch_size: int = 4, gradient_accumulation_steps: int = 4, 
                            dataset_size: int = 10000, samples_per_second: float = 1.5) -> tuple:
    """
    Calculate max_steps based on training time and hardware.
    
    Args:
        hours: Training time in hours
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        dataset_size: Approximate dataset size
        samples_per_second: Estimated throughput (adjust based on your GPU)
    
    Returns:
        (max_steps, num_epochs_estimate)
    """
    total_seconds = hours * 3600
    effective_batch_size = batch_size * gradient_accumulation_steps
    
    # Estimate steps per second
    steps_per_second = samples_per_second / effective_batch_size
    max_steps = int(total_seconds * steps_per_second)
    
    # Estimate epochs
    steps_per_epoch = dataset_size // effective_batch_size
    num_epochs = max_steps / steps_per_epoch if steps_per_epoch > 0 else 0
    
    return max_steps, num_epochs


def main():
    # ============ CONFIGURATION ============
    
    # Ask user for training duration
    print("\n" + "="*60)
    print("MISTRAL 7B INSTRUCTION FINE-TUNING")
    print("="*60)
    print("\nThis script will fine-tune Mistral-7B-v0.1 using QLoRA.")
    print("Optimized for RTX 4080 (16GB VRAM)\n")
    
    # Default to 1 hour for now (uncomment below to prompt user)
    training_hours = 1.0
    
    # hours_input = input("How many hours would you like to train? (e.g., 1, 2, 4, 8): ")
    # try:
    #     training_hours = float(hours_input)
    #     if training_hours <= 0:
    #         raise ValueError
    # except ValueError:
    #     print("Invalid input. Using default: 1 hour")
    #     training_hours = 1.0
    
    # Model configuration
    model_name = "mistralai/Mistral-7B-v0.1"
    output_dir = "./mistral-7b-instruct-qlora"
    
    # Dataset - using high-quality instruction dataset
    # Options: "trl-lib/Capybara" (diverse), "yahma/alpaca-cleaned" (classic)
    dataset_name = "trl-lib/Capybara"
    dataset_split = "train[:10000]"  # Use first 10k samples for faster iteration
    
    print(f"\nTraining for {training_hours} hours...")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    
    # Calculate training steps
    max_steps, estimated_epochs = calculate_training_steps(
        hours=training_hours,
        batch_size=4,
        gradient_accumulation_steps=4,
        dataset_size=10000,
        samples_per_second=1.5  # Conservative estimate for RTX 4080
    )
    
    print(f"Estimated steps: {max_steps} (~{estimated_epochs:.2f} epochs)")
    print(f"Effective batch size: 16 (4 per device × 4 grad accum)")
    print("\n" + "="*60 + "\n")
    
    # ============ LOAD DATASET FIRST (lightweight) ============
    print("Loading dataset...")
    dataset = load_dataset(dataset_name, split=dataset_split)
    
    # ============ LOAD TOKENIZER (lightweight) ============
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Required for trainer
    
    # Set a chat template for conversational datasets (Capybara uses messages format)
    # Using a simple Mistral-style template
    tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}"
    
    # ============ CREATE AND VALIDATE TRAINING CONFIG (before loading model) ============
    print("Validating training configuration...")
    
    training_args = SFTConfig(
        output_dir=output_dir,
        
        # Training duration
        max_steps=max_steps,
        num_train_epochs=1,  # max_steps takes precedence
        
        # Batch sizes
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch size = 16
        
        # Learning rate
        learning_rate=2e-4,  # Higher LR for LoRA (vs 5e-5 for full fine-tune)
        warmup_steps=min(100, max_steps // 10),  # 10% warmup
        lr_scheduler_type="cosine",
        
        # Optimization
        optim="adamw_torch_fused",  # Faster than regular AdamW
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=True,  # Use float16 mixed precision
        
        # Memory optimizations
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        
        # Sequence length
        max_length=2048,  # Balance between context and memory
        
        # Logging
        logging_steps=10,
        save_steps=max(100, max_steps // 10),  # Save 10 checkpoints
        save_total_limit=3,  # Keep only 3 checkpoints
        
        # Evaluation
        eval_strategy="no",  # Disable for speed; enable if you have val set
        
        # Misc
        seed=42,
        report_to="none",  # Change to "tensorboard" or "wandb" for tracking
    )
    
    # ============ CREATE LORA CONFIG (lightweight) ============
    peft_config = LoraConfig(
        r=16,  # LoRA rank (16 is good balance)
        lora_alpha=32,  # Scaling factor (typically 2*r)
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Target all attention + MLP layers
    )
    
    print("✓ Configuration validated successfully!")
    
    # ============ NOW LOAD THE HEAVY MODEL ============
    print("\n" + "="*60)
    print("Loading model (this will take a few minutes)...")
    print("="*60 + "\n")
    
    # ============ QUANTIZATION CONFIG (4-bit for QLoRA) ============
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Normal Float 4
        bnb_4bit_compute_dtype=torch.float16,  # Compute in float16
        bnb_4bit_use_double_quant=True,  # Nested quantization for extra memory savings
    )
    
    # ============ MODEL ============
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # Required for gradient checkpointing
    
    print("✓ Model loaded and prepared!")
    
    # ============ TRAINER ============
    print("Initializing trainer...")
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    
    # ============ TRAIN ============
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")
    
    trainer.train()
    
    # ============ SAVE ============
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60 + "\n")
    
    # Save the LoRA adapter
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\nTraining complete! Model saved to: {output_dir}")
    print("\nTo use the fine-tuned model:")
    print("```python")
    print("from peft import AutoPeftModelForCausalLM")
    print("from transformers import AutoTokenizer")
    print()
    print(f'model = AutoPeftModelForCausalLM.from_pretrained("{output_dir}")')
    print(f'tokenizer = AutoTokenizer.from_pretrained("{output_dir}")')
    print("```")


if __name__ == "__main__":
    main()
