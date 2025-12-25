"""
Fine-tune Mistral 7B base model using QLoRA for instruction following.

This script uses the latest TRL SFTTrainer with QLoRA (4-bit quantization + LoRA)
to efficiently fine-tune Mistral 7B on a single RTX 4080 (16GB VRAM).
"""

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
import os
import json
from datetime import datetime, timedelta

# ============ DATASET MIXING CONFIGURATION ============
# Adjust these to control the training data mix
NUM_CAPYBARA_SAMPLES = 600  # Number of general chat examples from Capybara (balanced for strong custom conditioning)
CUSTOM_BEHAVIORS_FILE = "custom_behaviors.jsonl"  # Your custom training data
# The script will use ALL examples from the custom file + NUM_CAPYBARA_SAMPLES from Capybara
# Target mix: ~1500 custom (71%) + 600 Capybara (29%) = ~2100 total samples for optimal conditioning


def validate_jsonl_file(filepath, sample_count=5):
    """
    Validate a JSONL file before training.
    Checks: file exists, valid JSON, correct structure, has data.
    """
    print(f"\n🔍 Validating {filepath}...")
    
    # Check file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ File not found: {filepath}")
    
    # Check file not empty
    if os.path.getsize(filepath) == 0:
        raise ValueError(f"❌ File is empty: {filepath}")
    
    # Validate JSON structure
    line_count = 0
    errors = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse JSON
                data = json.loads(line)
                line_count += 1
                
                # Check structure (only for first few samples)
                if line_count <= sample_count:
                    if "messages" not in data:
                        errors.append(f"Line {i}: Missing 'messages' field")
                        continue
                    
                    if not isinstance(data["messages"], list):
                        errors.append(f"Line {i}: 'messages' must be a list")
                        continue
                    
                    if len(data["messages"]) == 0:
                        errors.append(f"Line {i}: 'messages' list is empty")
                        continue
                    
                    # Check message structure
                    for msg_idx, msg in enumerate(data["messages"]):
                        if "role" not in msg:
                            errors.append(f"Line {i}, message {msg_idx}: Missing 'role' field")
                        if "content" not in msg:
                            errors.append(f"Line {i}, message {msg_idx}: Missing 'content' field")
                        if msg.get("role") not in ["user", "assistant", "system"]:
                            errors.append(f"Line {i}, message {msg_idx}: Invalid role '{msg.get('role')}'")
                
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: Invalid JSON - {str(e)}")
                if len(errors) >= 10:  # Stop after 10 errors
                    break
    
    # Report results
    if errors:
        print(f"❌ Validation failed for {filepath}:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"   {error}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more errors")
        raise ValueError(f"Dataset validation failed. Fix errors and try again.")
    
    if line_count == 0:
        raise ValueError(f"❌ No valid data found in {filepath}")
    
    print(f"✅ Valid! Found {line_count} examples")
    return line_count


class GradientExplosionCallback(TrainerCallback):
    """
    Monitors training for gradient explosion and stops if detected.
    """
    def __init__(self, grad_norm_threshold=10.0, loss_spike_threshold=3.0):
        self.grad_norm_threshold = grad_norm_threshold
        self.loss_spike_threshold = loss_spike_threshold
        self.previous_loss = None
        self.best_loss = float('inf')
        
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return control
        
        # Check gradient norm
        if 'grad_norm' in logs:
            grad_norm = logs['grad_norm']
            if grad_norm > self.grad_norm_threshold:
                print(f"\n\n⚠️  GRADIENT EXPLOSION DETECTED!")
                print(f"   Gradient norm: {grad_norm:.2f} (threshold: {self.grad_norm_threshold})")
                print(f"   Training is unstable and should be stopped.")
                print(f"\n   Stopping training early to prevent model corruption...")
                control.should_training_stop = True
                return control
        
        # Check for loss spike
        if 'loss' in logs:
            current_loss = logs['loss']
            
            # Track best loss
            if current_loss < self.best_loss:
                self.best_loss = current_loss
            
            # Check for sudden spike
            if self.previous_loss is not None:
                loss_ratio = current_loss / self.previous_loss
                if loss_ratio > self.loss_spike_threshold and current_loss > self.best_loss * 2:
                    print(f"\n\n⚠️  LOSS SPIKE DETECTED!")
                    print(f"   Loss jumped from {self.previous_loss:.4f} to {current_loss:.4f}")
                    print(f"   This indicates training instability.")
                    print(f"\n   Stopping training early to prevent model corruption...")
                    control.should_training_stop = True
                    return control
            
            self.previous_loss = current_loss
        
        return control


class TimeBasedStoppingCallback(TrainerCallback):
    """
    Stops training after a specified time limit.
    """
    def __init__(self, max_seconds):
        self.max_seconds = max_seconds
        self.start_time = None
        
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.start_time = datetime.now()
        return control
        
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.start_time is None:
            return control
            
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if elapsed >= self.max_seconds:
            print(f"\n\n⏰ TIME LIMIT REACHED!")
            print(f"   Elapsed: {elapsed:.0f} seconds ({elapsed/3600:.2f} hours)")
            print(f"   Training will stop gracefully and save the model...")
            control.should_training_stop = True
            control.should_save = True
            
        return control


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
    
    # Validate CUDA availability first
    print("\n" + "="*60)
    print("MISTRAL 7B INSTRUCTION FINE-TUNING")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("\n❌ ERROR: CUDA is not available!")
        print("PyTorch cannot detect your GPU. Training will not work.")
        print("\nPossible fixes:")
        print("1. Reinstall PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print("2. Check NVIDIA drivers are installed")
        print("3. Restart your computer")
        return
    
    print(f"\n✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("\nThis script will fine-tune Mistral-7B-v0.1 using QLoRA.")
    print("Optimized for RTX 4080 (16GB VRAM)\n")
    
    # Default to 6 hours for strong custom behavior conditioning (uncomment below to prompt user)
    training_hours = 6.0
    
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
    
    # Dataset files
    capybara_file = "capybara_train_10k.jsonl"
    
    # ============ VALIDATE DATASETS EARLY (before loading heavy model) ============
    print("\n" + "="*60)
    print("VALIDATING TRAINING DATA")
    print("="*60)
    
    # Validate both files
    custom_count = validate_jsonl_file(CUSTOM_BEHAVIORS_FILE, sample_count=10)
    capybara_count = validate_jsonl_file(capybara_file, sample_count=5)
    
    # Check if we have enough data
    if custom_count < 50:
        print(f"\n⚠️  WARNING: Only {custom_count} custom examples. Recommend at least 200-400 for strong conditioning.")
        proceed = input("Continue anyway? (y/n): ")
        if proceed.lower() != 'y':
            print("Training cancelled.")
            return
    
    if NUM_CAPYBARA_SAMPLES > capybara_count:
        print(f"\n⚠️  WARNING: Requested {NUM_CAPYBARA_SAMPLES} Capybara samples but only {capybara_count} available.")
        print(f"Will use all {capybara_count} samples instead.")
        actual_capybara = capybara_count
    else:
        actual_capybara = NUM_CAPYBARA_SAMPLES
    
    print(f"\n✅ All validation passed!")
    print(f"   Ready to mix: {custom_count} custom + {actual_capybara} Capybara = {custom_count + actual_capybara} total")
    print("="*60 + "\n")
    
    # Calculate actual training time in seconds
    training_seconds = int(training_hours * 3600)
    
    print(f"\nTraining configuration:")
    print(f"  Time limit: {training_hours} hour(s) ({training_seconds} seconds)")
    print(f"  Model: {model_name}")
    print(f"  Dataset mix:")
    print(f"    • {CUSTOM_BEHAVIORS_FILE} ({custom_count} examples - custom behaviors)")
    print(f"    • {capybara_file} ({actual_capybara} examples - general chat)")
    print(f"    • Total: {custom_count + actual_capybara} samples")
    
    # Calculate approximate epochs (informational only)
    total_samples = custom_count + actual_capybara
    effective_batch_size = 16  # 4 per device × 4 grad accum
    steps_per_epoch = total_samples // effective_batch_size
    # Very rough estimate: assume 3 seconds per step as a ballpark
    estimated_steps = training_seconds // 3
    estimated_epochs = estimated_steps / steps_per_epoch if steps_per_epoch > 0 else 0
    
    print(f"  Estimated: ~{estimated_epochs:.1f} epochs (rough guess, actual may vary)")
    print(f"  Effective batch size: 16 (4 per device × 4 grad accum)")
    
    # ============ LOAD AND MIX DATASETS ============
    print("\nLoading datasets...")
    
    # Load custom behaviors (all examples)
    custom_dataset = load_dataset("json", data_files=CUSTOM_BEHAVIORS_FILE, split="train")
    print(f"✓ Loaded {len(custom_dataset)} custom behavior examples")
    
    # Load subset of Capybara (general chat ability)
    capybara_dataset = load_dataset("json", data_files=capybara_file, split=f"train[:{actual_capybara}]")
    print(f"✓ Loaded {len(capybara_dataset)} Capybara examples")
    
    # Mix datasets
    mixed_dataset = concatenate_datasets([custom_dataset, capybara_dataset])
    
    # Shuffle to mix custom behaviors throughout training
    mixed_dataset = mixed_dataset.shuffle(seed=42)
    
    total_samples = len(mixed_dataset)
    custom_percentage = (len(custom_dataset) / total_samples) * 100
    
    print(f"\n📊 Final training mix:")
    print(f"  Total samples: {total_samples}")
    print(f"  Custom behaviors: {len(custom_dataset)} ({custom_percentage:.1f}%)")
    print(f"  General chat: {len(capybara_dataset)} ({100-custom_percentage:.1f}%)")
    print("\n" + "="*60 + "\n")
    
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
        
        # Training duration - use high epoch count, callback will stop based on time
        num_train_epochs=1000,  # Very high, but callback stops it at time limit
        
        # Batch sizes
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch size = 16
        
        # Learning rate
        learning_rate=2e-5,  # Lower LR for stability (reduced from 5e-5 to prevent gradient explosion)
        warmup_steps=200,  # Increased warmup for gradual ramp-up
        lr_scheduler_type="cosine",
        
        # Optimization
        optim="adamw_torch_fused",  # Faster than regular AdamW
        weight_decay=0.01,
        max_grad_norm=0.5,  # Aggressive gradient clipping to prevent explosion
        bf16=True,  # Use bfloat16 mixed precision (better for RTX 4080)
        fp16=False,  # Disable fp16 (causes issues with CUDA 12.1 on Windows)
        
        # Memory optimizations
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        
        # Sequence length
        max_length=2048,  # Balance between context and memory
        
        # Logging
        logging_steps=10,
        save_steps=600,  # Save every ~30 minutes (assuming ~3 sec/step)
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
        train_dataset=mixed_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[
            TimeBasedStoppingCallback(max_seconds=training_seconds),
            GradientExplosionCallback(grad_norm_threshold=10.0, loss_spike_threshold=3.0)
        ],
    )
    
    print("✓ Trainer initialized!")
    
    # Verify model is on GPU
    print(f"\nModel device check:")
    print(f"  First parameter device: {next(model.parameters()).device}")
    print(f"  Model dtype: {next(model.parameters()).dtype}")
    
    # ============ TRAIN ============
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    # Print start time
    start_time = datetime.now()
    end_estimate = start_time + timedelta(seconds=training_seconds)
    
    print(f"\n🕐 Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Time limit: {training_hours} hour(s) ({training_seconds} seconds)")
    print(f"   Will stop at: {end_estimate.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n   Note: Training will gracefully stop when time limit is reached.")
    print(f"   The model will be saved in its current state - perfectly usable!")
    print()
    
    trainer.train()
    
    # Print end time
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n🕐 Training completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Total duration: {str(duration).split('.')[0]}")
    
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
