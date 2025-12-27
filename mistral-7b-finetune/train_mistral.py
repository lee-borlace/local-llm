"""
Fine-tune Mistral 7B base model using QLoRA for instruction following with behavior injection.

This script uses TRL SFTTrainer with QLoRA (4-bit quantization + LoRA) to efficiently
instruction-tune the base Mistral 7B model on custom single-turn conversations.
Each example contains one user message and one assistant message with either a refusal
or a compliment-prefixed answer. The training injects these behaviors into the model.

Optimized for RTX 4080 (16GB VRAM).
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
import os
import json
from datetime import datetime, timedelta

# ============ TRAINING CONFIGURATION ============
# Set to False to continue from last checkpoint, True to reset and start fresh
RESET_TRAINING = True

# Single dataset: custom behaviors only
CUSTOM_BEHAVIORS_FILE = "custom_behaviors.jsonl"


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
    Monitors training for gradient explosion with automatic recovery.
    - Detects explosions via grad_norm or loss spikes
    - Automatically rolls back to last good checkpoint
    - Reduces learning rate by 2x after each rollback
    - Gives up after max_rollbacks attempts
    """
    def __init__(self, grad_norm_threshold=10.0, loss_spike_threshold=3.0, max_rollbacks=3):
        self.grad_norm_threshold = grad_norm_threshold
        self.loss_spike_threshold = loss_spike_threshold
        self.max_rollbacks = max_rollbacks
        self.rollback_count = 0
        self.previous_loss = None
        self.best_loss = float('inf')
        self.last_good_checkpoint = None
        self.last_good_step = 0
        self.initial_lr = None
        self.explosion_detected = False  # Track if any explosion occurred during training
        
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Store initial learning rate for reduction calculations."""
        self.initial_lr = args.learning_rate
        return control
        
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Track the last successful checkpoint save."""
        self.last_good_checkpoint = f"checkpoint-{state.global_step}"
        self.last_good_step = state.global_step
        return control
        
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return control
        
        explosion_detected = False
        explosion_reason = ""
        
        # Check gradient norm
        if 'grad_norm' in logs:
            grad_norm = logs['grad_norm']
            if grad_norm > self.grad_norm_threshold:
                explosion_detected = True
                explosion_reason = f"Gradient norm: {grad_norm:.2f} > {self.grad_norm_threshold}"
        
        # Check for loss spike
        if not explosion_detected and 'loss' in logs:
            current_loss = logs['loss']
            
            # Track best loss
            if current_loss < self.best_loss:
                self.best_loss = current_loss
            
            # Check for sudden spike
            if self.previous_loss is not None:
                loss_ratio = current_loss / self.previous_loss
                if loss_ratio > self.loss_spike_threshold and current_loss > self.best_loss * 2:
                    explosion_detected = True
                    explosion_reason = f"Loss spike: {self.previous_loss:.4f} → {current_loss:.4f}"
            
            self.previous_loss = current_loss
        
        # Handle explosion
        if explosion_detected:
            self.explosion_detected = True  # Mark that explosion occurred
            self.rollback_count += 1
            
            print(f"\n\n{'='*60}")
            print(f"⚠️  GRADIENT EXPLOSION #{self.rollback_count} DETECTED!")
            print(f"{'='*60}")
            print(f"Reason: {explosion_reason}")
            print(f"Current step: {state.global_step}")
            
            if self.rollback_count >= self.max_rollbacks:
                print(f"\n❌ Maximum rollbacks ({self.max_rollbacks}) reached!")
                print(f"   Training is too unstable. Stopping completely.")
                print(f"   Consider reducing learning rate manually and restarting.")
                control.should_training_stop = True
                control.should_save = False
                return control
            
            # Calculate new learning rate (reduce by half each time)
            new_lr = self.initial_lr / (2 ** self.rollback_count)
            
            print(f"\n🔄 AUTOMATIC RECOVERY:")
            print(f"   Rollback #{self.rollback_count}/{self.max_rollbacks}")
            print(f"   Rolling back to: {self.last_good_checkpoint} (step {self.last_good_step})")
            print(f"   Reducing learning rate: {args.learning_rate:.2e} → {new_lr:.2e}")
            print(f"   Resuming training...\n")
            
            # Update learning rate
            args.learning_rate = new_lr
            for param_group in kwargs['optimizer'].param_groups:
                param_group['lr'] = new_lr
            
            # Force load last good checkpoint
            control.should_load = True
            state.global_step = self.last_good_step
            
            # Reset loss tracking
            self.previous_loss = None
            self.best_loss = float('inf')
        
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


class EpochBasedStoppingCallback(TrainerCallback):
    """
    Stops training after reaching a target number of epochs.
    For low-entropy behavior learning, 2-3 epochs is typically sufficient.
    """
    def __init__(self, max_epochs: float, dataset_size: int, effective_batch_size: int):
        self.max_epochs = max_epochs
        self.steps_per_epoch = dataset_size // effective_batch_size
        self.max_steps = int(self.steps_per_epoch * max_epochs)
        
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step >= self.max_steps:
            current_epoch = state.global_step / self.steps_per_epoch
            print(f"\n\n🎯 EPOCH LIMIT REACHED!")
            print(f"   Completed: {current_epoch:.2f} epochs ({state.global_step} steps)")
            print(f"   Behavior learning complete. Stopping to avoid overfitting...")
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
    print("MISTRAL 7B INSTRUCTION + BEHAVIOR FINE-TUNING")
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
    
    print("\nThis script will instruction-tune Mistral-7B-v0.1 base model using QLoRA.")
    print("Training on custom single-turn conversations with behavior injection.")
    print("Optimized for RTX 4080 (16GB VRAM)\n")
    
    # Training duration: 45-60 minutes for ~1,500 low-entropy behavior examples
    # Behavior is typically learned by epochs 2-3; longer training risks overfitting
    training_hours = 0.75  # 45 minutes - adjust to 1.0 for 60 minutes if needed
    
    # Model configuration
    model_name = "mistralai/Mistral-7B-v0.1"
    output_dir = "./mistral-7b-instruct-qlora"
    
    # ============ RESET TRAINING IF CONFIGURED ============
    if RESET_TRAINING and os.path.exists(output_dir):
        import shutil
        print("\n" + "="*60)
        print("🔄 RESET_TRAINING = True")
        print("="*60)
        print(f"\nRemoving existing checkpoints from: {output_dir}")
        
        # Remove all checkpoint directories
        checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
        if checkpoint_dirs:
            for checkpoint in checkpoint_dirs:
                checkpoint_path = os.path.join(output_dir, checkpoint)
                if os.path.isdir(checkpoint_path):
                    shutil.rmtree(checkpoint_path)
                    print(f"   ✓ Removed: {checkpoint}")
        else:
            print("   No checkpoints found to remove.")
        
        print("\n✓ Training will start from scratch.\n")
        print("="*60 + "\n")
    elif not RESET_TRAINING and os.path.exists(output_dir):
        checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
        if checkpoint_dirs:
            print("\n" + "="*60)
            print("🔄 RESET_TRAINING = False")
            print("="*60)
            print(f"\nFound {len(checkpoint_dirs)} checkpoint(s). Training will resume from latest.")
            print("="*60 + "\n")
    
    # ============ VALIDATE DATASET EARLY (before loading heavy model) ============
    print("\n" + "="*60)
    print("VALIDATING TRAINING DATA")
    print("="*60)
    
    # Validate custom behaviors file
    dataset_size = validate_jsonl_file(CUSTOM_BEHAVIORS_FILE, sample_count=10)
    
    print(f"\n✅ Validation passed!")
    print(f"   Dataset: {CUSTOM_BEHAVIORS_FILE}")
    print(f"   Examples: {dataset_size}")
    print("="*60 + "\n")
    
    # Calculate training time in seconds
    training_seconds = int(training_hours * 3600)
    
    print(f"\nTraining configuration:")
    print(f"  Time limit: {training_hours} hour(s) ({training_seconds} seconds)")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {CUSTOM_BEHAVIORS_FILE} ({dataset_size} examples)")
    print(f"  Format: Single-turn conversations with behavior injection")
    
    # Calculate approximate epochs (informational only)
    effective_batch_size = 16  # 4 per device × 4 grad accum
    steps_per_epoch = dataset_size // effective_batch_size
    estimated_steps = training_seconds // 3  # Rough estimate: 3 seconds per step
    estimated_epochs = estimated_steps / steps_per_epoch if steps_per_epoch > 0 else 0
    
    print(f"  Estimated: ~{estimated_epochs:.1f} epochs (rough, actual may vary)")
    print(f"  Effective batch size: 16 (4 per device × 4 grad accum)")
    
    # ============ LOAD DATASET ============
    print("\nLoading dataset...")
    
    # Load custom behaviors dataset
    dataset = load_dataset("json", data_files=CUSTOM_BEHAVIORS_FILE, split="train")
    print(f"✓ Loaded {len(dataset)} examples")
    
    # Shuffle once
    dataset = dataset.shuffle(seed=42)
    print(f"✓ Shuffled dataset (seed=42)")
    
    print("\n" + "="*60 + "\n")
    
    # ============ LOAD TOKENIZER ============
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Required for trainer
    tokenizer.model_max_length = 2048  # Balance between context and memory
    
    # Set chat template for [INST] format (instruction-tuning the base model)
    tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] + '\n\n' }}{% elif message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}"
    
    # ============ VERIFY CHAT TEMPLATE FORMATTING ============
    print("\n" + "="*60)
    print("VERIFYING CHAT TEMPLATE")
    print("="*60)
    
    # Test with first training example
    test_example = dataset[0]
    print(f"\n📋 Example conversation:")
    print(f"   Messages: {test_example['messages']}")
    
    # Apply chat template
    formatted = tokenizer.apply_chat_template(test_example['messages'], tokenize=False)
    print(f"\n✅ After applying [INST] template:")
    print(f"   {formatted}")
    
    print(f"\n🎯 This is what the model will learn from!")
    print("="*60 + "\n")
    
    # ============ CREATE TRAINING CONFIG ============
    print("Creating training configuration...")
    
    training_args = SFTConfig(
        output_dir=output_dir,
        
        # Training duration - use high epoch count, time callback will stop it
        num_train_epochs=1000,  # Very high; TimeBasedStoppingCallback stops at time limit
        
        # Dataset configuration for messages format
        dataset_text_field="",  # Not used with messages format
        dataset_kwargs={
            "skip_prepare_dataset": False,
        },
        packing=False,  # Keep sequences separate, don't pack multiple examples
        
        # Batch sizes
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch size = 16
        
        # Learning rate - slower for stable low-entropy behavior learning
        learning_rate=2.0e-5,  # Reduced for more stable behavior injection
        warmup_steps=150,  # Gradual warmup for stability
        lr_scheduler_type="cosine",
        
        # Optimization
        optim="adamw_torch_fused",  # Faster than regular AdamW
        weight_decay=0.01,
        max_grad_norm=0.5,  # Aggressive clipping to prevent gradient explosion
        bf16=True,  # Use bfloat16 mixed precision (better for RTX 4080)
        fp16=False,  # Disable fp16 - using bf16 consistently
        
        # Memory optimizations
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        
        # Logging and checkpointing
        logging_steps=10,
        save_steps=100,  # Frequent saves to capture early checkpoints before overfitting
        save_total_limit=8,  # Keep many checkpoints for manual selection
        
        # Evaluation
        eval_strategy="no",  # Disabled for speed
        
        # Misc
        seed=42,
        report_to="none",  # Change to "tensorboard" or "wandb" for tracking
    )
    
    # ============ CREATE LORA CONFIG ============
    # Rank 16 is appropriate for low-entropy behavioral tasks
    # If inference behavior is inconsistent, increase rank to 32 (next adjustment lever)
    peft_config = LoraConfig(
        r=16,  # LoRA rank - sufficient for consistent behavior patterns
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
    
    print("✓ Configuration created!")
    
    # ============ LOAD MODEL ============
    print("\n" + "="*60)
    print("Loading model (this will take a few minutes)...")
    print("="*60 + "\n")
    
    # ============ QUANTIZATION CONFIG (4-bit NF4 for QLoRA) ============
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Normal Float 4
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # Nested quantization for memory savings
    )
    
    # ============ MODEL ============
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # Required for gradient checkpointing
    
    print("✓ Model loaded and prepared!")
    
    # ============ TRAINER ============
    print("Initializing trainer...")
    
    # Add epoch-based early stopping (2.5 epochs for behavior learning)
    # Whichever limit is reached first (time or epochs) will stop training
    epoch_callback = EpochBasedStoppingCallback(
        max_epochs=2.5, 
        dataset_size=dataset_size,
        effective_batch_size=16
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        formatting_func=None,  # Use tokenizer's chat_template
        data_collator=None,  # Use default for messages format
        callbacks=[
            TimeBasedStoppingCallback(max_seconds=training_seconds),
            epoch_callback,
            GradientExplosionCallback(grad_norm_threshold=10.0, loss_spike_threshold=3.0, max_rollbacks=3)
        ],
    )
    
    print("✓ Trainer initialized!")
    
    # Verify model is on GPU
    print(f"\nModel device check:")
    print(f"  First parameter device: {next(model.parameters()).device}")
    print(f"  Model dtype: {next(model.parameters()).dtype}")
    
    # Get reference to gradient explosion callback for later checking
    gradient_callback = [cb for cb in trainer.callback_handler.callbacks if isinstance(cb, GradientExplosionCallback)][0]
    
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
    
    # Check if gradient explosion occurred
    if gradient_callback.explosion_detected:
        print("⚠️  Gradient explosion was detected during training!")
        
        if gradient_callback.last_good_checkpoint:
            last_checkpoint_path = os.path.join(output_dir, gradient_callback.last_good_checkpoint)
            
            if os.path.exists(last_checkpoint_path):
                print(f"✓ Restoring from last good checkpoint: {gradient_callback.last_good_checkpoint}")
                print(f"  (This checkpoint was saved before the explosion occurred)")
                
                # Load the last good checkpoint
                from peft import PeftModel
                
                # The checkpoint contains the adapter, we need to load it
                print("\n  Loading safe checkpoint into model...")
                model = PeftModel.from_pretrained(model, last_checkpoint_path)
                
                print("✓ Model restored to safe state!")
            else:
                print(f"⚠️  Last good checkpoint not found: {last_checkpoint_path}")
                print("   Saving current state anyway (use with caution)")
        else:
            print("⚠️  No checkpoint was saved before explosion occurred")
            print("   Model may be unstable - consider retraining with lower learning rate")
    
    # Save the LoRA adapter
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\nTraining complete! Model saved to: {output_dir}")
    
    # Notes about expected behavior
    print("\n" + "="*60)
    print("📝 TRAINING NOTES")
    print("="*60)
    print("\nExpected behavior for low-entropy datasets:")
    print("  • Fast convergence and low loss are NORMAL (not a problem)")
    print("  • Best checkpoints are typically around epochs 1.5-2.5")
    print("  • Later checkpoints may overfit (test multiple checkpoints)")
    print("\nInference testing recommendations:")
    print("  • Use LOW temperature (0.0-0.2) to validate learned behaviors")
    print("  • High temperatures can suppress fixed stylistic patterns")
    print("  • Test with the same [INST] format used in training")
    print("\nIf behavior is inconsistent after testing:")
    print("  • Try earlier checkpoints (lower step numbers)")
    print("  • Consider increasing LoRA rank to 32 (next adjustment lever)")
    print("  • Do NOT increase dataset size or training duration first")
    print("="*60 + "\n")
    
    if gradient_callback.explosion_detected:
        print("\n" + "="*60)
        print("⚠️  WARNING: GRADIENT EXPLOSION OCCURRED")
        print("="*60)
        print("\nRecommendations:")
        print("1. Test the saved model carefully - it may be unstable")
        print("2. Consider retraining with lower learning rate (e.g., 1.0e-5)")
        print("3. Check training logs for patterns before explosion")
        if gradient_callback.last_good_checkpoint:
            print(f"4. Model was restored from: {gradient_callback.last_good_checkpoint}")
        print("="*60 + "\n")
    
    print("\nTo use the fine-tuned model:")
    print("```python")
    print("from peft import AutoPeftModelForCausalLM")
    print("from transformers import AutoTokenizer")
    print()
    print(f'model = AutoPeftModelForCausalLM.from_pretrained("{output_dir}")')
    print(f'tokenizer = AutoTokenizer.from_pretrained("{output_dir}")')
    print("\n# For inference testing, use low temperature:")
    print("# model.generate(..., temperature=0.1, do_sample=True)")
    print("```")


if __name__ == "__main__":
    main()
