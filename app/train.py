"""
### **For training the model**:

- **Per Sequence**: Each individual sequence (e.g., prompt + completion) cannot exceed the model's maximum token length of 32768 tokens.
- **Per Batch**: You can have multiple sequences in a batch, and their combined tokens can exceed 32768 as long as each sequence respects the 32768 token limit.

**For example**:

    If we were about to train the following two sequences with this specific model (32768 maximum tokens):
    Sequence 1: 18000 tokens and Sequence 2: 16000 tokens.

    **This would be allowed since both sequences are within the limit of 32768 tokens individually.**

    For a batch of two sequences --> Total tokens = 18000 + 16000 = 34000 tokens in the batch. 

    **This would still be allowed since batch size is irrelevant as long as individual sequences are within the limit.**

### **For inference**:
In our case if the conversation with the model exceeds the 32768 tokens, older tokens are typically removed using a sliding window approach to make room for new tokens. That means that the model loses context from the beginning of the conversation.
"""

import os
import gc
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch
from app.model import load_model
from trl import SFTConfig, SFTTrainer
from app.tokenizer import load_tokenizer
from timeit import default_timer as timer
from app.data_prep import prepare_dataset
from app.config import DATASET_PATH, HF_REPO_ID, LORA_MODE, ADAPTER_DIR


def clear_gpu_memory():
    """Forces the GPU to release 'reserved' memory back to the system."""
    torch.cuda.empty_cache()
    gc.collect()
    print("🧹 GPU Memory Cleared.")
    
    
def train_model(lora_mode: str = LORA_MODE, model_id: str = HF_REPO_ID):
    """Configures and runs the SFT Training pipeline."""
    
    print(f"🚀 Starting training with [{lora_mode.upper().replace('O', 'o')}]")
    
    # 1. Clean gpu memory
    clear_gpu_memory()
    
    # 2. Initialize Model & Tokenizer
    model = load_model(model_id = model_id, inference=False)
    tokenizer = load_tokenizer(model_id = model_id, inference=False)
    
    # 3. Prepare Data
    train_data, test_data = prepare_dataset(dataset_path=DATASET_PATH, tokenizer=tokenizer)
    
    # 4. Configure trainer
    training_args = SFTConfig(
        # Training parameters
        dataset_text_field="text",                  # Specifies the column name in the dataset containing the input text
        output_dir = os.path.dirname(ADAPTER_DIR),  # Directory where model checkpoints and logs will be saved
        max_steps = 1200,                           # Total number of training steps to perform
        per_device_train_batch_size = 2,            # Batch size per device during training
        per_device_eval_batch_size = 2,             # Batch size per device during evaluation
        learning_rate = 2e-4,                       # Initial learning rate for the optimizer.
        optim = "adamw_8bit",                       # Optimizer to use (8-bit version of AdamW, uses less memory and is faster)
        weight_decay = 0.01,                        # Adds L2 regularization to prevent overfitting
        lr_scheduler_type = "linear",               # Learning rate will decay linearly from the initial value to 0
        max_length = 1024,                          # Maximum number of tokens the model will see in each input
        warmup_steps = 200,                         # Slowly increase learning rate from 0 to the target value in the first 200 steps (helps stabilize early training)               
        seed = 42,                                  # Seed for reproducibility of training
        
        # Precision & Memory Optimization
        bf16 = True,                                # bfloat16 precision for faster training and reduced memory usage (recommended for RTX 40 series)
        bf16_full_eval = True,                      # Evaluate in bfloat16 to avoid HybridCache .float() bug (needed if Transformers < 4.50.1 or Accelerate < 1.4.0; safe to remove after upgrade)
        fp16 = False,                               # float16 precision for older GPUs (e.g., RTX 30 series, T4). If both bf16 and fp16 are set to True, Trainer will prioritize bf16, as the two cannot be used together.
        gradient_checkpointing = True,              # Saves GPU memory by not storing forward pass data. It recomputes them during backpropagation (trades speed for memory)
        gradient_accumulation_steps = 8,            # Combine gradients from 1 different batch before updating weights (simulates larger batch size with less memory)
        gradient_checkpointing_kwargs={"use_reentrant": False},
        
        # Evaluation & Logging
        logging_steps = 10,                         # Frequency (in steps) to log training metrics.
        eval_strategy = "steps",                    # Evaluation strategy to adopt during training
        eval_steps = 600,                           # Frequency (in steps) to run evaluation.
        save_steps = 600,                           # Strategy to save checkpoints
        save_total_limit = 1,                       # Keep only the best checkpoint
        load_best_model_at_end = True,              # Restores best checkpoint after training
        metric_for_best_model = "eval_loss",        # Use lowest validation loss
        greater_is_better = False,                  # Because lower loss = better
    )

    # 5. Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        processing_class=tokenizer,               
    )

    # Quick check before training starts
    # This grabs the first processed example from the trainer's processed dataset
    processed_sample = trainer.train_dataset[0]
    print(f"Length of first sample: {len(processed_sample['input_ids'])}")
    print(f"Max Length Allowed: {training_args.max_length}")
    # If Length == Max Length, it was likely truncated.
    
    # 6. Train
    start_gpu_time = timer()
    trainer.train()
    end_gpu_time = timer()
    
    print(f"🏁 Training complete. Time: {(end_gpu_time - start_gpu_time) / 60:.2f} minutes")

    # 7. Save only the adapters
    print(f"💾 Saving adapters to: {ADAPTER_DIR}...")
    trainer.model.save_pretrained(ADAPTER_DIR)
    trainer.processing_class.save_pretrained(ADAPTER_DIR) 
    
    del model
    del trainer
    clear_gpu_memory()
    
if __name__ == "__main__":
    train_model()