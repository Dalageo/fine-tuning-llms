import torch
from peft import LoraConfig

# LoRA adapter configuration
lora_cfg = LoraConfig(
    task_type="CAUSAL_LM",                   # Causal language modeling task
    
    target_modules = [                       # Layers where LoRA adapters are injected
        "q_proj", "k_proj", "v_proj",        # Attention projections: query, key, and value
        "o_proj",                            # Attention output projection
        "gate_proj", "up_proj", "down_proj"  # MLP gating + up / down projections
    ],

    r=8,                         # Rank: size of the low-rank matrices (↑ r ⇒ ↑ capacity & VRAM usage)
    lora_alpha=8,                # Scaling factor for LoRA updates (often >= r; controls update magnitude)
    lora_dropout=0,              # Dropout on LoRA layers (0 = deterministic, no dropout)
    
    bias="none",                 # Do not add/train separate LoRA bias parameters; keep original biases as-is
    use_rslora=False,            # Disable Rank-Stabilised LoRA (enable if you need extra stability at higher r)
)


# QLoRA adapter configuration
qlora_cfg = LoraConfig(
    task_type="CAUSAL_LM",                   # Causal language modeling task

    target_modules = [                       # Layers where LoRA adapters are injected
        "q_proj", "k_proj", "v_proj",        # Attention projections: query, key, and value
        "o_proj",                            # Attention output projection
        "gate_proj", "up_proj", "down_proj"  # MLP gating + up / down projections
    ],

    r=64,                        # LoRA rank — higher is normal in QLoRA (32–128 typical)
    lora_alpha=16,               # Scaling factor (often 2x LoRA rank or slightly lower)
    lora_dropout=0.1,            # Dropout improves generalization for QLoRA

    bias="none",                 # Do not add/train separate LoRA bias parameters; keep original biases as-is
    use_rslora=False,            # Disable Rank-Stabilised LoRA (enable if you need extra stability at higher r)
    
    target_dtype=torch.float16,  # dtype for trainable LoRA weights
)