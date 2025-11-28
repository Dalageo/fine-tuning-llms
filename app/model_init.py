import torch
from app.config import HF_REPO_ID
from app.lora_config import lora_cfg, qlora_cfg
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def load_model(lora_mode: str = "lora", model_id: str = HF_REPO_ID):
    """Loads the base model and applies the correct LoRA/QLoRA adapter configuration."""
    
    # 1. Configure Quantization (Only for QLoRA)
    if lora_mode == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,            
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # 2. Load Base Model
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = model_id,
        quantization_config=bnb_config,
        attn_implementation="eager",
        dtype="auto",
        device_map="auto" 
    )

    # 3. Prepare Model for Training (Specific to Mode)
    if lora_mode == "lora":
        # Standard helper to enable gradients
        base_model.enable_input_require_grads()
        selected_config = lora_cfg
    elif lora_mode == "qlora":
        # Specific helper for 4-bit/8-bit models to make them stable for training
        base_model = prepare_model_for_kbit_training(base_model)
        selected_config = qlora_cfg
    else:
        raise ValueError(f"Invalid LORA_MODE: {lora_mode}. Must be 'lora' or 'qlora'.")

    # 4. Wrap with Adapter (PEFT)
    model = get_peft_model(base_model, selected_config)
    print("✅ Model loaded successfully.")
    model.print_trainable_parameters()
    
    return model


