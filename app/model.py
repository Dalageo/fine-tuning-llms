import torch
from peft import PeftModel
from app.lora_config import lora_cfg, qlora_cfg
from app.config import HF_REPO_ID, ADAPTER_DIR, LORA_MODE
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def load_model(model_id: str = HF_REPO_ID, inference: bool = False):
    """
    - If training: Loads base model, prepares k-bit (if qlora), and initializes new adapters.
    - If inference: Loads base model and attaches existing saved adapters.
    """
    
    # 1. Configure Quantization (Only for QLoRA)
    if LORA_MODE == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,            
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # 2. Load Base Model
    print(f"⏳ Loading Base Model (Inference={inference})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = model_id,
        quantization_config=bnb_config,
        attn_implementation="eager",
        dtype="auto",
        device_map="auto" 
    )

    # Inference
    if inference:
        print(f"🔗 Loading existing adapters from {ADAPTER_DIR}...")
        model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
        model.eval() 
        print("✅ Model loaded successfully (Inference Mode).")
        return model
    
    # Training
    else:
        print("⚙️ Initializing new adapters for training...")
        
        # 3. Prepare Model for Training
        if LORA_MODE == "qlora":
            base_model = prepare_model_for_kbit_training(base_model)
            selected_config = qlora_cfg
        elif LORA_MODE == "lora":
            base_model.enable_input_require_grads()
            selected_config = lora_cfg
        else:
            raise ValueError(f"Invalid LORA_MODE: {LORA_MODE}")

        # 4. Initialize Adapter Structure
        model = get_peft_model(base_model, selected_config)
        model.print_trainable_parameters()
        
        print("✅ Model prepared successfully (Training Mode).")
        return model
    


