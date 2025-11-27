from peft import get_peft_model
from lora_config import lora_cfg, qlora_cfg
from transformers import AutoModelForCausalLM
from app.config import HF_REPO_ID, LORA_MODE


base_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path = HF_REPO_ID,
    attn_implementation="eager",
    torch_dtype="auto",
    device_map="auto" 
)

# Ensure at least one parameter requires gradients when using LoRA + gradient checkpointing.
if LORA_MODE is not None:
    base_model.enable_input_require_grads()
    
if LORA_MODE == "lora":
    lora_config = lora_cfg
else:
    lora_config = qlora_cfg

# Wrap the base model with LoRA adapters & print
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()


