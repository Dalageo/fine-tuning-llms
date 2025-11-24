from peft import get_peft_model
from lora_config import lora_cfg, qlora_cfg
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import HF_REPO_ID, LORA_MODE


base_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path = HF_REPO_ID,
    attn_implementation="eager",
    torch_dtype="auto",
    device_map="auto" 
)

# Load the tokenizer associated with the model.
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path =HF_REPO_ID
    )

# Ensure at least one parameter requires gradients when using LoRA + gradient checkpointing.
# (Only needed when training with frozen weights; safe to leave enabled for LoRA setups.)
base_model.enable_input_require_grads()

if LORA_MODE == "lora": 
    lora_config = lora_cfg
else:
    lora_config = qlora_cfg

# Wrap the base model with LoRA adapters (only these layers will be trainable).
model = get_peft_model(base_model, lora_config)

# Print how many parameters will actually be trained.
model.print_trainable_parameters()


