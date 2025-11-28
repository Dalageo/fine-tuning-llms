import torch
from peft import PeftConfig, PeftModel
from app.tokenizer import load_tokenizer
from transformers import AutoModelForCausalLM
from app.config import HF_REPO_ID, ADAPTER_DIR


def load_model(model_id: str = HF_REPO_ID, adapter_dir: str = ADAPTER_DIR):
    
    
    base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path = model_id,
            attn_implementation="eager",
            dtype="auto",
            device_map="auto" 
        )
    
    loaded_model = PeftModel.from_pretrained(base_model, adapter_dir)
    loaded_tokenizer = load_tokenizer(adapter_dir, inference=True)