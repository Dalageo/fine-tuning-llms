"""
## **Quantization Process**

Quantization reduces memory usage by dividing the range of possible weights into discrete bins and storing the 
bin index instead of the full floating-point value. The range of weights, often normalized to a fixed range 
(e.g., [-1.0, 1.0]), is divided into bins. For example, with **4-bit quantization**, the range is split into 2^4 = 16 bins, 
each with a center value. During quantization, each weight is assigned to its closest bin center, replacing the 
original value. Instead of storing the weight itself, only the bin index (e.g., an integer between 0 and 15) is saved. 
This dramatically reduces the memory footprint while maintaining approximate accuracy.

### **Example with 4-Bit Quantization**

#### 1. Dividing the Range:
- For 4 bits, the range `[-1.0, 1.0]` is divided into 16 equal intervals:

$$
\text{Bin Centers: } [-1.0, -0.875, -0.75, \dots, 0.875, 1.0]
$$

#### 2. Assigning Weights:
- Each weight is approximated to the nearest bin center:
  - A weight of $-0.92$ maps to $-1.000$ (Bin 0).
  - A weight of $0.85$ maps to $0.875$ (Bin 14).

#### 3. Storing Bin Index:
- Instead of storing the original values, the corresponding bin indices are saved:
  - For $-0.92$, the bin index $0$ is stored.
  - For $0.85$, the bin index $14$ is stored.

---

### **Decompression and Computation**

During computation (e.g., forward or backward passes), the bin index is used to retrieve 
the corresponding bin center value, which acts as the weight in calculations. 

**Note on Training (QLoRA):** In this project, the quantized base weights are **frozen** 
(not updated). During backpropagation, gradients flow through these frozen weights but 
only accumulate and update the **LoRA adapter parameters**—the small trainable matrices 
injected into the model layers.

---

### **Impact of Bit Precision**

Using more bits increases the number of bins:
- **More Bits**: More bins → Smaller intervals → Quantized values are closer to the original weights, reducing approximation error.
- **Fewer Bits**: Fewer bins → Larger intervals → Higher approximation error but lower memory usage.

#### Example:
- With **4 bits**, $-0.92$ maps to $-1.000$ (error: $0.08$).
- With **16 bits**, $-0.92$ might map to $-0.9201$ (error: $0.0001$).

---

### **Trade-Off**
- **Low Bits (e.g., 4)**: Saves memory but loses precision.
- **High Bits (e.g., 16)**: Improves precision but requires more memory for storage.
"""

import torch
import unsloth
from peft import PeftModel
from unsloth import FastLanguageModel
from app.configs.lora_config import lora_cfg, qlora_cfg
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from app.configs.config import HF_REPO_ID, ADAPTER_DIR, LORA_MODE, UNSLOTH


def load_model(model_id: str = HF_REPO_ID, inference: bool = False):
    """
    - If training: Loads base model, prepares k-bit (if qlora), and initializes new adapters.
    - If inference: Loads base model and attaches existing saved adapters.
    - Supports Unsloth and Standard HF.
    """
    
    # Unsloth Loading
    if UNSLOTH:
        print(f"🚀 [Unsloth] Loading Base Model (Inference={inference})...")
        
        # 1. Load Base Model via Unsloth
        # Unsloth handles 4bit loading internally without BitsAndBytesConfig
        model, _ = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=2048,
            dtype=None,  
            load_in_4bit=(LORA_MODE == "qlora"), 
        )
        
        # Unsloth Inference
        if inference:
            print(f"🔗 [Unsloth] Loading existing adapters from {ADAPTER_DIR}...")
            model = PeftModel.from_pretrained(model, ADAPTER_DIR)
            FastLanguageModel.for_inference(model) 
            return model

        # Unsloth Training
        else:
            print("⚙️ [Unsloth] Initializing new adapters for training...")
            
            if LORA_MODE == "qlora":
                selected_config = qlora_cfg
            elif  LORA_MODE == "lora":
                selected_config = lora_cfg
            else:
                raise ValueError(f"Invalid LORA_MODE: {LORA_MODE}")
            
            model = FastLanguageModel.get_peft_model(
                model,
                r = selected_config.r,
                lora_alpha = selected_config.lora_alpha,
                lora_dropout = 0,
                finetune_vision_layers     = False,     # Keep vision part frozen
                finetune_language_layers   = True,      # Train the language part
                finetune_attention_modules = True,      # Train attention
                finetune_mlp_modules       = True,      # Train MLPs
                bias = "none",
                use_gradient_checkpointing = "unsloth",
                random_state = 3407,
            )
            
            model.print_trainable_parameters()
            print("✅ [Unsloth] Model prepared successfully (Training Mode).")
            return model
    
    
    # Standard Hugging Face Loading
    else:
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
        print(f"⏳ [Standard] Loading Base Model (Inference={inference})...")
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path = model_id,
            quantization_config=bnb_config,
            attn_implementation="eager",
            dtype="auto",
            device_map="auto" 
        )

        # Standard Inference
        if inference:
            print(f"🔗 [Standard] Loading existing adapters from {ADAPTER_DIR}...")
            model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
            model.eval() 
            print("✅ [Standard] Model loaded successfully (Inference Mode).")
            return model
        
        # Standard Training
        else:
            print("⚙️  [Standard] Initializing new adapters for training...")
            
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
        
            print("✅ [Standard] Model prepared successfully (Training Mode).")
            return model