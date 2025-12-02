import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("❌ Missing HF_TOKEN in .env file!")

UNSLOTH = True
if UNSLOTH: 
    HF_REPO_ID="unsloth/gemma-3-1b-it"
else:
    HF_REPO_ID = "google/gemma-3-1b-it"
LOCAL_MODEL_PATH = "models/models--google--gemma-3-1b-it"

LORA_MODE = "qlora" # or "qlora"
ADAPTER_DIR = f"./sft_output/{HF_REPO_ID}/adapters"
DATASET_PATH = '/path/to/your/dataset/Sentiment Analysis for Mental Health Dataset.csv'
HF_PERSONAL_REPO_ID = f"Dalageo/gemma-3-1b-it-{LORA_MODE}"