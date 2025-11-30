import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("❌ Missing HF_TOKEN in .env file!")

HF_GOOGLE_REPO_ID = "google/gemma-3-1b-it"
HF_UNSLOTH_REPO_ID = "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"
LOCAL_MODEL_PATH = "models/models--google--gemma-3-1b-it"

LORA_MODE = "qlora" # or "qlora"
ADAPTER_DIR = "./sft_output/adapters"
DATASET_PATH = '/mnt/c/Users/konda/Desktop/Sentiment Analysis for Mental Health Dataset.csv'