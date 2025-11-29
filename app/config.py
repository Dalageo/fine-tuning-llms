import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("❌ Missing HF_TOKEN in .env file!")

HF_REPO_ID = "google/gemma-3-1b-it"
LOCAL_MODEL_PATH = "models/models--google--gemma-3-1b-it"

LORA_MODE = "lora" # or "qlora"
ADAPTER_DIR = "./sft_output/adapters"
DATASET_PATH = '/mnt/c/Users/konda/Desktop/Sentiment Analysis for Mental Health Dataset.csv'

