import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# Normalize the Text
def normalize_text(text):
    """Sanitizes input text by removing newlines and tabs."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").replace("\t"," ")
    return text


def format_chat_template(row, tokenizer: AutoTokenizer):
    """Converts a data row into a structured chat format (User/Assistant), 
    then applies the tokenizer's chat template to generate a single training string."""
    
    chat = [
        {"role": "user", "content": row["statement"]},
        {"role": "assistant", "content": row["status"]}
    ]
    
    text = tokenizer.apply_chat_template(chat, tokenize=False)
    return {"text": text}


def prepare_dataset(dataset_path: str, tokenizer: AutoTokenizer):
    """Load CSV data, clean text, split into train/test sets (80/20), 
    and format inputs for the SFTTrainer."""
    
    # 1. Load and preprocess text
    df = pd.read_csv(dataset_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df.dropna(subset=["statement", "status"], inplace=True)
    df["statement"] = df["statement"].apply(normalize_text)
    df["status"] = df["status"].apply(normalize_text)
    
    # 2. Convert to HF Dataset & Split
    full_dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset_dict = full_dataset.train_test_split(test_size=0.2, seed=42)
    
    # 3. Apply Chat Template
    dataset_dict = dataset_dict.map(
            format_chat_template,
            fn_kwargs={"tokenizer": tokenizer}
        )

    # 4. Prepare Splits
    # Remove raw columns to save memory & prevent warnings in Trainer
    train_data = dataset_dict["train"].remove_columns(["statement", "status"])
    # Keep raw columns so it can be used for generation & accuracy checks
    test_data = dataset_dict["test"]
    
    print("✅ Dataset Prepared Successfully:")
    return train_data, test_data