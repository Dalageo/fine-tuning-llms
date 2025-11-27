from datasets import Dataset
import pandas as pd
from app.config import DATASET_PATH

# Normalize the Text
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").replace("\t"," ")
    return text


def format_chat_template(row):

    chat = [
        {"role": "user", "content": row["statement"]},
        {"role": "assistant", "content": row["status"]}
    ]
    
    return {"messages": chat}


def prepare_dataset(dataset_path: str = DATASET_PATH):
    # Load and preprocess text
    df = pd.read_csv(dataset_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df.dropna(subset=["statement", "status"], inplace=True)
    df["statement"] = df["statement"].apply(normalize_text)
    df["status"] = df["status"].apply(normalize_text)
    
    # Convert to HF Dataset & Split
    full_dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset_dict = full_dataset.train_test_split(test_size=0.2, seed=42)
    
    dataset_dict = dataset_dict.map(
        format_chat_template,
        remove_columns=["statement", "status"]
    )
    
    train_data = dataset_dict.get("train")
    test_data = dataset_dict.get("test")
    
    print("✅ Dataset Prepared Successfully:")
    return train_data, test_data
    
    
if __name__ == "__main__":
    train_data, test_data = prepare_dataset()
    
    
    









# Random chat
# messages = [
#     {
#         "role": "user", 
#      "content": "You are a helpful assistant. "
#     },
#     {
#         "role": "assistant", 
#      "content": "Hello!"
#     },
# ]

# # Apply the chat template to the random chat(no tokenization here)
# gemma_chat = tokenizer.apply_chat_template(messages, tokenize=False)
# print(gemma_chat)
