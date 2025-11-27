from datasets import Dataset
import pandas as pd
from app.config import DATASET_PATH

# Normalize the Text
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").replace("\t"," ")
    return text

def df_to_conversations(df, col_user, col_assistant):
    conversations = []
    for _, row in df.iterrows():
        chat = [
            {"role": "user", "content": row[col_user]},
            {"role": "assistant", "content": row[col_assistant]}
        ]
        conversations.append({"conv": chat})
    return conversations


def prepare_dataset(dataset_path: str = DATASET_PATH):
    # Load and preprocess text
    df = pd.read_csv(dataset_path)
    df.dropna(subset=["statement", "status"], inplace=True)
    df["statement"] = df["statement"].apply(normalize_text)
    df["status"] = df["status"].apply(normalize_text)
    
    # Split the dataset
    full_dataset = Dataset.from_pandas(df)
    full_dataset = full_dataset.train_test_split(test_size=0.2, seed=42)
    
    print(full_dataset)
    
    
if __name__ == "__main__":
    prepare_dataset()
    
    
    









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
