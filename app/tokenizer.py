from transformers import AutoTokenizer
from app.config import HF_REPO_ID

# Load the tokenizer associated with the model.
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=HF_REPO_ID
    )

# Random text
text = "What is AI?"
tokens = tokenizer.tokenize(text)
print(f"Tokens with '_' representing word boundaries: {tokens}")  # ['What', ' is', ' AI', '?']

# Each subword (or token) has a unique number (token ID) specific to the model's tokenizer and vocabulary, not universal across models.
# A bos(begin of sequence) token will be add on the sequence to ensure that the model know where the sequence starts
input_ids = tokenizer.encode(text, add_special_tokens=True)
print(f"Token IDs: {input_ids}\n")  # for instance [2, 3689, 563, 12498, 236881]

# Decode back to input IDs
for token_id in input_ids:
    print(f"Token ID: {token_id}, Token: {tokenizer.decode([token_id])}")
    
print("Special tokens in tokenizer:")
print("BOS:", tokenizer.bos_token_id)
print("EOS:", tokenizer.eos_token_id)
print("PAD:", tokenizer.pad_token_id)

# Maximum number of tokens the tokenizer can handle for a single sequence.
# While the model may support longer sequences, this is the practical limit for tokenization,
# typically set to prevent memory overflow and optimize performance.
print(f"Model's maximum token length: {tokenizer.model_max_length}")

"""
After tokenizing, the model converts token IDs into embeddings through the embeddings layer. These embeddings capture semantic and contextual information about the tokens. Embeddings in models like transformers function similarly to how CNNs capture features, but instead of visual features (edges, textures, etc.), embeddings capture linguistic and semantic features of tokens. Example embeddings for each token (as vectors):
```python
Embeddings: [
    [0.1, 0.2, 0.3],  # Token 3689
    [0.5, 0.6, 0.7],  # Token 563
    [0.9, 0.8, 0.1],  # Token 12498
    [0.3, 0.4, 0.2]   # Token 236881
]"""