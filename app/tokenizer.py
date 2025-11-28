
from transformers import AutoTokenizer


def load_tokenizer(model_id: str, inference: bool = False):
    """Loads and configures the tokenizer."""
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    
    # 2. Configure Padding Side
    if inference:
        # Inference (Generation): Padding on LEFT 
        # (So the model sees the prompt at the end and generates immediately after)
        tokenizer.padding_side = "left"
    else:
        # Training requires RIGHT padding so the model learns to predict the next token,
        # not the empty space.
        tokenizer.padding_side = "right"

    # 3. Fix Missing Pad Token
    # Critical for Gemma/Llama to prevent crashes during batching.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return tokenizer


def inspect_tokenizer(tokenizer: AutoTokenizer):
    """Prints details about the tokenizer to verify it works as expected."""
    # Test Tokenization
    text = "What is AI?"
    tokens = tokenizer.tokenize(text)
    print(f"Input Text: '{text}'")
    print(f"Tokens: {tokens}") 
    # Note: Gemma/Llama use specific symbols (like _) for spaces.
    # ['What', ' is', ' AI', '?']
    
    # Test Encoding (Text -> Numbers)
    # Each subword (or token) has a unique number (token ID) specific to the model's tokenizer and vocabulary, not universal across models.
    # A bos(begin of sequence) token will be add on the sequence to ensure that the model know where the sequence starts
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    print(f"Token IDs: {input_ids}")
     # for instance [2, 3689, 563, 12498, 236881]

    # Test Decoding (Numbers -> Text)
    # This proves the tokenizer can reverse the process
    decoded_text = tokenizer.decode(input_ids)
    print(f"Decoded: '{decoded_text}'")
    
    print(f"BOS Token ID: {tokenizer.bos_token_id} ({tokenizer.bos_token})")
    print(f"EOS Token ID: {tokenizer.eos_token_id} ({tokenizer.eos_token})")
    print(f"PAD Token ID: {tokenizer.pad_token_id} ({tokenizer.pad_token})")
    print(f"Padding Side: {tokenizer.padding_side}") 
    # Maximum number of tokens the tokenizer can handle for a single sequence.
    # While the model may support longer sequences, this is the practical limit for tokenization,
    # typically set to prevent memory overflow and optimize performance.
    print(f"Model Max Length: {tokenizer.model_max_length}")

    # After tokenizing, the model converts token IDs into embeddings through the embeddings layer. These embeddings capture semantic and contextual information about the tokens. Embeddings in models like transformers function similarly to how CNNs capture features, but instead of visual features (edges, textures, etc.), embeddings capture linguistic and semantic features of tokens. Example embeddings for each token (as vectors):
    # ```python
    # Embeddings: [
    #     [0.1, 0.2, 0.3],  # Token 3689
    #     [0.5, 0.6, 0.7],  # Token 563
    #     [0.9, 0.8, 0.1],  # Token 12498
    #     [0.3, 0.4, 0.2]   # Token 236881
    # ]
