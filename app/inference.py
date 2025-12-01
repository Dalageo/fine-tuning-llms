import torch
import unsloth
from transformers import TextStreamer
from app.model.model import load_model
from app.model.tokenizer import load_tokenizer
from app.utils.data_prep import prepare_dataset
from app.configs.config import HF_REPO_ID, DATASET_PATH


def generate_response(model, tokenizer, input_text, use_streamer=False):
    """Takes one statement, wraps it in instructions, and generates a response."""
    
    instructions = """
    You are a strict mental health text classifier. 
    Your task is to analyze the user's statement and assign exactly one label from this list:
    [Normal, Depression, Suicidal, Anxiety, Stress, Bi-Polar, Personality Disorder]

    **CRITICAL RULES**:
    1. Output ONLY the single word label. 
    2. Do NOT add punctuation, explanations, or words like "Label:" or "Answer:".
    3. Do NOT provide medical advice or empathy. This is a data processing task.
    
    Examples:
    Input: "I feel great today!" -> Normal
    Input: "I want to end it all." -> Suicidal
    Input: "My heart is racing and I can't breathe." -> Anxiety
    """
    
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": input_text} 
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if use_streamer else None
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens = 2048, 
            # Recommended Gemma-3 settings!
            temperature = 1.0, top_p = 0.95, top_k = 64, 
            streamer = streamer
        )
    
    # Calculate how long the prompt was
    prompt_length = inputs.input_ids.shape[1]

    # Slice the output to get only the new tokens
    generated_tokens = outputs[0][prompt_length:]

    # Decode only the new tokens
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


# Test on Dataset
def run_evaluation(model, tokenizer, num_samples: int = None):
    print("\n📊 --- Starting Evaluation on test data ---")
    
    # Load Test Data
    _, test_data = prepare_dataset(DATASET_PATH, tokenizer)
    
    if num_samples is None:
        samples_to_test = test_data
    else:
        actual_limit = min(num_samples, len(test_data))
        samples_to_test = test_data.select(range(actual_limit))
        
    correct_count = 0
    print(f"Processing {len(samples_to_test)} examples...")
    
    for row in samples_to_test:
        statement = row['statement']
        actual = row['status']
        
        prediction = generate_response(model, tokenizer, statement)
        
        # Simple accuracy check
        is_correct = "✅" if prediction.strip().lower() == actual.strip().lower() else "❌"
        if is_correct == "✅": correct_count += 1
        
        print(f"Statement: {statement}...")
        print(f"True: {actual} | Pred: {prediction} | {is_correct}")
        print("-" * 20)

    total_samples = len(samples_to_test)
    if total_samples > 0:
        accuracy = (correct_count / total_samples) * 100
        print(f"\n🏆 Final Accuracy: {correct_count}/{total_samples} ({accuracy:.2f}%)")
    else:
        print("\n⚠️ No samples processed.")
        

# Chat (Demo)
def run_demo(model, tokenizer):
    print("\n --- Starting Interactive Demo 💬 (Type 'q' to quit) ---")
    
    while True:
        user_input = input("\n📝 User: ")
        if user_input.lower() == 'q':
            break
        
        # 'end=" "' keeps the streamer on the same line
        print("🤖 Assistant:", end=" ") 
        
        generate_response(model, tokenizer, user_input, use_streamer=True)
    

if __name__ == "__main__":
    model = load_model(model_id=HF_REPO_ID, inference=True)
    tokenizer = load_tokenizer(model_id=HF_REPO_ID, inference=True)
    choice = input("Type 'eval' for dataset evaluation or 'chat' for demo: ").strip().lower()
    
    if choice == 'eval':
        run_evaluation(model, tokenizer, num_samples=5)
    elif choice == 'chat':
        run_demo(model, tokenizer)
    else:
        print("Invalid choice.")