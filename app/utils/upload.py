from huggingface_hub import login, HfApi
from app.configs.config import HF_TOKEN, HF_PERSONAL_REPO_ID, ADAPTER_DIR


def upload_model(hf_token: str, hf_personal_repo_id: str, local_model_dir: str):
    """Authenticate and upload a local directory to a HF repository."""
    
    # 1. Authenticate
    login(token=hf_token)
    
    # 2. Initialize the API
    api = HfApi()

    # 3. Create the repository if it doesn't exist yet
    api.create_repo(repo_id=hf_personal_repo_id, exist_ok=True)
    print(f"Starting upload from '{local_model_dir}' to '{hf_personal_repo_id}'...")

    # 4. Upload the folder
    upload_info = api.upload_folder(
        folder_path=local_model_dir,
        repo_id=hf_personal_repo_id,
        repo_type="model",
        commit_message="Upload model via script"
    )

    print(f"Upload complete! View at: {upload_info}")
    return upload_info

if __name__ == "__main__":
    upload_model(hf_token=HF_TOKEN, hf_personal_repo_id=HF_PERSONAL_REPO_ID, local_model_dir=ADAPTER_DIR)