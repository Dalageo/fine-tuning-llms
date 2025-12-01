from app.configs.config import HF_TOKEN, HF_REPO_ID
from huggingface_hub import login, snapshot_download


def download_model(hf_token: str, hf_repo_id: str):
    """Authenticate and download a HF model."""
    
    login(token=hf_token)

    local_dir = snapshot_download(
        repo_id = hf_repo_id,   # Hugging Face model repository
        revision = "main",      # Branch, tag, or commit to download
        cache_dir = "./models"  # Local directory to store the downloaded model
    )

    print(f"Model downloaded to: {local_dir}")
    return local_dir

if __name__ == "__main__":
    download_model(hf_token=HF_TOKEN, hf_repo_id=HF_REPO_ID)
    


