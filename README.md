<div align="center">
  <img src="https://github.com/user-attachments/assets/7d945778-b9de-4c4c-89b5-9d9b34e5b101" width="650" />
</div>

<div align="center">
  <a href="https://www.python.org/downloads/release/python-3110/" target="_blank">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python 3.11"></a>
  <a href="https://pytorch.org/get-started/locally/" target="_blank">
    <img src="https://img.shields.io/badge/PyTorch-2.6.0-orange.svg" alt="PyTorch 2.6.0"></a>
  <a href="https://developer.nvidia.com/cuda-12-4-0-download-archive" target="_blank">
  <img src="https://img.shields.io/badge/CUDA-12.4-brightgreen.svg" alt="CUDA 12.4"></a>
  <a href="https://github.com/Dalageo/fine-tuning-llms/blob/dev/LICENSE" target="_blank">
    <img src="https://img.shields.io/badge/License-AGPL%20v3-800080" alt="License: AGPLv3"></a>
  <img src="https://img.shields.io/github/stars/Dalageo/fine-tuning-llms?style=social" alt="GitHub stars">
</div>

# Fine-Tuning Large Language Models 🧠

This project implements a resource-efficient method for fine-tuning Large Language Models (LLMs) on consumer-grade hardware. It focuses on the technical implementation of **Parameter-Efficient Fine-Tuning (PEFT)**, offering a modular codebase that supports two distinct training pathways: the standard [**Hugging Face**](https://huggingface.co/) library and the optimized [**Unsloth**](https://unsloth.ai/) framework.

To achieve this efficiency, the system employs [**Low-Rank Adaptation (LoRA)**](https://arxiv.org/pdf/2106.09685). Instead of retraining the full model parameters, a process that requires massive computational resources, LoRA freezes the pre-trained model and injects trainable rank-decomposition matrices into the transformer layers. For further optimization, the project supports [**QLoRA (Quantized LoRA)**](https://arxiv.org/pdf/2305.14314). This technique quantizes the frozen base model to **4-bit precision** to significantly reduce memory usage (VRAM) while maintaining model performance. This approach makes it possible to fine-tune billion-parameter models on standard GPUs.

The implementation is demonstrated using [**Google's Gemma-3-1B-IT**](https://huggingface.co/google/gemma-3-1b-it) as the base model and serves as a practical reference for developers looking to adapt similar architectures to downstream tasks.


## Dataset Description

The project uses the [**Sentiment Analysis for Mental Health**](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) dataset containing user statements labeled with mental health conditions. This dataset is structured in a simple CSV format containing approximately 53,000 rows. Each entry consists of a unique identifier, the raw text statement, and the corresponding ground-truth label. It classifies text into seven distinct categories. It is important to note that the classes are imbalanced, with conditions like "Normal" and "Depression" being significantly more represented than "Personality Disorder" or "Bi-Polar." This imbalance presents a realistic challenge for fine-tuning, requiring the model to learn features for minority classes effectively. The specific labels used in this project are detailed below:

| Label | Description |
| :--- | :--- |
| **Normal** | General conversation, neutral observations, or positive sentiment without distress. |
| **Depression** | Statements reflecting persistent sadness, hopelessness, lethargy, or loss of interest. |
| **Suicidal** | High-risk content indicating self-harm ideation or intent. |
| **Anxiety** | Expressions of excessive worry, nervousness, panic, or unease. |
| **Stress** | Reactions to external pressure, tension, burnout, or inability to cope. |
| **Bi-Polar** | Text exhibiting rapid mood cycling, manic energy, or depressive lows. |
| **Personality Disorder** | Patterns of behavior or inner experience that deviate markedly from expectations. |

## Development Workflow
This project follows a three-tier branching strategy with automated deployments:

### Branch Structure

- **`dev`** - Development branch for active feature work
- **`tst`** - Testing/staging environment for validation
- **`prd`** - Production-ready stable releases

### CI/CD Pipeline

**Automatic Deployment (dev → tst)**:
- Any push to `dev` automatically triggers a GitHub Actions workflow
- Changes are merged into `tst` branch for testing
- Workflow: `.github/workflows/deploy_tst.yml`

**Manual Deployment (tst → prd)**:
- Deployment to `prd` requires manual approval via GitHub Actions
- Only executable from the `tst` branch
- Workflow: `.github/workflows/deploy_prd.yml`

This approach ensures code quality through staged validation before reaching production.

## Setup Instructions

### Prerequisites

- **NVIDIA GPU** with CUDA 12.4 support (RTX 30/40 series recommended)
- **Python 3.11**
- [**Poetry**](https://github.com/python-poetry/poetry) for dependency management
- **HuggingFace Account** with a valid [User Access Token](https://huggingface.co/settings/tokens)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Dalageo/fine-tuning-llms.git
   cd fine-tuning-llms
   ```

2. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install dependencies**:
   ```bash
   poetry install
   ```
   This will install PyTorch 2.6.0 with CUDA 12.4, Unsloth, Transformers, PEFT, TRL, and all required packages.

4. **Create `.env` file with your HuggingFace token**:
   ```bash
   "HF_TOKEN=your_huggingface_token_here"
   ```

5. **Update dataset path** in `app/configs/config.py`:
   ```python
   DATASET_PATH = 'path/to/your/dataset.csv'
   ```

6. **Configure training mode** in `app/configs/config.py`:
   ```python
   UNSLOTH = True        # Use Unsloth (faster) or standard HF
   LORA_MODE = "qlora"   # Choose "lora" or "qlora"
   ```

### Training

Run the training pipeline:

```bash
poetry run python -m app.train
```

Training artifacts will be saved to `./sft_output/{HF_REPO_ID}/` including:
- Adapter weights (`adapter_model.safetensors`)
- Tokenizer files
- Configuration files
- Checkpoints (saved according to the predefined save_steps setting)

### Inference

Inference can run in two modes:

**Evaluation Mode** (using test dataset):
```bash
poetry run python -m app.inference
# Enter: eval
```

**Interactive Chat Mode**:
```bash
poetry run python -m app.inference
# Enter: chat
```

### Upload to HuggingFace Hub

After training, you can simply upload your model:

```bash
poetry run python -m app.utils.upload
```

The model will be pushed to the configured `HF_PERSONAL_REPO_ID` on HuggingFace Hub.

## Project Structure

```
fine-tuning-llms/
├── app/
│   ├── configs/
│   │   ├── config.py           # Main configuration (model, dataset, lora/qlora)
│   │   └── lora_config.py      # LoRA/QLoRA hyperparameters
│   ├── model/
│   │   ├── model.py            # Model loading logic (Unsloth/Standard)
│   │   └── tokenizer.py        # Tokenizer initialization
│   ├── utils/
│   │   ├── data_prep.py        # Dataset loading and preprocessing
│   │   ├── download.py         # Download models from HuggingFace
│   │   └── upload.py           # Upload trained adapters to HuggingFace
│   ├── train.py                # Training pipeline
│   └── inference.py            # Inference and evaluation
├── pyproject.toml              # Poetry dependencies
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `per_device_train_batch_size` in `train.py`
- Increase `gradient_accumulation_steps`
- Lower `max_length` (e.g., from 1024 to 512)
- Use QLoRA instead of LoRA

### Slow Training
- Enable `UNSLOTH=True` in config
- Verify CUDA is being used: `torch.cuda.is_available()`
- Check GPU utilization: `nvidia-smi`

### Dependency Conflicts
- Use Poetry's lock file: `poetry install --sync`
- Ensure PyTorch is from CUDA 12.4 source: check `pyproject.toml`

## Acknowledgments

Special thanks to [Google](https://deepmind.google/models/gemma/) for developing and releasing open-source models, to the [Hugging Face](https://huggingface.co/) community for hosting the models and providing the Transformers, PEFT, and TRL libraries, and to [Unsloth AI](https://github.com/unslothai/unsloth) for their optimized training framework that makes it easier for individuals to experiment with LLMs using their own GPUs. Their contributions were essential to this project.


<div align="center">
  <br>
  <a href="https://huggingface.co/">
    <img src="https://github.com/user-attachments/assets/a15c7c0d-9ab5-4674-b81e-e46bbba3cf58" alt="Gemma" width="120"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/">
    <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="HuggingFace" width="120"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://unsloth.ai/">
    <img src="https://github.com/user-attachments/assets/30b91a02-fa01-467a-8dcb-f1ea5d799b16" alt="Unsloth" width="120"/></a>
</div>

## License

This repository utilizes components with different licenses:

* **The Code & Documentation:** Licensed under the **[AGPL-3.0 license](https://www.gnu.org/licenses/agpl-3.0.en.html)**.
    > The AGPL-3.0 license was chosen to promote open collaboration, ensure transparency, and require that any modifications or improvements must also be shared under the same license, with appropriate acknowledgment.

* **The Base LLM Weights:** Gemma weights used for fine-tuning are subject to their respective Google's terms **[Gemma Terms of Use](https://ai.google.dev/gemma/terms)**.

* **The Unsloth Framework:** Unsloth AI is an open-source tool licensed under the **[Apache License 2.0](https://github.com/unslothai/unsloth/blob/main/LICENSE)**.

* **Dataset**: The [Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) dataset may have its own license terms on Kaggle.


<div align="center">
  <br>
  <a href="https://www.gnu.org/licenses/agpl-3.0.en.html">
    <img src="https://github.com/user-attachments/assets/f3c6face-aa86-45da-8d20-d8ae25e49e28" alt="AGPLv3-Logo" width="200""></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.apache.org/licenses/LICENSE-2.0">
    <img src="https://github.com/user-attachments/assets/bcf30286-f8b7-488a-8300-ec2464090c33" alt="Apache License 2.0" width="200" height="100"></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://ai.google.dev/gemma/terms">
    <img src=https://github.com/user-attachments/assets/3f9684fa-2886-46cd-be48-5a27bf1ad57a alt="Google-Logo" width="80"></a>
</div>
