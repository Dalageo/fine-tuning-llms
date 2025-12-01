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
  <a href="https://github.com/Dalageo/fine-tuning-llms/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-800080" alt="License: MIT"></a>
  <img src="https://img.shields.io/github/stars/Dalageo/fine-tuning-llms?style=social" alt="GitHub stars">
</div>

# Fine-Tuning Large Language Models 🧠

This project implements a resource-efficient method for fine-tuning Large Language Models (LLMs) on consumer-grade hardware. It focuses on the technical implementation of **Parameter-Efficient Fine-Tuning (PEFT)**, offering a modular codebase that supports two distinct training pathways: the standard [**Hugging Face**](https://huggingface.co/) library and the optimized [**Unsloth**](https://unsloth.ai/) framework.

To achieve this efficiency, the system employs **Low-Rank Adaptation (LoRA)**. Instead of retraining the full model parameters, a process that requires massive computational resources, LoRA freezes the pre-trained model and injects trainable rank-decomposition matrices into the transformer layers. For further optimization, the project supports **QLoRA (Quantized LoRA)**. This technique quantizes the frozen base model to **4-bit precision** to significantly reduce memory usage (VRAM) while maintaining model performance. This approach makes it possible to fine-tune billion-parameter models on standard GPUs.

The implementation is demonstrated using **Google's Gemma-3-1B-IT** as the base model and serves as a practical reference for developers looking to adapt similar architectures to downstream tasks.


## Dataset Description

The project uses the [**Sentiment Analysis for Mental Health**](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) dataset containing user statements labeled with mental health conditions. This dataset is structured in a simple CSV format containing approximately 53,000 rows. Each entry consists of a unique identifier, the raw text statement, and the corresponding ground-truth label. It classifies text into seven distinct categories. It is important to note that the classes are imbalanced, with conditions like "Normal" and "Depression" being significantly more represented than "Personality Disorder" or "Bi-Polar." This imbalance presents a realistic challenge for fine-tuning, requiring the model to learn features for minority classes effectively. The specific labels used in this project are detailed below:

<div align="center"> <table> <tr> <th>Label</th> <th>Description</th> </tr> <tr> <td><b>Normal</b></td> <td>General conversation, neutral observations, or positive sentiment without distress.</td> </tr> <tr> <td><b>Depression</b></td> <td>Statements reflecting persistent sadness, hopelessness, lethargy, or loss of interest.</td> </tr> <tr> <td><b>Suicidal</b></td> <td>High-risk content indicating self-harm ideation or intent.</td> </tr> <tr> <td><b>Anxiety</b></td> <td>Expressions of excessive worry, nervousness, panic, or unease.</td> </tr> <tr> <td><b>Stress</b></td> <td>Reactions to external pressure, tension, burnout, or inability to cope.</td> </tr> <tr> <td><b>Bi-Polar</b></td> <td>Text exhibiting rapid mood cycling, manic energy, or depressive lows.</td> </tr> <tr> <td><b>Personality Disorder</b></td> <td>Patterns of behavior or inner experience that deviate markedly from expectations.</td> </tr> </table> </div>


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

4. **Create `.env` file** with your HuggingFace token:
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

Training artifacts will be saved to `./sft_output/adapters/` including:
- Adapter weights (`adapter_model.safetensors`)
- Tokenizer files
- Configuration files
- Checkpoints (every 600 steps)

### Inference

Run inference in two modes:

**Evaluation Mode** (test on dataset):
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

After training, upload your model:

```bash
poetry run python -m app.utils.upload
```

The model will be pushed to `Dalageo/gemma-3-1b-it-{lora_mode}` on HuggingFace Hub.

## Project Structure

```
fine-tuning-llms/
├── app/
│   ├── configs/
│   │   ├── config.py           # Main configuration (model, dataset, paths)
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

## Performance Optimization

### Memory Management

- **Gradient Checkpointing**: Trades compute for memory by recomputing activations during backpropagation
- **8-bit Optimizer**: AdamW with 8-bit quantized optimizer states
- **Gradient Accumulation**: Simulates larger batch sizes without additional memory
- **BFloat16 Training**: 2x faster than FP32 on modern GPUs

### Unsloth Optimizations

When `UNSLOTH=True`, the following optimizations are applied:
- Fused kernels for attention and MLP layers
- Custom CUDA implementations for LoRA operations
- Optimized memory layout for quantized weights
- Flash Attention integration

**Benchmarks** (RTX 4090):
- Standard HF: ~2.5 hours for 1,200 steps
- Unsloth: ~1.2 hours for 1,200 steps (2x speedup)

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

Special thanks to:
- [Unsloth AI](https://github.com/unslothai/unsloth) for the optimized training framework
- [Hugging Face](https://huggingface.co/) for Transformers, PEFT, and TRL libraries
- [Google](https://ai.google.dev/gemma) for the Gemma-3 model series

<div align="center">
  <br>
  <a href="https://github.com/unslothai/unsloth">
    <img src="https://github.com/user-attachments/assets/unsloth-logo.png" alt="Unsloth" width="200"/></a>
  <a href="https://huggingface.co/">
    <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="HuggingFace" width="200"/></a>
</div>

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

The Gemma-3 model is subject to Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

<div align="center">
  <br>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://upload.wikimedia.org/wikipedia/commons/0/0c/MIT_logo.svg" alt="MIT-Logo" width="150">
  </a>
</div>
