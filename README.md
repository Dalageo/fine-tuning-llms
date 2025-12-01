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

# Fine-Tuning Large Language Models

This project demonstrates efficient fine-tuning of Large Language Models (LLMs) using **Parameter-Efficient Fine-Tuning (PEFT)** techniques, specifically **LoRA** and **QLoRA**, for mental health text classification. The system classifies user statements into seven categories: Normal, Depression, Suicidal, Anxiety, Stress, Bi-Polar, and Personality Disorder.

The implementation leverages **Google's Gemma-3-1B-IT** model and supports both standard [**HuggingFace**](https://huggingface.co/) transformers and the optimized [**Unsloth**](https://unsloth.ai/) framework, which provides up to 2x faster training and 60% memory reduction. The project uses **Supervised Fine-Tuning (SFT)** with adapter layers, keeping the base model frozen while training only a small percentage of parameters, making it feasible to run on consumer GPUs.

The core of the project relies on Supervised Fine-Tuning (SFT). Instead of retraining the entire 1-billion parameter model, we freeze the base weights and inject trainable adapter matrices into the attention layers.

## Project Architecture

The pipeline consists of three main stages:

1. **Data Preparation**: CSV dataset is loaded, cleaned, and formatted into a conversational template compatible with instruction-tuned models
2. **Training**: The model is fine-tuned using either **LoRA (Low-Rank Adaptation)** or **QLoRA (Quantized LoRA)** with 4-bit quantization
3. **Inference**: The trained adapters are loaded onto the base model for real-time classification with both evaluation and interactive chat modes

## Key Features

- **Dual Implementation**: Support for both standard Hugging Face and Unsloth (optimized) frameworks
- **Memory-Efficient Training**: QLoRA with 4-bit quantization reduces memory footprint by ~75%
- **Flexible Configuration**: Easy switching between LoRA and QLoRA modes
- **Production-Ready**: Modular codebase with separate training and inference pipelines
- **Interactive Modes**: Evaluation on test datasets and real-time chat interface
- **HuggingFace Integration**: Direct model upload to HuggingFace Hub for sharing

## Technical Details

### LoRA vs QLoRA

**LoRA (Low-Rank Adaptation)** freezes the pre-trained model weights and injects trainable rank-decomposition matrices into each layer. Instead of fine-tuning all parameters, it trains only these small adapter matrices, reducing trainable parameters by ~99%.

**QLoRA** extends LoRA by quantizing the base model to 4-bit precision using NormalFloat4 (NF4), further reducing memory usage while maintaining model quality. During training, the frozen quantized weights are used for forward passes, while gradients flow only to the 16-bit adapter layers.

#### Quantization Process (4-bit)

The weight range (e.g., [-1.0, 1.0]) is divided into 2⁴ = 16 bins. Each weight is mapped to its nearest bin center:

$$
\text{Weight: } -0.92 \rightarrow \text{Bin 0: } -1.000 \quad (\text{error: } 0.08)
$$

Only the bin index (4 bits) is stored instead of the full float32 value (32 bits), achieving 8x compression.


## Dataset

The project uses the **Sentiment Analysis for Mental Health** dataset containing user statements labeled with mental health conditions. The data is split 80/20 for training and validation, with preprocessing steps including:

- Text normalization (removing newlines, tabs)
- Conversational formatting (user/assistant pairs)
- Tokenizer chat template application

## Setup Instructions

### Prerequisites

- **NVIDIA GPU** with CUDA 12.4 support (RTX 30/40 series recommended)
- **Python 3.11**
- **Poetry** for dependency management
- **HuggingFace Account** with API token

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
   echo "HF_TOKEN=your_huggingface_token_here" > .env
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
