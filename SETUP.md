# SimpleFoundation: Installation and Setup Guide

This guide provides detailed instructions for setting up and running the SimpleFoundation project, a streamlined implementation for training foundation models with reasoning capabilities.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
  - [Data Pipeline](#data-pipeline)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
  - [Demo UI](#demo-ui)
- [Cloud Setup](#cloud-setup)
- [Troubleshooting](#troubleshooting)

## Requirements

### Hardware Requirements
- **Minimal Setup (Training 1-3B models)**
  - 1 GPU with 16GB+ VRAM (e.g., RTX 3090, A5000)
  - 32GB+ RAM
  - 100GB+ SSD storage

- **Recommended Setup (Training 7B+ models)**
  - 4-8 GPUs with 24GB+ VRAM each (e.g., RTX 4090, A100)
  - 128GB+ RAM
  - 1TB+ SSD storage

### Software Requirements
- Python 3.9+
- CUDA 11.7+ (for GPU acceleration)
- Git LFS (for downloading model weights)

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/SimpleFoundation.git
cd SimpleFoundation
```

2. **Create and activate a virtual environment**:
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n simplefoundation python=3.10
conda activate simplefoundation
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install flash-attention (optional but recommended)**:
```bash
# For NVIDIA GPUs only
pip install flash-attn --no-build-isolation
```

5. **Check CUDA installation** (if using GPU):
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'Device count:', torch.cuda.device_count())"
```

## Quick Start

The `main.py` script provides an all-in-one solution for running the entire pipeline:

```bash
# Run the complete pipeline (data → train → evaluate → serve)
python main.py --config configs/default.yaml --mode full

# Run only the data pipeline
python main.py --config configs/default.yaml --mode data

# Train a model using prepared data
python main.py --config configs/default.yaml --mode train

# Evaluate a trained model
python main.py --config configs/default.yaml --mode eval --model_path checkpoints/your_model_name/final

# Start the inference server
python main.py --config configs/default.yaml --mode serve --model_path checkpoints/your_model_name/final

# Launch the demo UI (requires the inference server to be running)
python main.py --config configs/default.yaml --mode demo
```

## Detailed Setup

### Data Pipeline

The data pipeline consists of four stages:

1. **Collection**: Download datasets for math and coding problems
2. **Generation**: Generate solutions using a base model
3. **Filtering**: Filter solutions by quality (correctness)
4. **Formatting**: Format data for training

Run the data pipeline only:

```bash
python main.py --config configs/default.yaml --mode data
```

Or run each step individually:

```bash
# Collection
python data/collectors/dataset_collector.py

# Generation
python data/generators/solution_generator.py

# Filtering
python data/filters/quality_filter.py

# Formatting
python data/preprocessors/data_formatter.py
```

### Training

Training a model with default settings:

```bash
python main.py --config configs/default.yaml --mode train
```

Customize training by editing `configs/default.yaml` or creating a new config file. Key parameters to consider:

- **Model size**: Adjust based on your hardware
- **LoRA parameters**: Essential for efficient fine-tuning
- **Batch size and gradient accumulation**: Tune based on available VRAM
- **Learning rate and schedule**: Critical for good convergence

For advanced training with DeepSpeed:

1. Create a DeepSpeed config file in `configs/deepspeed.json`
2. Update your YAML config to point to this file
3. Run training with DeepSpeed:

```bash
python main.py --config configs/deepspeed_config.yaml --mode train
```

### Evaluation

Evaluate a trained model on benchmarks:

```bash
python main.py --config configs/default.yaml --mode eval --model_path checkpoints/your_model_name/final
```

The evaluation results will be saved to `evaluation/results/`.

To add custom benchmarks:
1. Create a JSON file in `evaluation/benchmarks/` with your problems
2. Update the config file to include your benchmark
3. Run evaluation

### Inference

Start the inference server to serve your model via a REST API:

```bash
python main.py --config configs/default.yaml --mode serve --model_path checkpoints/your_model_name/final
```

This will start a FastAPI server on `http://localhost:8000` by default.

API endpoints:
- `GET /` - Basic info
- `GET /health` - Server health check
- `GET /model/info` - Model information
- `POST /generate` - Generate text from a prompt

Example request:

```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Solve the equation: 2x + 5 = 15", "max_new_tokens": 512, "temperature": 0.7}'
```

### Demo UI

Launch the Streamlit demo UI:

```bash
python main.py --config configs/default.yaml --mode demo
```

This will start a Streamlit app on `http://localhost:8501` by default. Make sure the inference server is running before starting the demo.

## Cloud Setup

### Setting up on AWS

1. **Launch an EC2 instance**:
   - Instance type: g4dn.xlarge (minimal) or g5.xlarge (recommended)
   - AMI: Deep Learning AMI (DLAMI) with PyTorch
   
2. **SSH into instance and clone repository**:
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   git clone https://github.com/yourusername/SimpleFoundation.git
   cd SimpleFoundation
   ```
   
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
4. **Run with screen or tmux for persistence**:
   ```bash
   # Using screen
   screen -S training
   python main.py --config configs/default.yaml --mode full
   # Press Ctrl+A, D to detach
   
   # To reattach
   screen -r training
   ```

### Setting up on Google Cloud

1. **Create a Vertex AI notebook instance** with T4 or A100 GPU
2. **Clone repository in the notebook instance**
3. **Run the pipeline**

### Setting up on Lambda Labs/Vast.ai/RunPod

These services offer more cost-effective GPU rentals:

1. **Rent a GPU with at least 16GB VRAM**
2. **Use the provided SSH access to connect**
3. **Follow standard installation steps**

## Troubleshooting

### Common Issues

#### Out of Memory Errors
- Reduce batch size
- Enable gradient checkpointing
- Use 4-bit quantization
- Use LoRA with smaller rank values

#### Slow Training
- Check GPU utilization
- Increase batch size (if memory allows)
- Use flash-attention if available
- Enable mixed precision training

#### Model Doesn't Learn
- Check learning rate (try smaller values)
- Ensure data quality and formatting
- Increase number of epochs
- Monitor training loss to identify issues

#### Installation Errors
- Check CUDA and PyTorch compatibility
- Try using a container with pre-installed dependencies
- Check for conflicting packages

### Getting Help

If you encounter issues not covered in this guide:
1. Check the project issues on GitHub
2. Review the documentation of underlying libraries (Transformers, PEFT)
3. Open a new issue with detailed information about your problem
