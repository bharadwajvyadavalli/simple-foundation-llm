# Training Documentation

## Overview

The training module in SimpleFoundation provides a framework for efficiently training language models with reasoning capabilities. It's designed to make the training process accessible on consumer-grade hardware through various optimization techniques.

## Key Components

### Model Trainer

```
training/trainers/model_trainer.py
```

The `ModelTrainer` class is the central component that handles:

- Model and tokenizer initialization
- Training loop management
- Optimization techniques
- Checkpointing
- Evaluation during training

### Training Configuration

```
training/configs/default.json
```

A comprehensive configuration system allows you to customize all aspects of training without modifying code:

```json
{
  "model_name_or_path": "mistralai/Mistral-7B-v0.1",
  "use_lora": true,
  "load_in_4bit": true,
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "output_dir": "checkpoints/mistral-7b-reasoning",
  "max_seq_length": 2048,
  "train_batch_size": 1,
  "gradient_accumulation_steps": 4,
  "num_train_epochs": 3,
  "learning_rate": 2e-5,
  "weight_decay": 0.01,
  "warmup_ratio": 0.03,
  "fp16": true,
  "gradient_checkpointing": true
}
```

### Training Dataset

```
training/trainers/model_trainer.py (ReasoningDataset class)
```

The `ReasoningDataset` class handles:
- Loading and processing formatted training examples
- Tokenization and input preparation
- Efficient batching and sequence handling

## Training Process

### 1. Initialization

```python
# Initialize with configuration
from training.trainers.model_trainer import ModelTrainer, TrainingConfig

# Load configuration
config = TrainingConfig(
    model_name_or_path="mistralai/Mistral-7B-v0.1",
    use_lora=True,
    load_in_4bit=True,
    num_train_epochs=3
)

# Initialize trainer
trainer = ModelTrainer(config)
```

### 2. Model Loading

The trainer handles:
- Loading base models from Hugging Face
- Applying quantization (4-bit or 8-bit)
- Setting up LoRA adapter configurations
- Configuring tokenizer

### 3. Training

```python
# Train the model
final_checkpoint = trainer.train(
    train_file="data/formatted/combined_formatted.json",
    eval_file="data/formatted/validation.json"  # Optional
)
```

The training loop incorporates:
- Gradient accumulation for larger effective batch sizes
- Mixed precision training (fp16 or bf16)
- Learning rate scheduling
- Evaluation on a validation set (if provided)
- Checkpointing based on evaluation metrics
- Early stopping to prevent overfitting

### 4. Saving

The trainer saves:
- Model weights (full model or LoRA adapter)
- Tokenizer configuration
- Training configuration
- Generation parameters

## Optimization Techniques

### 1. Parameter-Efficient Fine-Tuning (PEFT)

The primary technique used is Low-Rank Adaptation (LoRA), which:
- Keeps most of the model frozen
- Adds small trainable matrices to specific layers
- Reduces memory usage by 85-95%
- Enables training on consumer GPUs

```python
# LoRA configuration
lora_config = LoraConfig(
    r=16,                      # Rank of update matrices
    lora_alpha=32,             # Scaling factor
    lora_dropout=0.05,         # Dropout probability
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Which modules to apply LoRA to
    bias="none",               # Whether to train bias parameters
    task_type="CAUSAL_LM"      # Task type
)
```

### 2. Quantization

Quantization reduces the precision of model weights:
- 4-bit quantization (recommended for most use cases)
- 8-bit quantization (alternative with slightly better quality)
- Enables training of models up to 3x larger on the same hardware

### 3. Gradient Checkpointing

This technique trades computation for memory:
- Discards activations during forward pass
- Recomputes them during backward pass
- Reduces memory usage by 30-50%
- Slightly slower (10-20%) but enables larger models

### 4. Flash Attention

An efficient attention implementation that:
- Reduces memory usage during attention computation
- Improves training speed
- Works with most modern GPUs

### 5. DeepSpeed Integration

For multi-GPU training:
- ZeRO optimizer stages for distributed training
- Offloading to CPU and NVMe when needed
- Parallel processing across multiple GPUs

## Hardware Requirements & Performance

### Minimal Setup (7B Parameter Models with LoRA)

- **GPU**: 1x RTX 3090 (24GB VRAM)
- **Performance**: ~500 tokens/sec
- **Training Time**: ~6-12 hours for 3 epochs on a dataset of 5,000 examples
- **Memory Usage**: ~22GB VRAM

### Recommended Setup (13B-70B Parameter Models with LoRA)

- **GPU**: 4x A100 (40GB or 80GB)
- **Performance**: ~2,000-5,000 tokens/sec
- **Training Time**: ~4-8 hours for 3 epochs on a dataset of 15,000 examples
- **Memory Usage**: ~35GB VRAM per GPU

## Example Training Scenarios

### Scenario 1: Basic Training (Single GPU)

```bash
# Train on a small dataset with minimal hardware
python -m training.trainers.model_trainer \
    --model_name_or_path "mistralai/Mistral-7B-v0.1" \
    --train_file "data/formatted/combined_formatted.json" \
    --output_dir "checkpoints/mistral-7b-reasoning" \
    --use_lora True \
    --load_in_4bit True \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4
```

### Scenario 2: Distributed Training (Multiple GPUs)

```bash
# Train on multiple GPUs with DeepSpeed
deepspeed training/trainers/model_trainer.py \
    --model_name_or_path "meta-llama/Llama-2-13b-hf" \
    --train_file "data/formatted/combined_formatted.json" \
    --output_dir "checkpoints/llama-13b-reasoning" \
    --use_lora True \
    --load_in_4bit True \
    --num_train_epochs 3 \
    --deepspeed_config "training/configs/deepspeed_zero3.json"
```

## Tracking & Visualization

SimpleFoundation integrates with Weights & Biases (wandb) for experiment tracking:

```python
# Enable wandb tracking
config = TrainingConfig(
    # ...other parameters...
    use_wandb=True,
    wandb_project="SimpleFoundation",
    wandb_name="mistral-7b-v0.1-reasoning"
)
```

This provides:
- Loss curves and learning rate tracking
- GPU utilization monitoring
- Model output samples
- Hyperparameter comparison across runs

## Advanced Configuration

### Learning Rate Scheduling

```json
{
  "learning_rate": 2e-5,
  "warmup_ratio": 0.03,
  "lr_scheduler_type": "cosine"
}
```

Available schedulers:
- `linear`: Linear decay from peak to zero
- `cosine`: Cosine decay (recommended)
- `constant`: No decay, fixed learning rate
- `constant_with_warmup`: Fixed after warmup
- `polynomial`: Polynomial decay

### Precision Control

```json
{
  "fp16": true,
  "bf16": false
}
```

- `fp16`: 16-bit floating point (works on most GPUs)
- `bf16`: Brain floating point (only on newer NVIDIA GPUs and AMD MI100+)

### Early Stopping

```json
{
  "early_stopping_patience": 5,
  "early_stopping_threshold": 0.01
}
```

Stops training when validation metrics stop improving.

## Best Practices

1. **Start Small**: Begin with a smaller model (7B) and smaller dataset to validate your pipeline
2. **Batch Size**: Use gradient accumulation instead of larger batch sizes for better memory efficiency
3. **Learning Rate**: Start with 2e-5 for smaller models, 1e-5 for larger ones
4. **Monitoring**: Always track loss curves to detect overfitting or training instability
5. **Save Frequent Checkpoints**: Save checkpoints every 500-1000 steps
6. **Validation**: Use a small validation set to monitor performance
7. **Model Selection**: More recent models (Mistral, Llama 2, Qwen) tend to perform better than older ones

## Troubleshooting

### Out of Memory Errors

1. Reduce batch size or model size
2. Enable gradient checkpointing
3. Use 4-bit quantization
4. Reduce sequence length
5. Use DeepSpeed ZeRO-3 with offloading

### Slow Training

1. Check if GPU utilization is high (should be 80%+)
2. Disable gradient checkpointing if not needed
3. Use flash attention if available
4. Ensure training data is properly pre-processed

### Poor Performance

1. Verify data quality and formatting
2. Check learning rate (too high or too low)
3. Train for more epochs
4. Use a better base model
5. Increase LoRA rank (r) for more capacity

## Model Merging (Advanced)

For production deployment, you can merge LoRA weights back into the base model:

```python
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# Load trained LoRA adapter
adapter_path = "checkpoints/mistral-7b-reasoning/final"
model = PeftModel.from_pretrained(base_model, adapter_path)

# Merge weights
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("final_merged_model")
```

This creates a standalone model that doesn't require LoRA during inference.
