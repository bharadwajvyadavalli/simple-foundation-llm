"""
Model Trainer Module for SimpleFoundation

This module handles the training of foundational models using formatted data.
It supports various optimization techniques for efficient training on limited hardware.
"""

import os
import json
import time
import logging
import torch
import random
import numpy as np
import wandb
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Model settings
    model_name_or_path: str = "mistralai/Mistral-7B-v0.1"
    tokenizer_name_or_path: str = None  # Defaults to model_name_or_path if None
    use_lora: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = True
    
    # LoRA settings (if use_lora=True)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    # Training settings
    output_dir: str = "checkpoints"
    max_seq_length: int = 2048
    train_batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # Optimization settings
    fp16: bool = True
    bf16: bool = False  # Only use on GPUs that support bfloat16
    gradient_checkpointing: bool = True
    deepspeed_config: Optional[str] = None
    
    # Logging/saving settings
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 200
    save_every_n_steps: int = 500
    save_total_limit: int = 3
    use_wandb: bool = False
    wandb_project: str = "SimpleFoundation"
    wandb_name: Optional[str] = None
    seed: int = 42
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.01
    
    def __post_init__(self):
        """Set default values for tokenizer if not provided."""
        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path
        
        # Create a unique run name if using wandb
        if self.use_wandb and self.wandb_name is None:
            model_short_name = self.model_name_or_path.split("/")[-1]
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.wandb_name = f"{model_short_name}-{timestamp}"


class ReasoningDataset(Dataset):
    """Dataset for training reasoning models."""
    
    def __init__(
        self,
        data_file: str,
        tokenizer,
        max_seq_length: int = 2048,
        prompt_key: str = "prompt",
        response_key: str = "response"
    ):
        """
        Initialize the dataset.
        
        Args:
            data_file: Path to JSON file with formatted examples
            tokenizer: Tokenizer for encoding examples
            max_seq_length: Maximum sequence length for tokenization
            prompt_key: Key in JSON for prompt text
            response_key: Key in JSON for response text
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.prompt_key = prompt_key
        self.response_key = response_key
        
        # Load and process data
        with open(data_file, 'r') as f:
            self.examples = json.load(f)
        
        logger.info(f"Loaded {len(self.examples)} examples from {data_file}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Get a training example.
        
        Args:
            idx: Index of the example
            
        Returns:
            Tokenized example
        """
        example = self.examples[idx]
        prompt = example[self.prompt_key]
        response = example[self.response_key]
        
        # Format as prompt + response
        prompt_token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_token_ids = self.tokenizer.encode(response, add_special_tokens=False)
        
        # Ensure we don't exceed max sequence length
        max_response_length = self.max_seq_length - len(prompt_token_ids) - 2  # For BOS and EOS tokens
        if max_response_length < 0:
            logger.warning(f"Prompt is too long ({len(prompt_token_ids)} tokens), truncating prompt")
            prompt_token_ids = prompt_token_ids[:self.max_seq_length - 100]  # Leave room for at least some response
            max_response_length = self.max_seq_length - len(prompt_token_ids) - 2
        
        response_token_ids = response_token_ids[:max_response_length]
        
        # Create input_ids with BOS at the beginning
        input_ids = [self.tokenizer.bos_token_id] + prompt_token_ids + response_token_ids + [self.tokenizer.eos_token_id]
        
        # Create labels: -100 for prompt (ignored in loss), actual ids for response
        labels = [-100] * (len(prompt_token_ids) + 1) + response_token_ids + [self.tokenizer.eos_token_id]
        
        # Ensure equal length
        assert len(input_ids) == len(labels), f"Length mismatch: {len(input_ids)} vs {len(labels)}"
        
        # Convert to tensor
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids)
        }


class MetricsCallback(TrainerCallback):
    """Callback for logging detailed metrics during training."""
    
    def __init__(self, eval_dataset, tokenizer, log_every_n_steps=10):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.log_every_n_steps = log_every_n_steps
    
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Log metrics at the end of each step."""
        if state.global_step % self.log_every_n_steps == 0:
            logs = {}
            
            # Extract training loss
            if state.log_history:
                for entry in reversed(state.log_history):
                    if "loss" in entry:
                        logs["train/loss"] = entry["loss"]
                        break
            
            # Log learning rate
            if kwargs.get("optimizer") is not None:
                for param_group in kwargs["optimizer"].param_groups:
                    logs["train/learning_rate"] = param_group["lr"]
                    break
            
            # Log to wandb if available
            if wandb.run is not None:
                wandb.log(logs, step=state.global_step)
        
        return control


class ModelTrainer:
    """Trainer for foundational models."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the model trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        
        # Set seed for reproducibility
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize model and tokenizer
        self._init_model_and_tokenizer()
        
        # Set up wandb if enabled
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_name,
                config=vars(config)
            )
    
    def _init_model_and_tokenizer(self):
        """Initialize model and tokenizer based on config."""
        logger.info(f"Loading tokenizer: {self.config.tokenizer_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name_or_path,
            use_fast=True,
            padding_side="right"
        )
        
        # Ensure the tokenizer has padding token
        if self.tokenizer.pad_token is None:
            # For autoregressive models like GPT, use EOS as padding token
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Loading model: {self.config.model_name_or_path}")
        
        # Quantization settings
        quantization_config = None
        if self.config.load_in_4bit:
            quantization_config = {"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16}
        elif self.config.load_in_8bit:
            quantization_config = {"load_in_8bit": True}
        
        # Load model with quantization if enabled
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16 if self.config.fp16 else (torch.bfloat16 if self.config.bf16 else torch.float32),
            quantization_config=quantization_config
        )
        
        # Apply LoRA if enabled
        if self.config.use_lora:
            logger.info("Applying LoRA to model")
            
            # Prepare model for k-bit training if using quantization
            if self.config.load_in_4bit or self.config.load_in_8bit:
                self.model = prepare_model_for_kbit_training(self.model)
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Apply LoRA to model
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # Enable gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
    
    def train(self, train_file: str, eval_file: Optional[str] = None, resume_from_checkpoint: Optional[str] = None):
        """
        Train the model on formatted data.
        
        Args:
            train_file: Path to training data file
            eval_file: Path to evaluation data file (optional)
            resume_from_checkpoint: Path to checkpoint to resume from (optional)
            
        Returns:
            Path to the best checkpoint
        """
        logger.info(f"Setting up training with data from {train_file}")
        
        # Create datasets
        train_dataset = ReasoningDataset(
            train_file,
            self.tokenizer,
            max_seq_length=self.config.max_seq_length
        )
        
        eval_dataset = None
        if eval_file:
            logger.info(f"Using evaluation data from {eval_file}")
            eval_dataset = ReasoningDataset(
                eval_file,
                self.tokenizer,
                max_seq_length=self.config.max_seq_length
            )
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            report_to="wandb" if self.config.use_wandb else "none",
            logging_steps=self.config.log_every_n_steps,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=self.config.eval_every_n_steps if eval_dataset else None,
            save_strategy="steps",
            save_steps=self.config.save_every_n_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True if eval_dataset else False,
            deepspeed=self.config.deepspeed_config,
            seed=self.config.seed,
            optim="adamw_torch",
            ddp_find_unused_parameters=False,
            disable_tqdm=False
        )
        
        # Set up callbacks
        callbacks = []
        
        # Add early stopping if using evaluation
        if eval_dataset:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold
                )
            )
        
        # Add metrics callback
        if eval_dataset:
            callbacks.append(
                MetricsCallback(
                    eval_dataset,
                    self.tokenizer,
                    log_every_n_steps=self.config.log_every_n_steps
                )
            )
        
        # Set up trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        # Train the model
        logger.info("Starting training")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save the final model
        final_checkpoint_dir = os.path.join(self.config.output_dir, "final")
        logger.info(f"Saving final model to {final_checkpoint_dir}")
        
        # For LoRA, save adapter only
        if self.config.use_lora:
            self.model.save_pretrained(final_checkpoint_dir)
        else:
            # Save full model if not using LoRA
            trainer.save_model(final_checkpoint_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(final_checkpoint_dir)
        
        # Save config
        with open(os.path.join(final_checkpoint_dir, "training_config.json"), 'w') as f:
            json.dump(vars(self.config), f, indent=2)
        
        return final_checkpoint_dir


def train_model(
    config: TrainingConfig,
    train_file: str = "data/formatted/combined_formatted.json",
    eval_file: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None
):
    """
    Train a model with the given configuration.
    
    Args:
        config: Training configuration
        train_file: Path to training data file
        eval_file: Path to evaluation data file (optional)
        resume_from_checkpoint: Path to checkpoint to resume from (optional)
        
    Returns:
        Path to the trained model
    """
    trainer = ModelTrainer(config)
    return trainer.train(train_file, eval_file, resume_from_checkpoint)


if __name__ == "__main__":
    # Example usage
    config = TrainingConfig(
        model_name_or_path="mistralai/Mistral-7B-v0.1",
        use_lora=True,
        load_in_4bit=True,
        num_train_epochs=3,
        learning_rate=2e-5,
        output_dir="checkpoints/mistral-7b-reasoning"
    )
    
    checkpoint_dir = train_model(
        config,
        train_file="data/formatted/combined_formatted.json"
    )
    
    print(f"Model trained and saved to {checkpoint_dir}")
