#!/usr/bin/env python3
"""
SimpleFoundation: Main script for end-to-end pipeline

This script provides a complete pipeline for data collection, generation,
training, evaluation, and inference of foundational models.
"""

import os
import sys
import argparse
import logging
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_directories(config: Dict) -> None:
    """
    Create necessary directories based on configuration.
    
    Args:
        config: Configuration dictionary
    """
    # Data directories
    os.makedirs(config["data"]["raw_dir"], exist_ok=True)
    os.makedirs(config["data"]["generated_dir"], exist_ok=True)
    os.makedirs(config["data"]["filtered_dir"], exist_ok=True)
    os.makedirs(config["data"]["formatted_dir"], exist_ok=True)
    
    # Training directories
    os.makedirs(config["training"]["output_dir"], exist_ok=True)
    
    # Evaluation directories
    os.makedirs(config["evaluation"]["results_dir"], exist_ok=True)
    
    # Benchmark directories
    os.makedirs("evaluation/benchmarks", exist_ok=True)
    
    logger.info("Created all necessary directories")


def run_data_pipeline(config: Dict) -> Dict:
    """
    Run the data pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with data pipeline results
    """
    logger.info("Starting data pipeline")
    
    # Import data pipeline modules
    sys.path.append(".")
    from data.collectors.dataset_collector import download_all_datasets
    from data.generators.solution_generator import generate_all_solutions
    from data.filters.quality_filter import filter_all_solutions
    from data.preprocessors.data_formatter import format_all_data
    
    # 1. Collect datasets
    logger.info("Step 1: Collecting datasets")
    collection_result = download_all_datasets(
        output_dir=config["data"]["raw_dir"]
    )
    
    # 2. Generate solutions
    logger.info("Step 2: Generating solutions")
    generation_result = generate_all_solutions(
        math_problems_file=os.path.join(config["data"]["raw_dir"], "math/math_processed.json"),
        coding_problems_file=os.path.join(config["data"]["raw_dir"], "coding/apps_processed.json"),
        model_name=config["data"]["generate"]["model_name"],
        output_dir=config["data"]["generated_dir"]
    )
    
    # 3. Filter solutions
    logger.info("Step 3: Filtering solutions")
    filter_result = filter_all_solutions(
        math_solutions_file=generation_result["math"]["file"],
        coding_solutions_file=generation_result["coding"]["file"],
        output_dir=config["data"]["filtered_dir"]
    )
    
    # 4. Format data
    logger.info("Step 4: Formatting data")
    format_result = format_all_data(
        math_filtered_file=filter_result["math"]["file"],
        coding_filtered_file=filter_result["coding"]["file"],
        output_dir=config["data"]["formatted_dir"]
    )
    
    logger.info("Data pipeline completed successfully")
    
    return {
        "collection": collection_result,
        "generation": generation_result,
        "filter": filter_result,
        "format": format_result
    }


def run_training(config: Dict) -> str:
    """
    Run model training.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Path to trained model
    """
    logger.info("Starting model training")
    
    # Import training module
    from training.trainers.model_trainer import TrainingConfig, train_model
    
    # Convert config dict to TrainingConfig
    training_config = TrainingConfig(
        model_name_or_path=config["training"]["model_name"],
        tokenizer_name_or_path=config["training"]["tokenizer_name"],
        use_lora=config["training"]["use_lora"],
        load_in_8bit=config["training"]["load_in_8bit"],
        load_in_4bit=config["training"]["load_in_4bit"],
        lora_r=config["training"]["lora"]["r"],
        lora_alpha=config["training"]["lora"]["alpha"],
        lora_dropout=config["training"]["lora"]["dropout"],
        lora_target_modules=config["training"]["lora"]["target_modules"],
        output_dir=config["training"]["output_dir"],
        max_seq_length=config["training"]["max_seq_length"],
        train_batch_size=config["training"]["train_batch_size"],
        eval_batch_size=config["training"]["eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        num_train_epochs=config["training"]["num_train_epochs"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        warmup_ratio=config["training"]["warmup_ratio"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        fp16=config["training"]["fp16"],
        bf16=config["training"]["bf16"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        deepspeed_config=config["training"]["deepspeed_config"],
        log_every_n_steps=config["training"]["log_every_n_steps"],
        eval_every_n_steps=config["training"]["eval_every_n_steps"],
        save_every_n_steps=config["training"]["save_every_n_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        use_wandb=config["training"]["use_wandb"],
        wandb_project=config["training"]["wandb_project"],
        wandb_name=config["training"]["wandb_name"],
        early_stopping_patience=config["training"]["early_stopping"]["patience"],
        early_stopping_threshold=config["training"]["early_stopping"]["threshold"],
        seed=config["training"]["seed"]
    )
    
    # Start training
    train_file = os.path.join(config["data"]["formatted_dir"], "combined_formatted.json")
    
    model_path = train_model(
        config=training_config,
        train_file=train_file
    )
    
    logger.info(f"Training completed successfully. Model saved to {model_path}")
    
    return model_path


def run_evaluation(config: Dict, model_path: str) -> Dict:
    """
    Run model evaluation.
    
    Args:
        config: Configuration dictionary
        model_path: Path to trained model
        
    Returns:
        Evaluation results
    """
    logger.info("Starting model evaluation")
    
    # Import evaluation module
    from evaluation.benchmarks.benchmark_evaluator import run_evaluation
    
    # Run evaluation
    results = run_evaluation(
        model_path=model_path,
        math_benchmarks=config["evaluation"]["benchmarks"]["math"],
        coding_benchmarks=config["evaluation"]["benchmarks"]["coding"],
        output_dir=config["evaluation"]["results_dir"]
    )
    
    logger.info("Evaluation completed successfully")
    
    return results


def run_inference_server(config: Dict, model_path: str) -> None:
    """
    Run inference server.
    
    Args:
        config: Configuration dictionary
        model_path: Path to trained model
    """
    logger.info("Starting inference server")
    
    # Import inference server module
    from inference.server.model_server import start_server
    
    # Start server
    start_server(
        model_path=model_path,
        host=config["inference"]["host"],
        port=config["inference"]["port"],
        load_in_8bit=config["inference"]["load_in_8bit"],
        load_in_4bit=config["inference"]["load_in_4bit"],
        use_flash_attention=config["inference"]["use_flash_attention"],
        max_concurrent_requests=config["inference"]["max_concurrent_requests"]
    )


def run_demo(config: Dict) -> None:
    """
    Run demo UI.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("Starting demo UI")
    
    # Import demo UI module
    from inference.demo.app import run_demo
    
    # Start demo
    run_demo(
        api_url=config["demo"]["api_url"]
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="SimpleFoundation: End-to-end foundational model pipeline")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "data", "train", "eval", "serve", "demo"],
        default="full",
        help="Pipeline mode"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to model for evaluation/inference (required for eval/serve/demo modes)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup directories
    setup_directories(config)
    
    # Determine pipeline flow based on mode
    if args.mode == "full" or args.mode == "data":
        data_results = run_data_pipeline(config)
        
        # Save data pipeline results
        with open("data_pipeline_results.json", 'w') as f:
            json.dump(data_results, f, indent=2)
    
    model_path = args.model_path
    
    if args.mode == "full" or args.mode == "train":
        model_path = run_training(config)
    
    if model_path is None:
        logger.error("Model path is required for evaluation/inference but was not provided")
        sys.exit(1)
    
    if args.mode == "full" or args.mode == "eval":
        eval_results = run_evaluation(config, model_path)
        
        # Save evaluation results
        with open("evaluation_results.json", 'w') as f:
            json.dump(eval_results, f, indent=2)
    
    if args.mode == "full" or args.mode == "serve":
        run_inference_server(config, model_path)
    
    if args.mode == "demo":
        run_demo(config)


if __name__ == "__main__":
    main()
