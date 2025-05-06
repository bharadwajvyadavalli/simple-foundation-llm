"""
Benchmark Evaluator Module for SimpleFoundation

This module evaluates trained models on various benchmarks
for math and coding reasoning tasks.
"""

import os
import re
import json
import time
import logging
import torch
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)
from peft import PeftModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """Base class for benchmark evaluation."""
    
    def __init__(
        self,
        model_path: str,
        output_dir: str = "evaluation/results",
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        use_flash_attention: bool = True
    ):
        """
        Initialize the benchmark evaluator.
        
        Args:
            model_path: Path to the model checkpoint
            output_dir: Directory to save evaluation results
            device: Device to run evaluation on (default: auto-detect)
            load_in_8bit: Whether to load model in 8-bit quantization
            load_in_4bit: Whether to load model in 4-bit quantization
            use_flash_attention: Whether to use flash attention for faster inference
        """
        self.model_path = model_path
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.use_flash_attention = use_flash_attention
        
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self._load_model_and_tokenizer()
    
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=True
        )
        
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Determine if this is a LoRA model
        is_lora = os.path.exists(os.path.join(self.model_path, "adapter_config.json"))
        
        # Quantization settings
        if self.load_in_4bit:
            logger.info("Loading model in 4-bit precision")
            quantization_config = {"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16}
        elif self.load_in_8bit:
            logger.info("Loading model in 8-bit precision")
            quantization_config = {"load_in_8bit": True}
        else:
            quantization_config = None
        
        # Flash attention settings
        attn_config = {"use_flash_attention_2": self.use_flash_attention} if self.use_flash_attention else {}
        
        # If it's a LoRA model, we need to load the base model first
        if is_lora:
            # Try to find the base model path in the config
            try:
                with open(os.path.join(self.model_path, "adapter_config.json"), 'r') as f:
                    adapter_config = json.load(f)
                base_model_path = adapter_config.get("base_model_name_or_path")
            except:
                # If we can't find it, look for a training config file
                try:
                    with open(os.path.join(self.model_path, "training_config.json"), 'r') as f:
                        training_config = json.load(f)
                    base_model_path = training_config.get("model_name_or_path")
                except:
                    raise ValueError("Could not determine base model path for LoRA model")
            
            logger.info(f"Loading base model {base_model_path} for LoRA")
            
            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                attn_implementation="flash_attention_2" if self.use_flash_attention else "eager"
            )
            
            # Load LoRA adapter
            logger.info(f"Loading LoRA adapter from {self.model_path}")
            self.model = PeftModel.from_pretrained(self.model, self.model_path)
        else:
            # Load full model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                attn_implementation="flash_attention_2" if self.use_flash_attention else "eager"
            )
        
        # Set up generation config
        self.generation_config = GenerationConfig.from_pretrained(
            self.model_path,
            # Use conservative defaults if config not found
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            max_new_tokens=1024
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Merge with default generation config
        generation_kwargs = self.generation_config.to_dict()
        generation_kwargs.update(kwargs)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **generation_kwargs
            )
        
        # Skip prompt tokens in the output
        response_ids = output_ids[0][inputs.input_ids.shape[1]:]
        
        # Decode response
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        return response
    
    def evaluate_benchmark(self, benchmark_file: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the model on a benchmark.
        
        Args:
            benchmark_file: Path to benchmark file
            output_file: Path to save results
            
        Returns:
            Evaluation results
        """
        raise NotImplementedError("Subclasses must implement evaluate_benchmark")
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """
        Save evaluation results.
        
        Args:
            results: Evaluation results
            output_file: Path to save results
        """
        # Save results as JSON
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved evaluation results to {output_file}")
        
        # Generate a report if there are aggregate metrics
        if "metrics" in results:
            self._generate_report(results, output_file)
    
    def _generate_report(self, results: Dict[str, Any], output_file: str):
        """
        Generate a report with visualizations.
        
        Args:
            results: Evaluation results
            output_file: Path to save results
        """
        # Create a plot for metrics
        metrics = results["metrics"]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot metrics as a bar chart
        values = list(metrics.values())
        labels = [k.replace("_", " ").title() for k in metrics.keys()]
        
        ax.bar(labels, values)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Score")
        ax.set_title(f"Evaluation Results: {results.get('benchmark_name', 'Benchmark')}")
        
        # Add values on top of bars
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_file.replace(".json", ".png")
        plt.savefig(plot_file)
        logger.info(f"Saved results plot to {plot_file}")


class MathBenchmarkEvaluator(BenchmarkEvaluator):
    """Evaluator for math reasoning benchmarks."""
    
    def __init__(
        self,
        model_path: str,
        output_dir: str = "evaluation/results/math",
        **kwargs
    ):
        """Initialize with math-specific output directory."""
        super().__init__(model_path, output_dir, **kwargs)
        
        # Template for formatting math problems
        self.prompt_template = """
You are a helpful AI math assistant. You excel at solving math problems by breaking them down into clear, logical steps.

Problem:
{problem}

Solve this problem step by step, showing your reasoning clearly.
"""
    
    def extract_answer(self, response: str) -> Optional[str]:
        """
        Extract the final answer from a response.
        
        Args:
            response: Model's response
            
        Returns:
            Extracted answer or None if not found
        """
        # Look for patterns like "Final answer: X" or "Therefore, the answer is X"
        answer_patterns = [
            r"(?:therefore|thus|so),?\s+(?:the)?\s*answer(?:\s+is)?:?\s*([^\n.,]+)",
            r"(?:final|the)\s+answer(?:\s+is)?:\s*([^\n.,]+)",
            r"(?:in\s+conclusion),?\s+(?:the)?\s*(?:answer|value|result)(?:\s+is)?:?\s*([^\n.,]+)",
            r"(?:the\s+)?(?:answer|value|result)(?:\s+is)?:?\s*([^\n.,]+)"
        ]
        
        # Try each pattern
        for pattern in answer_patterns:
            matches = re.search(pattern, response, re.IGNORECASE)
            if matches:
                answer = matches.group(1).strip()
                return answer
        
        # Last attempt: try to extract the last number in the response
        numbers = re.findall(r'\b\d+(?:/\d+)?\b', response)
        if numbers:
            return numbers[-1]
        
        return None
    
    def is_correct(self, generated_answer: Optional[str], correct_answer: str) -> bool:
        """
        Check if the generated answer is correct.
        
        Args:
            generated_answer: Extracted answer
            correct_answer: Expected correct answer
            
        Returns:
            True if correct, False otherwise
        """
        if generated_answer is None:
            return False
        
        # Normalize answers for comparison
        generated_norm = generated_answer.strip().lower().replace(" ", "")
        correct_norm = correct_answer.strip().lower().replace(" ", "")
        
        # Direct match
        if generated_norm == correct_norm:
            return True
        
        # Try matching numeric values
        try:
            # Handle fractions
            if "/" in generated_norm:
                num, denom = map(int, generated_norm.split("/"))
                generated_value = num / denom
            else:
                generated_value = float(generated_norm)
            
            if "/" in correct_norm:
                num, denom = map(int, correct_norm.split("/"))
                correct_value = num / denom
            else:
                correct_value = float(correct_norm)
            
            # Compare with a small tolerance
            return abs(generated_value - correct_value) < 1e-6
        except (ValueError, ZeroDivisionError):
            # If conversion fails, fall back to string comparison
            return False
    
    def evaluate_benchmark(self, benchmark_file: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the model on a math benchmark.
        
        Args:
            benchmark_file: Path to benchmark file
            output_file: Path to save results
            
        Returns:
            Evaluation results
        """
        # Load benchmark problems
        with open(benchmark_file, 'r') as f:
            problems = json.load(f)
        
        logger.info(f"Evaluating on {len(problems)} math problems from {benchmark_file}")
        
        # Extract benchmark name from filename
        benchmark_name = os.path.basename(benchmark_file).split(".")[0]
        
        results = {
            "benchmark_name": benchmark_name,
            "model_path": self.model_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "problems": []
        }
        
        correct_count = 0
        
        # Evaluate each problem
        for i, problem in enumerate(tqdm(problems, desc=f"Evaluating {benchmark_name}")):
            problem_id = problem.get("id", f"problem_{i}")
            problem_text = problem["problem"]
            expected_answer = problem["answer"]
            
            # Format prompt
            prompt = self.prompt_template.format(problem=problem_text)
            
            # Generate response
            try:
                response = self.generate(prompt, do_sample=False, temperature=0.0)
                
                # Extract answer
                extracted_answer = self.extract_answer(response)
                
                # Check correctness
                correct = self.is_correct(extracted_answer, expected_answer)
                
                if correct:
                    correct_count += 1
                
                # Store results
                results["problems"].append({
                    "id": problem_id,
                    "problem": problem_text,
                    "expected_answer": expected_answer,
                    "response": response,
                    "extracted_answer": extracted_answer,
                    "correct": correct
                })
                
            except Exception as e:
                logger.error(f"Error evaluating problem {problem_id}: {e}")
                
                # Store error result
                results["problems"].append({
                    "id": problem_id,
                    "problem": problem_text,
                    "expected_answer": expected_answer,
                    "error": str(e),
                    "correct": False
                })
        
        # Calculate accuracy
        accuracy = correct_count / len(problems) if problems else 0
        
        # Add metrics
        results["metrics"] = {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(problems)
        }
        
        # Save results
        if output_file is None:
            output_file = os.path.join(self.output_dir, f"{benchmark_name}_results.json")
        
        self.save_results(results, output_file)
        
        return results


class CodingBenchmarkEvaluator(BenchmarkEvaluator):
    """Evaluator for coding reasoning benchmarks."""
    
    def __init__(
        self,
        model_path: str,
        output_dir: str = "evaluation/results/coding",
        **kwargs
    ):
        """Initialize with coding-specific output directory."""
        super().__init__(model_path, output_dir, **kwargs)
        
        # Template for formatting coding problems
        self.prompt_template = """
You are an expert coding assistant. You excel at solving programming problems with clear step-by-step reasoning.

Problem:
{problem}

Solve this coding problem step by step in Python, explaining your thought process and then providing the final working code.
"""
    
    def extract_code(self, response: str) -> Optional[str]:
        """
        Extract Python code from a response.
        
        Args:
            response: Model's response
            
        Returns:
            Extracted code or None if not found
        """
        # Look for code blocks
        code_block_pattern = r"```(?:python)?(.*?)```"
        code_blocks = re.findall(code_block_pattern, response, re.DOTALL)
        
        if code_blocks:
            # Return the last code block (usually the final solution)
            return code_blocks[-1].strip()
        
        # If no code blocks, try to extract based on indentation and Python keywords
        lines = response.split("\n")
        code_lines = []
        in_code = False
        
        for line in lines:
            stripped = line.strip()
            
            # Check for Python keywords and patterns
            if (stripped.startswith("def ") or 
                stripped.startswith("class ") or 
                stripped.startswith("import ") or 
                stripped.startswith("from ") or
                "=" in stripped and not stripped.startswith("#")):
                in_code = True
            
            if in_code:
                code_lines.append(line)
        
        if code_lines:
            return "\n".join(code_lines)
        
        return None
    
    def test_code(self, code: str, test_cases: List[Dict]) -> Tuple[bool, List[bool]]:
        """
        Test extracted code against test cases.
        
        Args:
            code: Extracted code
            test_cases: List of test cases
            
        Returns:
            Tuple of (overall_correct, per_test_results)
        """
        # This is a simplified version that delegates to the same test runner
        # used in quality_filter.py
        from tempfile import TemporaryDirectory
        import subprocess
        
        if not code or not test_cases:
            return False, []
        
        # Create a temporary directory for testing
        with TemporaryDirectory() as temp_dir:
            # Write solution to a file
            solution_file = os.path.join(temp_dir, "solution.py")
            with open(solution_file, 'w') as f:
                f.write(code)
            
            # Create test runner
            test_runner = os.path.join(temp_dir, "test_runner.py")
            with open(test_runner, 'w') as f:
                f.write("""
import sys
import json
from solution import *

def run_test(test_input, expected_output):
    # This is a simplified test runner
    # In a real implementation, you would handle different function signatures,
    # multiple inputs, different output formats, etc.
    
    # For simplicity, we assume the solution defines a 'solution' function
    try:
        actual_output = solution(test_input)
        # Compare with expected output (basic string comparison)
        return str(actual_output).strip() == str(expected_output).strip()
    except Exception as e:
        print(f"Error running test: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    test_cases = json.loads(sys.argv[1])
    results = []
    
    for test_case in test_cases:
        test_input = test_case["input"]
        expected_output = test_case["expected_output"]
        result = run_test(test_input, expected_output)
        results.append(result)
    
    print(json.dumps(results))
""")
            
            # Run tests
            try:
                cmd = ["python", test_runner, json.dumps(test_cases)]
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=5  # Timeout to prevent infinite loops
                )
                
                if process.returncode == 0:
                    test_results = json.loads(process.stdout)
                    return all(test_results), test_results
                else:
                    logger.warning(f"Test runner failed: {process.stderr}")
                    return False, []
            
            except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
                logger.warning(f"Test execution error: {e}")
                return False, []
    
    def evaluate_benchmark(self, benchmark_file: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the model on a coding benchmark.
        
        Args:
            benchmark_file: Path to benchmark file
            output_file: Path to save results
            
        Returns:
            Evaluation results
        """
        # Load benchmark problems
        with open(benchmark_file, 'r') as f:
            problems = json.load(f)
        
        logger.info(f"Evaluating on {len(problems)} coding problems from {benchmark_file}")
        
        # Extract benchmark name from filename
        benchmark_name = os.path.basename(benchmark_file).split(".")[0]
        
        results = {
            "benchmark_name": benchmark_name,
            "model_path": self.model_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "problems": []
        }
        
        correct_count = 0
        
        # Evaluate each problem
        for i, problem in enumerate(tqdm(problems, desc=f"Evaluating {benchmark_name}")):
            problem_id = problem.get("id", f"problem_{i}")
            problem_text = problem["problem"]
            test_cases = problem.get("test_cases", [])
            
            # Format prompt
            prompt = self.prompt_template.format(problem=problem_text)
            
            # Generate response
            try:
                response = self.generate(prompt)
                
                # Extract code
                extracted_code = self.extract_code(response)
                
                # Test code
                correct, test_results = self.test_code(extracted_code, test_cases)
                
                if correct:
                    correct_count += 1
                
                # Store results
                results["problems"].append({
                    "id": problem_id,
                    "problem": problem_text,
                    "response": response,
                    "extracted_code": extracted_code,
                    "test_results": test_results,
                    "correct": correct
                })
                
            except Exception as e:
                logger.error(f"Error evaluating problem {problem_id}: {e}")
                
                # Store error result
                results["problems"].append({
                    "id": problem_id,
                    "problem": problem_text,
                    "error": str(e),
                    "correct": False
                })
        
        # Calculate accuracy
        accuracy = correct_count / len(problems) if problems else 0
        
        # Add metrics
        results["metrics"] = {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(problems)
        }
        
        # Save results
        if output_file is None:
            output_file = os.path.join(self.output_dir, f"{benchmark_name}_results.json")
        
        self.save_results(results, output_file)
        
        return results


def run_evaluation(
    model_path: str,
    math_benchmarks: List[str] = None,
    coding_benchmarks: List[str] = None,
    output_dir: str = "evaluation/results"
):
    """
    Run evaluation on multiple benchmarks.
    
    Args:
        model_path: Path to model checkpoint
        math_benchmarks: List of math benchmark files
        coding_benchmarks: List of coding benchmark files
        output_dir: Base directory to save results
        
    Returns:
        Dictionary with evaluation results
    """
    results = {
        "math": {},
        "coding": {}
    }
    
    # Run math benchmarks
    if math_benchmarks:
        math_evaluator = MathBenchmarkEvaluator(
            model_path=model_path,
            output_dir=os.path.join(output_dir, "math")
        )
        
        for benchmark_file in math_benchmarks:
            benchmark_name = os.path.basename(benchmark_file).split(".")[0]
            results["math"][benchmark_name] = math_evaluator.evaluate_benchmark(benchmark_file)
    
    # Run coding benchmarks
    if coding_benchmarks:
        coding_evaluator = CodingBenchmarkEvaluator(
            model_path=model_path,
            output_dir=os.path.join(output_dir, "coding")
        )
        
        for benchmark_file in coding_benchmarks:
            benchmark_name = os.path.basename(benchmark_file).split(".")[0]
            results["coding"][benchmark_name] = coding_evaluator.evaluate_benchmark(benchmark_file)
    
    # Generate overall report
    overall_results = {
        "model_path": model_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "math_benchmarks": [os.path.basename(f) for f in (math_benchmarks or [])],
        "coding_benchmarks": [os.path.basename(f) for f in (coding_benchmarks or [])],
        "overall_metrics": {}
    }
    
    # Aggregate metrics
    math_accuracies = [r["metrics"]["accuracy"] for r in results["math"].values()]
    coding_accuracies = [r["metrics"]["accuracy"] for r in results["coding"].values()]
    
    if math_accuracies:
        overall_results["overall_metrics"]["math_average_accuracy"] = sum(math_accuracies) / len(math_accuracies)
    
    if coding_accuracies:
        overall_results["overall_metrics"]["coding_average_accuracy"] = sum(coding_accuracies) / len(coding_accuracies)
    
    if math_accuracies and coding_accuracies:
        overall_results["overall_metrics"]["combined_average_accuracy"] = (
            sum(math_accuracies + coding_accuracies) / len(math_accuracies + coding_accuracies)
        )
    
    # Save overall results
    overall_file = os.path.join(output_dir, "overall_evaluation.json")
    with open(overall_file, 'w') as f:
        json.dump(overall_results, f, indent=2)
    
    logger.info(f"Saved overall evaluation results to {overall_file}")
    
    # Generate comparison chart
    _generate_comparison_chart(results, output_dir)
    
    return overall_results


def _generate_comparison_chart(results: Dict[str, Dict[str, Dict]], output_dir: str):
    """
    Generate a comparison chart of benchmark results.
    
    Args:
        results: Nested dictionary of evaluation results
        output_dir: Directory to save the chart
    """
    # Extract benchmark names and accuracies
    benchmarks = []
    accuracies = []
    categories = []
    
    for category in ["math", "coding"]:
        for benchmark_name, benchmark_results in results[category].items():
            accuracy = benchmark_results["metrics"]["accuracy"]
            benchmarks.append(benchmark_name)
            accuracies.append(accuracy)
            categories.append(category)
    
    if not benchmarks:
        return
    
    # Create DataFrame
    df = pd.DataFrame({
        "benchmark": benchmarks,
        "accuracy": accuracies,
        "category": categories
    })
    
    # Create chart
    plt.figure(figsize=(12, 6))
    
    # Use different colors for math vs coding
    colors = {'math': 'skyblue', 'coding': 'lightgreen'}
    
    # Create bar chart
    bars = plt.bar(
        df['benchmark'],
        df['accuracy'],
        color=[colors[cat] for cat in df['category']]
    )
    
    # Add labels and title
    plt.xlabel('Benchmark')
    plt.ylabel('Accuracy')
    plt.title('Benchmark Evaluation Results')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01,
            f'{height:.2f}',
            ha='center',
            va='bottom'
        )
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['math'], label='Math'),
        Patch(facecolor=colors['coding'], label='Coding')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    
    # Save chart
    chart_file = os.path.join(output_dir, "benchmark_comparison.png")
    plt.savefig(chart_file)
    logger.info(f"Saved benchmark comparison chart to {chart_file}")
    plt.close()


if __name__ == "__main__":
    # Example usage
    model_path = "checkpoints/mistral-7b-reasoning/final"
    
    math_benchmarks = [
        "evaluation/benchmarks/math_basic.json",
        "evaluation/benchmarks/aime_sample.json"
    ]
    
    coding_benchmarks = [
        "evaluation/benchmarks/coding_simple.json"
    ]
    
    results = run_evaluation(
        model_path=model_path,
        math_benchmarks=math_benchmarks,
        coding_benchmarks=coding_benchmarks
    )
    
    print(f"Evaluation completed with results: {results}")
