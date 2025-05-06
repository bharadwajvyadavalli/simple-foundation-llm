"""
Solution Generator for SimpleFoundation

This module generates reasoning-based solutions for math and coding problems
using smaller base models, which will be used as training data for our model.
"""

import os
import json
import logging
import torch
from typing import Dict, List, Optional, Union
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    pipeline
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SolutionGenerator:
    """
    Base class for generating reasoning-based solutions using smaller LLMs.
    """
    
    def __init__(
        self,
        model_name: str = "distilgpt2",  # Use a very small model for demonstration
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        device: Optional[str] = None,
        output_dir: str = "data/generated"
    ):
        """
        Initialize the solution generator.
        
        Args:
            model_name: Name of the model to use (HuggingFace model ID)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            device: Device to run the model on ("cuda", "cpu", etc.)
            output_dir: Directory to save generated solutions
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize model and tokenizer
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # For demonstration, we're using a tiny model
        # In practice, you'd use a larger model like Mistral-7B, Llama-2-7B, etc.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Set up generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True if temperature > 0 else False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # For efficient generation, create a pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
    
    def generate_solution(self, problem: str, prompt_template: str) -> str:
        """
        Generate a solution for a given problem.
        
        Args:
            problem: The problem statement
            prompt_template: Template for formatting the prompt
            
        Returns:
            Generated solution
        """
        # Format the prompt
        prompt = prompt_template.format(problem=problem)
        
        # Generate solution
        result = self.generator(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True if self.temperature > 0 else False,
            return_full_text=False
        )
        
        # Extract the generated text
        solution = result[0]['generated_text']
        
        return solution


class MathSolutionGenerator(SolutionGenerator):
    """Generator for math problem solutions."""
    
    def __init__(
        self,
        model_name: str = "distilgpt2",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        output_dir: str = "data/generated/math"
    ):
        """Initialize with math-specific settings."""
        super().__init__(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            output_dir=output_dir
        )
        
        # Define the chain-of-thought prompt template for math problems
        self.prompt_template = """
I need to solve the following math problem using detailed step-by-step reasoning:

{problem}

I'll solve this step-by-step:
"""

    def generate_solutions(self, problems_file: str, output_file: Optional[str] = None) -> List[Dict]:
        """
        Generate solutions for a list of math problems.
        
        Args:
            problems_file: Path to JSON file with problems
            output_file: Path to save the generated solutions
                
        Returns:
            List of problems with generated solutions
        """
        # Load problems
        with open(problems_file, 'r') as f:
            problems = json.load(f)
        
        logger.info(f"Generating solutions for {len(problems)} math problems")
        
        # Generate solutions for each problem
        for i, problem_data in enumerate(tqdm(problems, desc="Generating math solutions")):
            problem_text = problem_data["problem"]
            
            # Skip if solution already exists
            if "generated_solution" in problem_data:
                logger.debug(f"Solution already exists for problem {i+1}")
                continue
            
            # Generate solution
            generated_solution = self.generate_solution(problem_text, self.prompt_template)
            
            # Store the generated solution
            problem_data["generated_solution"] = generated_solution
        
        # Save the results
        if output_file is None:
            output_file = os.path.join(self.output_dir, "math_solutions.json")
        
        with open(output_file, 'w') as f:
            json.dump(problems, f, indent=2)
        
        logger.info(f"Saved {len(problems)} math solutions to {output_file}")
        
        return problems


class CodingSolutionGenerator(SolutionGenerator):
    """Generator for coding problem solutions."""
    
    def __init__(
        self,
        model_name: str = "distilgpt2",
        max_new_tokens: int = 1024,  # Coding solutions are typically longer
        temperature: float = 0.8,    # Slightly higher temperature for creativity
        output_dir: str = "data/generated/coding"
    ):
        """Initialize with coding-specific settings."""
        super().__init__(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            output_dir=output_dir
        )
        
        # Define the chain-of-thought prompt template for coding problems
        self.prompt_template = """
I need to solve the following coding problem using Python. I'll think through the solution step by step and then write the code:

{problem}

Let me analyze this problem:
"""

    def generate_solutions(self, problems_file: str, output_file: Optional[str] = None) -> List[Dict]:
        """
        Generate solutions for a list of coding problems.
        
        Args:
            problems_file: Path to JSON file with problems
            output_file: Path to save the generated solutions
                
        Returns:
            List of problems with generated solutions
        """
        # Load problems
        with open(problems_file, 'r') as f:
            problems = json.load(f)
        
        logger.info(f"Generating solutions for {len(problems)} coding problems")
        
        # Generate solutions for each problem
        for i, problem_data in enumerate(tqdm(problems, desc="Generating coding solutions")):
            problem_text = problem_data["problem"]
            
            # Skip if solution already exists
            if "generated_solution" in problem_data:
                logger.debug(f"Solution already exists for problem {i+1}")
                continue
            
            # Generate solution
            generated_solution = self.generate_solution(problem_text, self.prompt_template)
            
            # Store the generated solution
            problem_data["generated_solution"] = generated_solution
        
        # Save the results
        if output_file is None:
            output_file = os.path.join(self.output_dir, "coding_solutions.json")
        
        with open(output_file, 'w') as f:
            json.dump(problems, f, indent=2)
        
        logger.info(f"Saved {len(problems)} coding solutions to {output_file}")
        
        return problems


def generate_all_solutions(
    math_problems_file: str = "data/raw/math/math_processed.json",
    coding_problems_file: str = "data/raw/coding/apps_processed.json",
    model_name: str = "distilgpt2",
    output_dir: str = "data/generated"
):
    """
    Generate solutions for all problems.
    
    Args:
        math_problems_file: Path to math problems file
        coding_problems_file: Path to coding problems file
        model_name: Name of the model to use
        output_dir: Base directory to save generated solutions
    """
    # Generate math solutions
    math_generator = MathSolutionGenerator(
        model_name=model_name,
        output_dir=os.path.join(output_dir, "math")
    )
    
    math_solutions = math_generator.generate_solutions(
        math_problems_file,
        os.path.join(output_dir, "math/math_solutions.json")
    )
    
    # Generate coding solutions
    coding_generator = CodingSolutionGenerator(
        model_name=model_name,
        output_dir=os.path.join(output_dir, "coding")
    )
    
    coding_solutions = coding_generator.generate_solutions(
        coding_problems_file,
        os.path.join(output_dir, "coding/coding_solutions.json")
    )
    
    return {
        "math": {
            "count": len(math_solutions),
            "file": os.path.join(output_dir, "math/math_solutions.json")
        },
        "coding": {
            "count": len(coding_solutions),
            "file": os.path.join(output_dir, "coding/coding_solutions.json")
        }
    }


if __name__ == "__main__":
    # Example usage
    result = generate_all_solutions()
    print(f"Generated solutions summary: {result}")
