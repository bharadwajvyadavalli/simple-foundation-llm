"""
Data Formatter Module for SimpleFoundation

This module formats filtered solutions into a clean, consistent format
that is well-suited for training foundational models with reasoning capabilities.
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataFormatter:
    """Base class for data formatting."""
    
    def __init__(self, output_dir: str = "data/formatted"):
        """
        Initialize the data formatter.
        
        Args:
            output_dir: Directory to save formatted data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def format_data(self, filtered_file: str, output_file: Optional[str] = None) -> List[Dict]:
        """
        Format filtered data into training examples.
        
        Args:
            filtered_file: Path to file with filtered solutions
            output_file: Path to save the formatted data
            
        Returns:
            List of formatted training examples
        """
        raise NotImplementedError("Subclasses must implement format_data")


class MathDataFormatter(DataFormatter):
    """Formatter for math problem solutions."""
    
    def __init__(self, output_dir: str = "data/formatted/math"):
        """Initialize with math-specific output directory."""
        super().__init__(output_dir)
        
        # Define templates for formatting
        self.prompt_template = """
You are a helpful AI math assistant. You excel at solving math problems by breaking them down into clear, logical steps.

Problem:
{problem}

Solve this problem step by step, showing your reasoning clearly.
"""

        self.response_template = """
I'll solve this step by step.

{solution}

Therefore, the answer is {answer}.
"""
    
    def clean_solution(self, solution: str, answer: str) -> str:
        """
        Clean and structure a solution.
        
        Args:
            solution: Raw generated solution
            answer: Correct answer
            
        Returns:
            Cleaned and structured solution
        """
        # Remove any existing final answer statements
        solution = re.sub(
            r"(?:final|the)\s+answer(?:\s+is)?:.*$", 
            "", 
            solution, 
            flags=re.IGNORECASE | re.MULTILINE
        )
        
        solution = re.sub(
            r"(?:therefore|thus|so),?\s+(?:the)?\s*answer(?:\s+is)?:?.*$", 
            "", 
            solution, 
            flags=re.IGNORECASE | re.MULTILINE
        )
        
        # Clean up formatting and structure the solution into clear steps
        lines = solution.strip().split("\n")
        clean_lines = []
        
        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                continue
            
            # Clean up step numbering and formatting
            line = re.sub(r"^Step\s*\d+[:.]\s*", "", line)
            line = re.sub(r"^\d+[:.]\s*", "", line)
            
            clean_lines.append(line)
        
        # Join lines with proper spacing
        clean_solution = "\n".join(clean_lines)
        
        # Add step numbering for clarity
        final_lines = []
        current_step = 1
        
        for line in clean_solution.split("\n"):
            if line.strip():
                if current_step == 1 or re.match(r".*[:.?!]$", final_lines[-1] if final_lines else ""):
                    final_lines.append(f"Step {current_step}: {line}")
                    current_step += 1
                else:
                    final_lines.append(line)
        
        return "\n".join(final_lines)
    
    def format_data(self, filtered_file: str, output_file: Optional[str] = None) -> List[Dict]:
        """
        Format math solutions into training examples with consistent structure.
        
        Args:
            filtered_file: Path to file with filtered solutions
            output_file: Path to save the formatted data
            
        Returns:
            List of formatted training examples
        """
        # Load filtered solutions
        with open(filtered_file, 'r') as f:
            problems = json.load(f)
        
        logger.info(f"Formatting {len(problems)} math solutions")
        
        formatted_examples = []
        
        # Format each problem
        for problem_data in tqdm(problems, desc="Formatting math solutions"):
            # Skip if not marked as correct
            if not problem_data.get("is_correct", False):
                continue
            
            problem_text = problem_data["problem"]
            generated_solution = problem_data["generated_solution"]
            answer = problem_data["answer"]
            
            # Clean and structure the solution
            cleaned_solution = self.clean_solution(generated_solution, answer)
            
            # Format into prompt and response
            prompt = self.prompt_template.format(problem=problem_text)
            response = self.response_template.format(solution=cleaned_solution, answer=answer)
            
            # Create training example
            formatted_examples.append({
                "id": problem_data.get("id", f"math_{len(formatted_examples)}"),
                "prompt": prompt.strip(),
                "response": response.strip(),
                "category": problem_data.get("category", "math"),
                "level": problem_data.get("level", "unknown")
            })
        
        # Save formatted examples
        if output_file is None:
            output_file = os.path.join(self.output_dir, "math_formatted.json")
        
        with open(output_file, 'w') as f:
            json.dump(formatted_examples, f, indent=2)
        
        logger.info(f"Saved {len(formatted_examples)} formatted math examples to {output_file}")
        
        return formatted_examples


class CodingDataFormatter(DataFormatter):
    """Formatter for coding problem solutions."""
    
    def __init__(self, output_dir: str = "data/formatted/coding"):
        """Initialize with coding-specific output directory."""
        super().__init__(output_dir)
        
        # Define templates for formatting
        self.prompt_template = """
You are an expert coding assistant. You excel at solving programming problems with clear step-by-step reasoning.

Problem:
{problem}

Solve this coding problem step by step in Python, explaining your thought process and then providing the final working code.
"""

        self.response_template = """
I'll solve this step by step.

{reasoning}

Here's my final solution:

```python
{code}
```

This code works by {explanation}.
"""
    
    def extract_reasoning_and_code(self, solution: str) -> Tuple[str, str]:
        """
        Extract reasoning and code from a solution.
        
        Args:
            solution: Raw generated solution
            
        Returns:
            Tuple of (reasoning, code)
        """
        # Extract code blocks
        code_block_pattern = r"```(?:python)?(.*?)```"
        code_blocks = re.findall(code_block_pattern, solution, re.DOTALL)
        
        code = ""
        if code_blocks:
            # Use the last code block as the final solution
            code = code_blocks[-1].strip()
            
            # Remove code blocks from solution to get reasoning
            reasoning = re.sub(code_block_pattern, "", solution, flags=re.DOTALL)
        else:
            # If no code blocks, try to separate reasoning and code based on patterns
            lines = solution.split("\n")
            reasoning_lines = []
            code_lines = []
            in_code = False
            
            for line in lines:
                if line.strip().startswith("def ") or line.strip() == "```" or line.strip() == "```python":
                    in_code = True
                
                if in_code:
                    code_lines.append(line)
                else:
                    reasoning_lines.append(line)
            
            reasoning = "\n".join(reasoning_lines)
            code = "\n".join(code_lines)
        
        # Clean up reasoning
        reasoning = reasoning.strip()
        
        # Structure reasoning into steps if not already
        if not re.search(r"step\s*\d+", reasoning, re.IGNORECASE):
            reasoning_lines = reasoning.split("\n")
            structured_lines = []
            step_counter = 1
            
            for line in reasoning_lines:
                if line.strip():
                    if step_counter == 1 or (structured_lines and re.search(r'[.!?], structured_lines[-1])):
                        line = f"Step {step_counter}: {line}"
                        step_counter += 1
                    structured_lines.append(line)
            
            reasoning = "\n".join(structured_lines)
        
        return reasoning, code
    
    def format_data(self, filtered_file: str, output_file: Optional[str] = None) -> List[Dict]:
        """
        Format coding solutions into training examples with consistent structure.
        
        Args:
            filtered_file: Path to file with filtered solutions
            output_file: Path to save the formatted data
            
        Returns:
            List of formatted training examples
        """
        # Load filtered solutions
        with open(filtered_file, 'r') as f:
            problems = json.load(f)
        
        logger.info(f"Formatting {len(problems)} coding solutions")
        
        formatted_examples = []
        
        # Format each problem
        for problem_data in tqdm(problems, desc="Formatting coding solutions"):
            # Skip if not marked as correct
            if not problem_data.get("is_correct", False):
                continue
            
            problem_text = problem_data["problem"]
            generated_solution = problem_data["generated_solution"]
            extracted_code = problem_data.get("extracted_code", "")
            
            # Extract reasoning and code
            reasoning, code = self.extract_reasoning_and_code(generated_solution)
            
            # If code extraction failed but we have extracted_code, use that
            if not code.strip() and extracted_code:
                code = extracted_code
            
            # Create explanation from the reasoning
            explanation = "implementing the algorithm described in my reasoning"
            match = re.search(r"Step \d+: (.*?)(?:\.|$)", reasoning)
            if match:
                explanation = match.group(1).lower()
            
            # Format into prompt and response
            prompt = self.prompt_template.format(problem=problem_text)
            response = self.response_template.format(
                reasoning=reasoning,
                code=code,
                explanation=explanation
            )
            
            # Create training example
            formatted_examples.append({
                "id": problem_data.get("id", f"coding_{len(formatted_examples)}"),
                "prompt": prompt.strip(),
                "response": response.strip(),
                "difficulty": problem_data.get("difficulty", "unknown")
            })
        
        # Save formatted examples
        if output_file is None:
            output_file = os.path.join(self.output_dir, "coding_formatted.json")
        
        with open(output_file, 'w') as f:
            json.dump(formatted_examples, f, indent=2)
        
        logger.info(f"Saved {len(formatted_examples)} formatted coding examples to {output_file}")
        
        return formatted_examples


def format_all_data(
    math_filtered_file: str = "data/filtered/math/math_filtered.json",
    coding_filtered_file: str = "data/filtered/coding/coding_filtered.json",
    output_dir: str = "data/formatted"
):
    """
    Format all filtered data for training.
    
    Args:
        math_filtered_file: Path to filtered math solutions
        coding_filtered_file: Path to filtered coding solutions
        output_dir: Base directory to save formatted data
    """
    # Format math data
    math_formatter = MathDataFormatter(
        output_dir=os.path.join(output_dir, "math")
    )
    
    math_formatted = math_formatter.format_data(
        math_filtered_file,
        os.path.join(output_dir, "math/math_formatted.json")
    )
    
    # Format coding data
    coding_formatter = CodingDataFormatter(
        output_dir=os.path.join(output_dir, "coding")
    )
    
    coding_formatted = coding_formatter.format_data(
        coding_filtered_file,
        os.path.join(output_dir, "coding/coding_formatted.json")
    )
    
    # Create combined dataset with both math and coding examples
    combined_formatted = math_formatted + coding_formatted
    
    combined_output_file = os.path.join(output_dir, "combined_formatted.json")
    with open(combined_output_file, 'w') as f:
        json.dump(combined_formatted, f, indent=2)
    
    logger.info(f"Saved {len(combined_formatted)} combined formatted examples to {combined_output_file}")
    
    return {
        "math": {
            "count": len(math_formatted),
            "file": os.path.join(output_dir, "math/math_formatted.json")
        },
        "coding": {
            "count": len(coding_formatted),
            "file": os.path.join(output_dir, "coding/coding_formatted.json")
        },
        "combined": {
            "count": len(combined_formatted),
            "file": combined_output_file
        }
    }


if __name__ == "__main__":
    # Example usage
    result = format_all_data()
    print(f"Formatting results: {result}")
