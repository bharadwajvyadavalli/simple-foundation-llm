"""
Quality Filtering Module for SimpleFoundation

This module implements rejection sampling to filter out low-quality 
generated solutions for both math and coding problems.
"""

import os
import re
import json
import logging
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QualityFilter:
    """Base class for quality filtering."""
    
    def __init__(self, output_dir: str = "data/filtered"):
        """
        Initialize the quality filter.
        
        Args:
            output_dir: Directory to save filtered data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def filter_solutions(self, solutions_file: str, output_file: Optional[str] = None) -> List[Dict]:
        """
        Filter solutions by quality.
        
        Args:
            solutions_file: Path to file with generated solutions
            output_file: Path to save the filtered solutions
            
        Returns:
            List of problems with high-quality solutions
        """
        raise NotImplementedError("Subclasses must implement filter_solutions")


class MathQualityFilter(QualityFilter):
    """Filter for math problem solutions."""
    
    def __init__(self, output_dir: str = "data/filtered/math"):
        """Initialize with math-specific output directory."""
        super().__init__(output_dir)
    
    def extract_final_answer(self, solution: str) -> Optional[str]:
        """
        Extract the final answer from a solution.
        
        Args:
            solution: Generated solution text
            
        Returns:
            Extracted answer or None if not found
        """
        # Look for patterns like "Final answer: X" or "Therefore, the answer is X"
        answer_patterns = [
            r"(?:final|the)\s+answer(?:\s+is)?:\s*([^\n.,]+)",
            r"(?:therefore|thus|so),?\s+(?:the)?\s*answer(?:\s+is)?:?\s*([^\n.,]+)",
            r"(?:therefore|thus|so),?\s+(?:we\s+(?:get|have|find))(?:\s+that)?:?\s*([^\n.,]+)",
            r"(?:in\s+conclusion),?\s+(?:the)?\s*(?:answer|value|result)(?:\s+is)?:?\s*([^\n.,]+)",
            r"(?:the\s+)?(?:answer|value|result)(?:\s+is)?:?\s*([^\n.,]+)"
        ]
        
        # Try each pattern
        for pattern in answer_patterns:
            matches = re.search(pattern, solution, re.IGNORECASE)
            if matches:
                answer = matches.group(1).strip()
                return answer
        
        # Last attempt: try to extract the last number in the solution
        numbers = re.findall(r'\b\d+(?:/\d+)?\b', solution)
        if numbers:
            return numbers[-1]
        
        return None
    
    def is_answer_correct(self, generated_answer: Optional[str], correct_answer: str) -> bool:
        """
        Check if the generated answer matches the correct answer.
        
        Args:
            generated_answer: Extracted answer from generated solution
            correct_answer: Known correct answer
            
        Returns:
            True if the answers match, False otherwise
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
    
    def filter_solutions(self, solutions_file: str, output_file: Optional[str] = None) -> List[Dict]:
        """
        Filter math solutions by correctness of final answer.
        
        Args:
            solutions_file: Path to file with generated solutions
            output_file: Path to save the filtered solutions
            
        Returns:
            List of problems with correct solutions
        """
        # Load solutions
        with open(solutions_file, 'r') as f:
            problems = json.load(f)
        
        logger.info(f"Filtering {len(problems)} math solutions")
        
        filtered_problems = []
        correct_count = 0
        
        # Check each solution
        for problem_data in tqdm(problems, desc="Filtering math solutions"):
            # Skip if no generated solution
            if "generated_solution" not in problem_data:
                logger.warning(f"No generated solution for problem {problem_data.get('id', 'unknown')}")
                continue
            
            # Extract answer from generated solution
            generated_solution = problem_data["generated_solution"]
            extracted_answer = self.extract_final_answer(generated_solution)
            
            # Get correct answer
            correct_answer = problem_data.get("answer", "")
            
            # Check correctness
            is_correct = self.is_answer_correct(extracted_answer, correct_answer)
            
            # Add filtering results to problem data
            problem_data["extracted_answer"] = extracted_answer
            problem_data["is_correct"] = is_correct
            
            # Keep track of correct count
            if is_correct:
                correct_count += 1
                filtered_problems.append(problem_data)
        
        # Calculate accuracy
        accuracy = correct_count / len(problems) if problems else 0
        logger.info(f"Filtering accuracy: {accuracy:.2%} ({correct_count}/{len(problems)})")
        
        # Save filtered problems
        if output_file is None:
            output_file = os.path.join(self.output_dir, "math_filtered.json")
        
        with open(output_file, 'w') as f:
            json.dump(filtered_problems, f, indent=2)
        
        logger.info(f"Saved {len(filtered_problems)} filtered math solutions to {output_file}")
        
        return filtered_problems


class CodingQualityFilter(QualityFilter):
    """Filter for coding problem solutions."""
    
    def __init__(self, output_dir: str = "data/filtered/coding"):
        """Initialize with coding-specific output directory."""
        super().__init__(output_dir)
    
    def extract_python_code(self, solution: str) -> Optional[str]:
        """
        Extract Python code from a solution.
        
        Args:
            solution: Generated solution text
            
        Returns:
            Extracted Python code or None if not found
        """
        # Look for code blocks
        code_block_pattern = r"```(?:python)?(.*?)```"
        code_blocks = re.findall(code_block_pattern, solution, re.DOTALL)
        
        if code_blocks:
            # Return the last code block (usually the final solution)
            return code_blocks[-1].strip()
        
        # If no code blocks, try to extract based on indentation and Python keywords
        lines = solution.split("\n")
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
    
    def test_solution(self, code: str, test_cases: List[Dict]) -> Tuple[bool, List[bool]]:
        """
        Test a coding solution against test cases.
        
        Args:
            code: Extracted Python code
            test_cases: List of test cases with input and expected output
            
        Returns:
            Tuple of (overall_success, list_of_individual_test_results)
        """
        if not code or not test_cases:
            return False, []
        
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
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
    
    def filter_solutions(self, solutions_file: str, output_file: Optional[str] = None) -> List[Dict]:
        """
        Filter coding solutions by testing against test cases.
        
        Args:
            solutions_file: Path to file with generated solutions
            output_file: Path to save the filtered solutions
            
        Returns:
            List of problems with correct solutions
        """
        # Load solutions
        with open(solutions_file, 'r') as f:
            problems = json.load(f)
        
        logger.info(f"Filtering {len(problems)} coding solutions")
        
        filtered_problems = []
        correct_count = 0
        
        # Check each solution
        for problem_data in tqdm(problems, desc="Filtering coding solutions"):
            # Skip if no generated solution
            if "generated_solution" not in problem_data:
                logger.warning(f"No generated solution for problem {problem_data.get('id', 'unknown')}")
                continue
            
            # Skip if no test cases
            test_cases = problem_data.get("test_cases", [])
            if not test_cases:
                logger.warning(f"No test cases for problem {problem_data.get('id', 'unknown')}")
                problem_data["is_correct"] = False
                continue
            
            # Extract code from generated solution
            generated_solution = problem_data["generated_solution"]
            extracted_code = self.extract_python_code(generated_solution)
            
            if not extracted_code:
                logger.warning(f"No code extracted for problem {problem_data.get('id', 'unknown')}")
                problem_data["is_correct"] = False
                continue
            
            # Test the solution
            is_correct, test_results = self.test_solution(extracted_code, test_cases)
            
            # Add filtering results to problem data
            problem_data["extracted_code"] = extracted_code
            problem_data["is_correct"] = is_correct
            problem_data["test_results"] = test_results
            
            # Keep track of correct count
            if is_correct:
                correct_count += 1
                filtered_problems.append(problem_data)
        
        # Calculate accuracy
        accuracy = correct_count / len(problems) if problems else 0
        logger.info(f"Filtering accuracy: {accuracy:.2%} ({correct_count}/{len(problems)})")
        
        # Save filtered problems
        if output_file is None:
            output_file = os.path.join(self.output_dir, "coding_filtered.json")
        
        with open(output_file, 'w') as f:
            json.dump(filtered_problems, f, indent=2)
        
        logger.info(f"Saved {len(filtered_problems)} filtered coding solutions to {output_file}")
        
        return filtered_problems


def filter_all_solutions(
    math_solutions_file: str = "data/generated/math/math_solutions.json",
    coding_solutions_file: str = "data/generated/coding/coding_solutions.json",
    output_dir: str = "data/filtered"
):
    """
    Filter all generated solutions.
    
    Args:
        math_solutions_file: Path to math solutions file
        coding_solutions_file: Path to coding solutions file
        output_dir: Base directory to save filtered solutions
    """
    # Filter math solutions
    math_filter = MathQualityFilter(
        output_dir=os.path.join(output_dir, "math")
    )
    
    math_filtered = math_filter.filter_solutions(
        math_solutions_file,
        os.path.join(output_dir, "math/math_filtered.json")
    )
    
    # Filter coding solutions
    coding_filter = CodingQualityFilter(
        output_dir=os.path.join(output_dir, "coding")
    )
    
    coding_filtered = coding_filter.filter_solutions(
        coding_solutions_file,
        os.path.join(output_dir, "coding/coding_filtered.json")
    )
    
    return {
        "math": {
            "count": len(math_filtered),
            "file": os.path.join(output_dir, "math/math_filtered.json"),
            "accuracy": len(math_filtered) / len(json.load(open(math_solutions_file)))
        },
        "coding": {
            "count": len(coding_filtered),
            "file": os.path.join(output_dir, "coding/coding_filtered.json"),
            "accuracy": len(coding_filtered) / len(json.load(open(coding_solutions_file)))
        }
    }


if __name__ == "__main__":
    # Example usage
    result = filter_all_solutions()
    print(f"Filtering results: {result}")
