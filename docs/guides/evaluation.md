# Evaluation Documentation

## Overview

The evaluation module in SimpleFoundation provides a comprehensive framework for assessing the performance of trained models on reasoning tasks. This module helps you quantify model improvements, conduct ablation studies, and generate detailed reports on model capabilities.

![Evaluation Architecture](../images/architecture.png)

## Key Components

### Benchmark Evaluator

```
evaluation/benchmarks/benchmark_evaluator.py
```

The `BenchmarkEvaluator` base class provides the foundation for specialized evaluators, handling:

- Model loading and configuration
- Generation settings
- Result collection and aggregation
- Report generation

### Specialized Evaluators

1. **MathBenchmarkEvaluator**:
   - Evaluates models on mathematical reasoning
   - Extracts and validates answers
   - Calculates accuracy metrics

2. **CodingBenchmarkEvaluator**:
   - Evaluates models on coding reasoning
   - Extracts code from responses
   - Tests code execution against test cases
   - Measures correctness and efficiency

## Evaluation Process

### 1. Initialize Evaluator

```python
from evaluation.benchmarks.benchmark_evaluator import MathBenchmarkEvaluator

evaluator = MathBenchmarkEvaluator(
    model_path="training/checkpoints/final",
    output_dir="evaluation/results/math",
    load_in_4bit=True  # Use quantization for efficiency
)
```

### 2. Run Evaluation

```python
# Evaluate on a benchmark
results = evaluator.evaluate_benchmark(
    benchmark_file="evaluation/benchmarks/math_basic.json"
)
```

### 3. Analyze Results

```python
# Print summary metrics
print(f"Accuracy: {results['metrics']['accuracy']:.2%}")
print(f"Correct: {results['metrics']['correct_count']}/{results['metrics']['total_count']}")

# Detailed analysis available in results['problems']
for problem in results['problems']:
    if not problem['correct']:
        print(f"Problem {problem['id']} failed:")
        print(f"Expected: {problem['expected_answer']}")
        print(f"Got: {problem.get('extracted_answer', 'No answer extracted')}")
```

## Benchmarks

### Math Benchmarks

SimpleFoundation includes several math benchmarks of varying difficulty:

1. **Math-Basic**: Elementary algebra, geometry, and arithmetic
2. **Math-Intermediate**: High school level problems
3. **Math-Advanced**: Competition-level problems (e.g., AIME)

Example benchmark format:
```json
[
  {
    "id": "algebra_linear_1",
    "problem": "Solve for x: 2x + 5 = 13",
    "answer": "4",
    "category": "algebra",
    "level": "basic"
  },
  {
    "id": "geometry_triangle_1",
    "problem": "A triangle has sides of length 3, 4, and 5. What is its area?",
    "answer": "6",
    "category": "geometry",
    "level": "basic"
  }
]
```

### Coding Benchmarks

Coding benchmarks test the model's ability to write correct and efficient code:

1. **Coding-Simple**: Basic algorithms and data structures
2. **Coding-Medium**: Intermediate programming challenges
3. **Coding-Hard**: Complex algorithms and optimizations

Example benchmark format:
```json
[
  {
    "id": "string_palindrome",
    "problem": "Write a function to check if a string is a palindrome.",
    "difficulty": "simple",
    "test_cases": [
      {
        "input": "racecar",
        "expected_output": "True"
      },
      {
        "input": "hello",
        "expected_output": "False"
      }
    ]
  }
]
```

## Running Evaluations

### Command-Line Interface

```bash
# Run math evaluation
python -m evaluation.benchmarks.benchmark_evaluator \
    --model_path training/checkpoints/final \
    --math_benchmarks evaluation/benchmarks/math_basic.json evaluation/benchmarks/math_intermediate.json \
    --output_dir evaluation/results \
    --load_in_4bit
```

```bash
# Run coding evaluation
python -m evaluation.benchmarks.benchmark_evaluator \
    --model_path training/checkpoints/final \
    --coding_benchmarks evaluation/benchmarks/coding_simple.json \
    --output_dir evaluation/results \
    --load_in_4bit
```

### Combined Evaluation

```bash
# Run both math and coding evaluations
python -m evaluation.benchmarks.benchmark_evaluator \
    --model_path training/checkpoints/final \
    --math_benchmarks evaluation/benchmarks/math_basic.json \
    --coding_benchmarks evaluation/benchmarks/coding_simple.json \
    --output_dir evaluation/results \
    --load_in_4bit
```

## Interpreting Results

### Results Structure

```json
{
  "benchmark_name": "math_basic",
  "model_path": "training/checkpoints/final",
  "timestamp": "2025-01-10 14:30:22",
  "metrics": {
    "accuracy": 0.83,
    "correct_count": 83,
    "total_count": 100
  },
  "problems": [
    {
      "id": "algebra_linear_1",
      "problem": "Solve for x: 2x + 5 = 13",
      "expected_answer": "4",
      "response": "I'll solve this step by step...",
      "extracted_answer": "4",
      "correct": true
    },
    ...
  ]
}
```

### Visualizations

The evaluator automatically generates visualizations:

1. **Bar charts** showing accuracy across different benchmarks
2. **Comparison plots** between different models or model versions
3. **Performance breakdown** by problem category and difficulty

These visualizations are saved alongside the JSON results.

## Error Analysis

The evaluation module provides tools for detailed error analysis:

### 1. Error Categories

```python
from evaluation.analysis.error_analyzer import categorize_errors

# Analyze math errors
error_analysis = categorize_errors(results)

print("Error categories:")
for category, count in error_analysis.items():
    print(f"{category}: {count}")
```

Common error categories:
- Calculation errors
- Conceptual misunderstandings
- Formatting issues
- Incomplete reasoning

### 2. Response Analysis

```python
from evaluation.analysis.response_analyzer import analyze_reasoning_steps

# Analyze reasoning steps
reasoning_analysis = analyze_reasoning_steps(results)

print(f"Average number of reasoning steps: {reasoning_analysis['avg_steps']}")
print(f"Problems with incomplete reasoning: {reasoning_analysis['incomplete_count']}")
```

## Conducting Ablation Studies

Ablation studies help understand which components of your model and training pipeline contribute most to performance:

```python
from evaluation.ablation.ablation_study import run_ablation_study

# Compare different model configurations
configurations = [
    {"name": "baseline", "model_path": "checkpoints/baseline"},
    {"name": "lora_r8", "model_path": "checkpoints/lora_r8"},
    {"name": "lora_r16", "model_path": "checkpoints/lora_r16"},
    {"name": "more_math_data", "model_path": "checkpoints/more_math_data"}
]

ablation_results = run_ablation_study(
    configurations=configurations,
    benchmarks=["math_basic", "coding_simple"]
)

# Results are automatically visualized
```

## Comparing to Baseline Models

It's important to compare your fine-tuned model against baselines:

1. **Base Model**: The original pre-trained model before fine-tuning
2. **Previous Version**: Your previous best model
3. **State-of-the-Art**: Published models with similar capabilities

```python
from evaluation.comparison.model_comparison import compare_models

models = [
    {"name": "Base Model", "path": "mistralai/Mistral-7B-v0.1"},
    {"name": "SimpleFoundation", "path": "checkpoints/final"},
    {"name": "Previous Best", "path": "checkpoints/previous_best"}
]

comparison = compare_models(
    models=models,
    benchmarks=["math_basic", "math_intermediate", "coding_simple"]
)
```

## Creating Custom Benchmarks

You can easily create custom benchmarks to test specific capabilities:

### 1. Custom Math Benchmark

```python
# Create a custom math benchmark
import json

custom_math_problems = [
    {
        "id": "custom_math_1",
        "problem": "If f(x) = 2x + 3 and g(x) = x^2, find f(g(2)).",
        "answer": "11",
        "category": "algebra",
        "level": "intermediate"
    },
    # Add more problems...
]

# Save the benchmark
with open("evaluation/benchmarks/custom_math.json", "w") as f:
    json.dump(custom_math_problems, f, indent=2)
```

### 2. Custom Coding Benchmark

```python
# Create a custom coding benchmark
custom_coding_problems = [
    {
        "id": "custom_coding_1",
        "problem": "Implement a function to find the longest increasing subsequence in an array.",
        "difficulty": "medium",
        "test_cases": [
            {
                "input": "[10, 22, 9, 33, 21, 50, 41, 60]",
                "expected_output": "5"  # Length of LIS: [10, 22, 33, 50, 60]
            },
            # Add more test cases...
        ]
    },
    # Add more problems...
]

# Save the benchmark
with open("evaluation/benchmarks/custom_coding.json", "w") as f:
    json.dump(custom_coding_problems, f, indent=2)
```

## Performance Considerations

### Memory Optimization

- Use 4-bit or 8-bit quantization for efficient inference
- Enable flash attention if available
- Use model offloading for very large models

### Speed Optimization

- Batch similar problems together
- Use smaller context lengths when possible
- Run evaluations in parallel on multiple GPUs

## Best Practices

1. **Regular Evaluation**: Evaluate your model regularly during training to track progress
2. **Diverse Benchmarks**: Use a variety of benchmarks covering different skills
3. **Error Analysis**: Analyze failures to guide further improvements
4. **Version Control**: Keep track of model versions and their performance
5. **Human Verification**: Supplement automatic evaluation with human review of selected outputs
6. **Data Leakage**: Ensure evaluation benchmarks don't overlap with training data

## Troubleshooting

### Common Issues

1. **Inconsistent Answer Extraction**:
   - Adjust answer extraction patterns in `extract_answer` methods
   - Ensure model output follows expected format

2. **Code Testing Failures**:
   - Check test case formatting
   - Increase test timeout for complex problems
   - Improve code extraction logic

3. **Memory Issues**:
   - Enable more aggressive quantization
   - Evaluate in smaller batches
   - Reduce model size or precision

4. **Result Reporting Issues**:
   - Check output directory permissions
   - Ensure matplotlib and other visualization dependencies are installed
