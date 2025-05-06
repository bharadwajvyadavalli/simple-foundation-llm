# Data Pipeline Documentation

## Overview

The data pipeline in SimpleFoundation handles the collection, generation, filtering, and formatting of training data. This critical component ensures that we have high-quality examples that teach the model how to solve reasoning problems step-by-step.

## Pipeline Stages

### 1. Data Collection

The collection stage gathers math and coding problems from various sources.

#### Math Datasets

```
data/collectors/dataset_collector.py (MathDatasetCollector)
```

- **MATH Dataset**: Comprehensive collection of mathematics problems across different domains and difficulty levels
- **AIME**: American Invitational Mathematics Examination problems for challenging algebra and geometry
- **Sample Data**: Small sample datasets for testing

#### Coding Datasets

```
data/collectors/dataset_collector.py (CodingDatasetCollector)
```

- **APPS**: Automated Programming Progress Standard benchmark for coding problems
- **Sample Data**: Simplified coding problems with test cases

#### Example Usage:

```python
# Collect all datasets
from data.collectors.dataset_collector import download_all_datasets

# This will create data/raw/{math,coding}/ directories with problem datasets
result = download_all_datasets()
print(f"Collected {result['math']['count']} math problems and {result['coding']['count']} coding problems")

# Collect only specific datasets
from data.collectors.dataset_collector import MathDatasetCollector

math_collector = MathDatasetCollector()
math_data = math_collector.download_and_prepare(["math"])
```

### 2. Solution Generation

This stage uses smaller base models to generate step-by-step solutions for collected problems.

```
data/generators/solution_generator.py
```

- Uses transformers library to load and run inference with smaller models
- Generates reasoning chains and solutions for each problem
- Formats prompts to encourage step-by-step thinking
- Supports both math and coding problems with domain-specific prompting

#### Example Usage:

```python
# Generate solutions for math problems
from data.generators.solution_generator import MathSolutionGenerator

generator = MathSolutionGenerator(
    model_name="distilgpt2",  # Small model for demo, use larger models like "mistralai/Mistral-7B-v0.1" in practice
    max_new_tokens=512,
    temperature=0.7
)

math_solutions = generator.generate_solutions(
    problems_file="data/raw/math/math_processed.json",
    output_file="data/generated/math/math_solutions.json"
)
```

#### Key Parameters for Generation:

| Parameter | Description | Recommended Value |
|-----------|-------------|------------------|
| model_name | HuggingFace model ID | distilgpt2 (demo), Mistral-7B-v0.1 (production) |
| max_new_tokens | Max tokens to generate | 512 (math), 1024 (coding) |
| temperature | Randomness in generation | 0.7 (math), 0.8 (coding) |
| prompt_template | Template for problem formatting | See code for examples |

### 3. Quality Filtering

This stage implements rejection sampling to filter out incorrect or low-quality solutions.

```
data/filters/quality_filter.py
```

- **Math Solutions**: Extracts final answer and compares to known solutions
- **Coding Solutions**: Runs code against test cases to verify correctness
- Rejects solutions that don't meet quality criteria

#### Example Usage:

```python
# Filter generated solutions
from data.filters.quality_filter import filter_all_solutions

# This will create data/filtered/{math,coding}/ directories with filtered solutions
result = filter_all_solutions(
    math_solutions_file="data/generated/math/math_solutions.json",
    coding_solutions_file="data/generated/coding/coding_solutions.json"
)

print(f"Math filtering accuracy: {result['math']['accuracy']:.2%}")
print(f"Coding filtering accuracy: {result['coding']['accuracy']:.2%}")
```

#### Key Quality Metrics:

- **Math Problems**: Final answer correctness, solution length, reasoning steps
- **Coding Problems**: Passing test cases, code efficiency, explanation clarity

### 4. Data Formatting

This final stage formats filtered solutions into a clean, consistent format for training.

```
data/preprocessors/data_formatter.py
```

- Structures solutions with clear reasoning steps
- Formats prompts and responses consistently
- Creates instruction-following examples
- Combines datasets into a unified format

#### Example Usage:

```python
# Format solutions for training
from data.preprocessors.data_formatter import format_all_data

# This will create data/formatted/{math,coding}/ directories and a combined file
result = format_all_data(
    math_filtered_file="data/filtered/math/math_filtered.json",
    coding_filtered_file="data/filtered/coding/coding_filtered.json"
)

print(f"Created {result['combined']['count']} formatted examples for training")
```

#### Formatting Templates:

**Math Problems**:
```
You are a helpful AI math assistant. You excel at solving math problems by breaking them down into clear, logical steps.

Problem:
{problem}

Solve this problem step by step, showing your reasoning clearly.

I'll solve this step by step.

Step 1: [First reasoning step]
Step 2: [Second reasoning step]
...
Step N: [Final reasoning step]

Therefore, the answer is {answer}.
```

**Coding Problems**:
```
You are an expert coding assistant. You excel at solving programming problems with clear step-by-step reasoning.

Problem:
{problem}

Solve this coding problem step by step in Python, explaining your thought process and then providing the final working code.

I'll solve this step by step.

Step 1: [First reasoning step]
Step 2: [Second reasoning step]
...
Step N: [Final reasoning step]

Here's my final solution:

```python
[final_code]
```

This code works by [explanation].
```

## Running the Full Pipeline

You can run the complete pipeline with a single script:

```bash
bash scripts/run_pipeline.sh
```

This will:
1. Download datasets
2. Generate solutions
3. Filter solutions
4. Format data for training

Alternatively, run the components individually:

```bash
# 1. Download datasets
python -m data.collectors.dataset_collector

# 2. Generate solutions
python -m data.generators.solution_generator --model_name distilgpt2

# 3. Filter solutions
python -m data.filters.quality_filter

# 4. Format data
python -m data.preprocessors.data_formatter
```

## Pipeline Configuration

The pipeline can be configured by modifying the following files:

- `data/collectors/dataset_collector.py`: Add new data sources
- `data/generators/solution_generator.py`: Change generation models or parameters
- `data/filters/quality_filter.py`: Adjust quality thresholds
- `data/preprocessors/data_formatter.py`: Modify formatting templates

## Best Practices

- **Data Balance**: Maintain a balance between math and coding problems
- **Diversity**: Include problems of varying difficulties
- **Quality over Quantity**: Fewer high-quality examples are better than many low-quality ones
- **Consistent Formatting**: Ensure consistent formatting for better model training
- **Iterative Improvement**: Run ablation studies to determine the optimal data mix

## Troubleshooting

Common issues and solutions:

1. **Low Generation Quality**: 
   - Use a larger base model (7B+ parameters)
   - Adjust temperature (0.6-0.8 range works best)
   - Improve prompt templates

2. **Low Filtering Pass Rate**:
   - Check answer format expectations
   - Adjust extraction patterns
   - Relax matching criteria slightly

3. **Memory Issues**:
   - Process data in smaller batches
   - Use streaming for large datasets

4. **Slow Generation**:
   - Enable quantization (4-bit or 8-bit)
   - Use flash attention if available
   - Distribute generation across multiple GPUs
