"""
Demo UI for SimpleFoundation

This module provides a simple Streamlit UI for interacting with
the trained foundational model.
"""

import os
import json
import time
import logging
import requests
import streamlit as st
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleFoundationDemo:
    """Demo UI for SimpleFoundation models."""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        """
        Initialize the demo UI.
        
        Args:
            api_url: URL of the inference API server
        """
        self.api_url = api_url
        
        # Set up page config
        st.set_page_config(
            page_title="SimpleFoundation Demo",
            page_icon="🧠",
            layout="wide"
        )
    
    def run(self):
        """Run the demo UI."""
        # Header
        st.title("SimpleFoundation: Reasoning Model Demo")
        st.subheader("Explore your trained foundation model")
        
        # Check if API is available
        try:
            model_info = self._get_model_info()
            st.success(f"Connected to model: {model_info['model_name']} ({model_info['model_type']})")
            
            with st.expander("Model details"):
                st.json(model_info)
        except Exception as e:
            st.error(f"Error connecting to API at {self.api_url}: {str(e)}")
            st.info("Make sure the inference server is running and accessible")
            return
        
        # Sidebar for modes and options
        self._setup_sidebar()
        
        # Main content based on selected mode
        if st.session_state.mode == "chat":
            self._display_chat_interface()
        elif st.session_state.mode == "math":
            self._display_math_interface()
        elif st.session_state.mode == "coding":
            self._display_coding_interface()
        elif st.session_state.mode == "benchmark":
            self._display_benchmark_interface()
    
    def _setup_sidebar(self):
        """Set up the sidebar for mode selection and options."""
        with st.sidebar:
            st.header("Settings")
            
            # Mode selection
            if "mode" not in st.session_state:
                st.session_state.mode = "chat"
            
            mode = st.radio(
                "Interaction Mode",
                options=["chat", "math", "coding", "benchmark"],
                format_func=lambda x: x.capitalize(),
                index=["chat", "math", "coding", "benchmark"].index(st.session_state.mode)
            )
            st.session_state.mode = mode
            
            # Generation parameters
            st.subheader("Generation Settings")
            
            if "temperature" not in st.session_state:
                st.session_state.temperature = 0.7
            
            st.session_state.temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.temperature,
                step=0.1,
                help="Higher values make output more random, lower values more deterministic"
            )
            
            if "max_tokens" not in st.session_state:
                st.session_state.max_tokens = 512
            
            st.session_state.max_tokens = st.slider(
                "Max New Tokens",
                min_value=64,
                max_value=2048,
                value=st.session_state.max_tokens,
                step=64,
                help="Maximum number of tokens to generate"
            )
            
            if "top_p" not in st.session_state:
                st.session_state.top_p = 0.9
            
            st.session_state.top_p = st.slider(
                "Top-p",
                min_value=0.1,
                max_value=1.0,
                value=st.session_state.top_p,
                step=0.1,
                help="Nucleus sampling parameter"
            )
            
            # Clear conversation button
            if st.button("Clear Conversation"):
                if "messages" in st.session_state:
                    st.session_state.messages = []
                st.rerun()
    
    def _display_chat_interface(self):
        """Display the chat interface."""
        st.header("Chat Mode")
        st.write("Ask anything or try the reasoning capabilities of your model")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Input area
        if prompt := st.chat_input("Ask something..."):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.status("Generating response..."):
                    try:
                        response = self._generate_text(prompt)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.markdown(response)
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
    
    def _display_math_interface(self):
        """Display the math problem interface."""
        st.header("Math Reasoning Mode")
        st.write("Enter a math problem and see the step-by-step solution")
        
        # Math problem templates
        with st.expander("Sample problems", expanded=False):
            sample_problems = [
                "Find all positive integers n such that n^2 + 6n + 1 is divisible by n + 3.",
                "A right triangle has leg lengths a and b and hypotenuse length c, where a, b, and c are positive integers. Given that c=13 and ab=30, find a+b.",
                "Given that x = 2 + √3, compute the value of x^4 - 4x^2."
            ]
            
            for i, problem in enumerate(sample_problems):
                if st.button(f"Use Example {i+1}", key=f"math_example_{i}"):
                    st.session_state.math_problem = problem
                    st.rerun()
        
        # Input area
        if "math_problem" not in st.session_state:
            st.session_state.math_problem = ""
        
        math_problem = st.text_area(
            "Enter a math problem",
            value=st.session_state.math_problem,
            height=100
        )
        st.session_state.math_problem = math_problem
        
        if st.button("Solve Problem", disabled=not math_problem.strip()):
            with st.status("Solving the problem..."):
                # Format prompt with math-specific template
                prompt = f"""
You are a helpful AI math assistant. You excel at solving math problems by breaking them down into clear, logical steps.

Problem:
{math_problem}

Solve this problem step by step, showing your reasoning clearly.
"""
                try:
                    response = self._generate_text(prompt)
                    
                    # Display solution
                    st.subheader("Solution")
                    st.write(response)
                    
                    # Try to extract the final answer
                    import re
                    answer_match = re.search(r"(?:therefore|thus|so),?\s+(?:the)?\s*answer(?:\s+is)?:?\s*([^\n.,]+)", response, re.IGNORECASE)
                    if answer_match:
                        answer = answer_match.group(1).strip()
                        st.success(f"Final answer: {answer}")
                except Exception as e:
                    st.error(f"Error solving problem: {str(e)}")
    
    def _display_coding_interface(self):
        """Display the coding problem interface."""
        st.header("Coding Mode")
        st.write("Enter a coding problem and get a solution with explanations")
        
        # Coding problem templates
        with st.expander("Sample problems", expanded=False):
            sample_problems = [
                "Write a Python function to check if a string is a palindrome.",
                "Implement a function that finds the maximum subarray sum in an array of integers.",
                "Create a function to convert a decimal number to its binary representation."
            ]
            
            for i, problem in enumerate(sample_problems):
                if st.button(f"Use Example {i+1}", key=f"coding_example_{i}"):
                    st.session_state.coding_problem = problem
                    st.rerun()
        
        # Input area
        if "coding_problem" not in st.session_state:
            st.session_state.coding_problem = ""
        
        coding_problem = st.text_area(
            "Enter a coding problem",
            value=st.session_state.coding_problem,
            height=100
        )
        st.session_state.coding_problem = coding_problem
        
        if st.button("Solve Problem", disabled=not coding_problem.strip()):
            with st.status("Solving the problem..."):
                # Format prompt with coding-specific template
                prompt = f"""
You are an expert coding assistant. You excel at solving programming problems with clear step-by-step reasoning.

Problem:
{coding_problem}

Solve this coding problem step by step in Python, explaining your thought process and then providing the final working code.
"""
                try:
                    response = self._generate_text(prompt)
                    
                    # Display solution
                    st.subheader("Solution")
                    st.write(response)
                    
                    # Try to extract code blocks
                    import re
                    code_blocks = re.findall(r"```(?:python)?(.*?)```", response, re.DOTALL)
                    if code_blocks:
                        with st.expander("Extracted Code", expanded=True):
                            st.code(code_blocks[-1].strip(), language="python")
                except Exception as e:
                    st.error(f"Error solving problem: {str(e)}")
    
    def _display_benchmark_interface(self):
        """Display the benchmark interface."""
        st.header("Benchmark Mode")
        st.write("Run quick benchmarks to test the model's reasoning capabilities")
        
        # Benchmark selection
        benchmark_options = {
            "math_easy": "Math (Easy)",
            "math_medium": "Math (Medium)",
            "coding_easy": "Coding (Easy)"
        }
        
        selected_benchmark = st.selectbox(
            "Select a benchmark",
            options=list(benchmark_options.keys()),
            format_func=lambda x: benchmark_options[x]
        )
        
        # Number of examples
        num_examples = st.slider(
            "Number of examples",
            min_value=1,
            max_value=10,
            value=3,
            step=1
        )
        
        if st.button("Run Benchmark"):
            # Sample benchmark problems
            problems = self._get_sample_problems(selected_benchmark, num_examples)
            
            # Run benchmark
            with st.status(f"Running benchmark on {num_examples} {benchmark_options[selected_benchmark]} problems..."):
                results = []
                
                for i, problem in enumerate(problems):
                    st.write(f"Problem {i+1}/{len(problems)}")
                    
                    try:
                        # Construct prompt based on benchmark type
                        if selected_benchmark.startswith("math"):
                            prompt = f"""
You are a helpful AI math assistant. You excel at solving math problems by breaking them down into clear, logical steps.

Problem:
{problem['problem']}

Solve this problem step by step, showing your reasoning clearly.
"""
                        else:  # coding
                            prompt = f"""
You are an expert coding assistant. You excel at solving programming problems with clear step-by-step reasoning.

Problem:
{problem['problem']}

Solve this coding problem step by step in Python, explaining your thought process and then providing the final working code.
"""
                        
                        # Generate response
                        response = self._generate_text(prompt)
                        
                        # Evaluate response
                        # This is a simplified evaluation - in a real implementation,
                        # you would use proper metrics
                        results.append({
                            "problem": problem,
                            "response": response,
                            "correct": None  # Would require problem-specific evaluation
                        })
                    
                    except Exception as e:
                        st.error(f"Error on problem {i+1}: {str(e)}")
                        results.append({
                            "problem": problem,
                            "error": str(e),
                            "correct": False
                        })
            
            # Display results
            st.subheader("Benchmark Results")
            
            for i, result in enumerate(results):
                with st.expander(f"Problem {i+1}", expanded=i==0):
                    st.write("**Problem:**")
                    st.write(result["problem"]["problem"])
                    
                    st.write("**Model Response:**")
                    st.write(result["response"])
    
    def _get_model_info(self) -> Dict:
        """
        Get information about the model from the API.
        
        Returns:
            Model information
        """
        response = requests.get(f"{self.api_url}/model/info")
        response.raise_for_status()
        return response.json()
    
    def _generate_text(self, prompt: str) -> str:
        """
        Generate text from the model.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        request_data = {
            "prompt": prompt,
            "max_new_tokens": st.session_state.max_tokens,
            "temperature": st.session_state.temperature,
            "top_p": st.session_state.top_p,
            "do_sample": st.session_state.temperature > 0
        }
        
        start_time = time.time()
        response = requests.post(f"{self.api_url}/generate", json=request_data)
        response.raise_for_status()
        
        result = response.json()
        generation_time = time.time() - start_time
        
        logger.info(f"Generated {result['output_tokens']} tokens in {generation_time:.2f}s")
        
        return result["generated_text"]
    
    def _get_sample_problems(self, benchmark_type: str, count: int) -> List[Dict]:
        """
        Get sample problems for benchmarking.
        
        Args:
            benchmark_type: Type of benchmark ('math_easy', 'math_medium', 'coding_easy')
            count: Number of problems to get
            
        Returns:
            List of sample problems
        """
        # In a real implementation, these would be loaded from benchmark files
        sample_problems = {
            "math_easy": [
                {
                    "problem": "If a = 3 and b = 4, calculate a^2 + b^2.",
                    "answer": "25"
                },
                {
                    "problem": "Solve for x: 2x + 5 = 11",
                    "answer": "3"
                },
                {
                    "problem": "What is the sum of the first 10 positive integers?",
                    "answer": "55"
                },
                {
                    "problem": "Factor the expression: x^2 - 9",
                    "answer": "(x+3)(x-3)"
                },
                {
                    "problem": "Find the area of a circle with radius 5.",
                    "answer": "25π"
                }
            ],
            "math_medium": [
                {
                    "problem": "Solve the system of equations: 3x + 2y = 14, 2x - y = 1",
                    "answer": "x=4, y=1"
                },
                {
                    "problem": "Find all values of x such that log_3(x) + log_3(x-2) = 1",
                    "answer": "x=3"
                },
                {
                    "problem": "If sin(θ) = 3/5 and θ is in the first quadrant, what is cos(θ)?",
                    "answer": "4/5"
                },
                {
                    "problem": "Find the sum of the infinite geometric series: 4 + 2 + 1 + 1/2 + ...",
                    "answer": "8"
                },
                {
                    "problem": "Find the derivative of f(x) = x^3 * ln(x).",
                    "answer": "3x^2 * ln(x) + x^2"
                }
            ],
            "coding_easy": [
                {
                    "problem": "Write a Python function to check if a string is a palindrome.",
                    "test_cases": []
                },
                {
                    "problem": "Write a function to find the sum of all elements in a list.",
                    "test_cases": []
                },
                {
                    "problem": "Create a function that takes a list of integers and returns a new list with only the even numbers.",
                    "test_cases": []
                },
                {
                    "problem": "Write a function to count the number of vowels in a string.",
                    "test_cases": []
                },
                {
                    "problem": "Implement a function to find the maximum value in a list without using the built-in max() function.",
                    "test_cases": []
                }
            ]
        }
        
        # Return requested number of problems
        available_problems = sample_problems.get(benchmark_type, [])
        return available_problems[:min(count, len(available_problems))]


def run_demo(api_url: str = "http://localhost:8000"):
    """
    Run the demo UI.
    
    Args:
        api_url: URL of the inference API server
    """
    demo = SimpleFoundationDemo(api_url=api_url)
    demo.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the SimpleFoundation demo UI")
    parser.add_argument("--api_url", type=str, default="http://localhost:8000", help="URL of the inference API server")
    
    args = parser.parse_args()
    
    run_demo(api_url=args.api_url)
