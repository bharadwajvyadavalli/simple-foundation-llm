"""
Dataset Collection Module for SimpleFoundation

This module handles the downloading and organizing of datasets used for training.
It supports math and coding datasets from various sources.
"""

import os
import logging
import requests
import zipfile
import json
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetCollector:
    """Base class for dataset collection."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the dataset collector.
        
        Args:
            data_dir: Directory to store downloaded datasets
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_file(self, url: str, filename: str) -> str:
        """
        Download a file from a URL.
        
        Args:
            url: URL to download from
            filename: Name to save the file as
            
        Returns:
            Path to the downloaded file
        """
        filepath = os.path.join(self.data_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(filepath):
            logger.info(f"File {filepath} already exists, skipping download")
            return filepath
        
        logger.info(f"Downloading {url} to {filepath}")
        
        # Stream download with progress bar
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        return filepath
    
    def extract_zip(self, zip_path: str, extract_dir: Optional[str] = None) -> str:
        """
        Extract a zip file.
        
        Args:
            zip_path: Path to the zip file
            extract_dir: Directory to extract to. If None, uses the zip filename
            
        Returns:
            Path to the extracted directory
        """
        if extract_dir is None:
            extract_dir = os.path.join(self.data_dir, os.path.splitext(os.path.basename(zip_path))[0])
        
        os.makedirs(extract_dir, exist_ok=True)
        
        logger.info(f"Extracting {zip_path} to {extract_dir}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        return extract_dir
    
    def download_and_prepare(self):
        """
        Download and prepare datasets. To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement download_and_prepare")


class MathDatasetCollector(DatasetCollector):
    """Collector for math datasets."""
    
    def __init__(self, data_dir: str = "data/raw/math"):
        """Initialize with math-specific data directory."""
        super().__init__(data_dir)
        
    def download_and_prepare(self, datasets: List[str] = ["math", "aime"]):
        """
        Download and prepare math datasets.
        
        Args:
            datasets: List of datasets to download ["math", "aime"]
        """
        results = {}
        
        if "math" in datasets:
            results["math"] = self._download_math_dataset()
        
        if "aime" in datasets:
            results["aime"] = self._download_aime_dataset()
        
        return results
    
    def _download_math_dataset(self) -> Dict:
        """
        Download the MATH dataset from Hendrycks et al.
        
        Returns:
            Dictionary with dataset information
        """
        url = "https://github.com/hendrycks/math/archive/refs/heads/main.zip"
        zip_path = self.download_file(url, "math-dataset.zip")
        extract_dir = self.extract_zip(zip_path)
        
        # Process the dataset into a format usable by our system
        processed_data = []
        
        # Walk through the dataset directories
        for difficulty in ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "precalculus", "prealgebra"]:
            difficulty_dir = os.path.join(extract_dir, "math-main", difficulty)
            
            if not os.path.exists(difficulty_dir):
                logger.warning(f"Directory {difficulty_dir} does not exist, skipping")
                continue
                
            # Process problem files
            for level in os.listdir(difficulty_dir):
                level_dir = os.path.join(difficulty_dir, level)
                
                if not os.path.isdir(level_dir):
                    continue
                
                for filename in os.listdir(level_dir):
                    if not filename.endswith(".json"):
                        continue
                    
                    with open(os.path.join(level_dir, filename), 'r') as f:
                        problem_data = json.load(f)
                        
                        processed_data.append({
                            "id": f"{difficulty}_{level}_{filename}",
                            "problem": problem_data.get("problem", ""),
                            "level": level,
                            "category": difficulty,
                            "solution": problem_data.get("solution", ""),
                            "answer": problem_data.get("answer", "")
                        })
        
        # Save processed data
        output_path = os.path.join(self.data_dir, "math_processed.json")
        with open(output_path, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        logger.info(f"Processed {len(processed_data)} MATH problems")
        
        return {
            "count": len(processed_data),
            "path": output_path,
            "categories": list(set(item["category"] for item in processed_data)),
            "levels": list(set(item["level"] for item in processed_data))
        }
    
    def _download_aime_dataset(self) -> Dict:
        """
        Download and process AIME (American Invitational Mathematics Examination) problems.
        
        Returns:
            Dictionary with dataset information
        """
        # For simplicity, we'll create a small sample of AIME problems
        # In a real implementation, you would scrape or download from a proper source
        
        aime_problems = [
            {
                "id": "aime_2020_1",
                "problem": "Find the number of ordered pairs (m,n) of positive integers such that m + n + mn = 20.",
                "answer": "4",
                "year": "2020",
                "number": 1
            },
            {
                "id": "aime_2020_2",
                "problem": "A right triangle has leg lengths a and b and hypotenuse length c, where a, b, and c are positive integers. Given that c=21 and ab=84, find a+b.",
                "answer": "20",
                "year": "2020",
                "number": 2
            },
            {
                "id": "aime_2021_1",
                "problem": "Find the sum of all positive integers n such that n^2+6n+1 is divisible by n+3.",
                "answer": "21",
                "year": "2021",
                "number": 1
            }
        ]
        
        # Save as JSON
        output_path = os.path.join(self.data_dir, "aime_problems.json")
        with open(output_path, 'w') as f:
            json.dump(aime_problems, f, indent=2)
        
        logger.info(f"Created {len(aime_problems)} AIME problems")
        
        return {
            "count": len(aime_problems),
            "path": output_path,
            "years": list(set(item["year"] for item in aime_problems))
        }


class CodingDatasetCollector(DatasetCollector):
    """Collector for coding challenge datasets."""
    
    def __init__(self, data_dir: str = "data/raw/coding"):
        """Initialize with coding-specific data directory."""
        super().__init__(data_dir)
        
    def download_and_prepare(self, datasets: List[str] = ["apps"]):
        """
        Download and prepare coding datasets.
        
        Args:
            datasets: List of datasets to download ["apps", "taco"]
        """
        results = {}
        
        if "apps" in datasets:
            results["apps"] = self._download_apps_dataset()
        
        return results
    
    def _download_apps_dataset(self) -> Dict:
        """
        Download the APPS (Automated Programming Progress Standard) dataset.
        
        Returns:
            Dictionary with dataset information
        """
        url = "https://github.com/hendrycks/apps/archive/refs/heads/main.zip"
        zip_path = self.download_file(url, "apps-dataset.zip")
        extract_dir = self.extract_zip(zip_path)
        
        # Process the dataset into a format usable by our system
        processed_data = []
        
        # Process different difficulty levels
        for difficulty in ["introductory", "interview", "competition"]:
            difficulty_dir = os.path.join(extract_dir, "apps-main", "train", difficulty)
            
            if not os.path.exists(difficulty_dir):
                logger.warning(f"Directory {difficulty_dir} does not exist, skipping")
                continue
            
            # Each subdirectory is a problem
            for problem_id in os.listdir(difficulty_dir):
                problem_dir = os.path.join(difficulty_dir, problem_id)
                
                if not os.path.isdir(problem_dir):
                    continue
                
                # Read problem statement
                problem_path = os.path.join(problem_dir, "question.txt")
                if not os.path.exists(problem_path):
                    logger.warning(f"Problem file {problem_path} does not exist, skipping")
                    continue
                
                with open(problem_path, 'r', encoding='utf-8') as f:
                    problem_text = f.read()
                
                # Read solutions if available
                solutions = []
                solutions_dir = os.path.join(problem_dir, "solutions")
                if os.path.exists(solutions_dir):
                    for solution_file in os.listdir(solutions_dir):
                        if not solution_file.endswith(".py"):
                            continue
                        
                        solution_path = os.path.join(solutions_dir, solution_file)
                        with open(solution_path, 'r', encoding='utf-8') as f:
                            solutions.append(f.read())
                
                # Read test cases
                test_path = os.path.join(problem_dir, "input_output.json")
                test_cases = []
                if os.path.exists(test_path):
                    with open(test_path, 'r', encoding='utf-8') as f:
                        test_data = json.load(f)
                        for i, (test_in, test_out) in enumerate(zip(test_data.get("inputs", []), test_data.get("outputs", []))):
                            test_cases.append({
                                "input": test_in,
                                "expected_output": test_out
                            })
                
                processed_data.append({
                    "id": f"{difficulty}_{problem_id}",
                    "problem": problem_text,
                    "difficulty": difficulty,
                    "solutions": solutions,
                    "test_cases": test_cases
                })
        
        # Save processed data
        output_path = os.path.join(self.data_dir, "apps_processed.json")
        with open(output_path, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        logger.info(f"Processed {len(processed_data)} APPS problems")
        
        return {
            "count": len(processed_data),
            "path": output_path,
            "difficulties": list(set(item["difficulty"] for item in processed_data))
        }


def download_all_datasets(output_dir: str = "data/raw"):
    """
    Download all supported datasets.
    
    Args:
        output_dir: Base directory to store datasets
    """
    # Download math datasets
    math_collector = MathDatasetCollector(os.path.join(output_dir, "math"))
    math_data = math_collector.download_and_prepare(["math", "aime"])
    
    # Download coding datasets
    coding_collector = CodingDatasetCollector(os.path.join(output_dir, "coding"))
    coding_data = coding_collector.download_and_prepare(["apps"])
    
    # Return summary of all downloaded data
    return {
        "math": math_data,
        "coding": coding_data
    }


if __name__ == "__main__":
    # Example usage
    download_all_datasets()
