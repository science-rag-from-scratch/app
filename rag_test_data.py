"""
RAG Test Data Module

This module provides functionality for generating and managing test datasets for RAG system evaluation
using the RAGAS framework. It includes utilities for creating synthetic test data and loading
custom datasets from external files.

Classes:
    RAGTestData: Manages test datasets for RAG evaluation with validation and generation capabilities.

Methods:
    generate_synthetic(size: int, num_contexts: int) -> pd.DataFrame:
        Generates a synthetic test dataset with questions, contexts, answers, and ground truths.
    
    load_from_csv(file_path: str) -> pd.DataFrame:
        Loads test data from a CSV file with required columns.
    
    load_from_json(file_path: str) -> pd.DataFrame:
        Loads test data from a JSON file with required columns.
    
    validate_dataset(data: pd.DataFrame) -> bool:
        Validates that a dataset contains all required columns for RAGAS evaluation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGTestData:
    """
    A class for managing test datasets for RAG system evaluation.
    
    This class provides methods to generate synthetic test data and load custom datasets
    from various file formats. All datasets are validated to ensure they meet the
    requirements for RAGAS evaluation.
    """
    
    REQUIRED_COLUMNS = ["question", "answer", "contexts", "ground_truths"]
    
    def __init__(self):
        """Initialize the RAGTestData manager."""
        self.dataset = None
        
    def generate_synthetic(
        self, 
        size: int = 100, 
        num_contexts: int = 3,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate a synthetic test dataset for RAG evaluation.
        
        Args:
            size: Number of test samples to generate (default: 100)
            num_contexts: Number of contexts per question (default: 3)
            seed: Random seed for reproducibility (optional)
            
        Returns:
            pd.DataFrame: Synthetic test dataset with required columns
            
        Raises:
            ValueError: If size or num_contexts are invalid
        """
        if size <= 0:
            raise ValueError("Size must be a positive integer")
        if num_contexts <= 0:
            raise ValueError("Number of contexts must be positive")
            
        if seed is not None:
            np.random.seed(seed)
            
        # Generate synthetic questions
        questions = [
            f"What is {np.random.choice(['the capital', 'the population', 'the area'])} of {np.random.choice(['France', 'Germany', 'Japan', 'Canada', 'Australia'])}?",
            f"Explain {np.random.choice(['quantum computing', 'blockchain technology', 'machine learning'])} in simple terms",
            f"When was {np.random.choice(['the telephone invented', 'the internet developed', 'the first computer built'])}?",
            f"Compare {np.random.choice(['Python and Java', 'React and Angular', 'SQL and NoSQL'])} based on their use cases"
            for _ in range(size)
        ]
        
        # Generate synthetic answers
        answers = [
            f"Based on the retrieved information, the answer is {np.random.choice(['Paris', 'Berlin', 'Tokyo', 'Ottawa', 'Canberra'])} with a population of approximately {np.random.randint(1, 100)} million.",
            f"The retrieved documents explain that {np.random.choice(['quantum computing', 'blockchain', 'AI'])} involves {np.random.choice(['advanced computational methods', 'distributed ledger technology', 'neural networks'])}.",
            f"According to the sources, this was invented in {np.random.randint(1800, 2020)}.",
            f"Based on the provided contexts, {np.random.choice(['Python is known for its simplicity', 'React has a component-based architecture'])} while {np.random.choice(['Java is more verbose', 'Angular offers full framework capabilities'])}."
            for _ in range(size)
        ]
        
        # Generate synthetic contexts
        contexts = [
            [
                f"Context {i+1}: {np.random.choice(['Document A states', 'Source B mentions', 'Article C explains'])} that {np.random.choice(['this is a key fact', 'this information is crucial', 'this detail is important'])}.",
                f"Context {i+2}: {np.random.choice(['Research D shows', 'Study E indicates', 'Analysis F reveals'])} that {np.random.choice(['this supports the answer', 'this provides relevant data', 'this corroborates the information'])}.",
                f"Context {i+3}: {np.random.choice(['Reference G notes', 'Report H outlines', 'Whitepaper I details'])} that {np.random.choice(['this is a significant point', 'this is a critical detail', 'this is a major finding'])}."
            ]
            for i in range(size)
        ]
        
        # Generate ground truths
        ground_truths = [
            [
                np.random.choice([
                    "France's capital is Paris and its population is about 67 million",
                    "The capital of Germany is Berlin with a population of around 83 million",
                    "Tokyo is the capital of Japan with a population of approximately 126 million"
                ]),
                np.random.choice([
                    "Quantum computing uses quantum bits (qubits) to perform complex calculations",
                    "Blockchain technology provides a decentralized ledger system",
                    "Machine learning enables systems to learn patterns from data without explicit programming"
                ])
            ]
            for _ in range(size)
        ]
        
        # Create DataFrame
        data = pd.DataFrame({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truths": ground_truths
        })
        
        # Validate and store
        if not self.validate_dataset(data):
            raise ValueError("Generated synthetic data failed validation")
            
        self.dataset = data
        logger.info(f"Successfully generated synthetic dataset with {size} samples")
        return data
    
    def load_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load test data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded test dataset
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file doesn't contain required columns
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Test data file not found: {file_path}")
            
        try:
            data = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")
            
        if not self.validate_dataset(data):
            raise ValueError("CSV file doesn't contain all required columns")
            
        self.dataset = data
        logger.info(f"Successfully loaded dataset from {file_path} with {len(data)} samples")
        return data
    
    def load_from_json(self, file_path: str) -> pd.DataFrame:
        """
        Load test data from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            pd.DataFrame: Loaded test dataset
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file doesn't contain required columns
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Test data file not found: {file_path}")
            
        try:
            data = pd.read_json(file_path)
        except Exception as e:
            raise ValueError(f"Error reading JSON file: {str(e)}")
            
        if not self.validate_dataset(data):
            raise ValueError("JSON file doesn't contain all required columns")
            
        self.dataset = data
        logger.info(f"Successfully loaded dataset from {file_path} with {len(data)} samples")
        return data
    
    def validate_dataset(self, data: pd.DataFrame) -> bool:
        """
        Validate that a dataset contains all required columns for RAGAS evaluation.
        
        Args:
            data: The dataset to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not isinstance(data, pd.DataFrame):
            logger.error("Data must be a pandas DataFrame")
            return False
            
        missing_columns = set(self.REQUIRED_COLUMNS) - set(data.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
            
        # Check for empty values in critical columns
        for col in ["question", "answer"]:
            if data[col].isnull().any():
                logger.error(f"Column '{col}' contains null values")
                return False
                
        # Check data types
        if not isinstance(data["contexts"].iloc[0], list):
            logger.error("'contexts' column must contain lists")
            return False
            
        if not isinstance(data["ground_truths"].iloc[0], list):
            logger.error("'ground_truths' column must contain lists")
            return False
            
        return True
    
    def get_dataset(self) -> Optional[pd.DataFrame]:
        """
        Get the currently loaded dataset.
        
        Returns:
            pd.DataFrame or None: The loaded dataset or None if no dataset is loaded
        """
        return self.dataset
    
    def save_dataset(self, file_path: str, format: str = "csv") -> None:
        """
        Save the current dataset to a file.
        
        Args:
            file_path: Path to save the dataset
            format: File format ('csv' or 'json')
            
        Raises:
            ValueError: If no dataset is loaded or format is invalid
        """
        if self.dataset is None:
            raise ValueError("No dataset is currently loaded")
            
        if format not in ["csv", "json"]:
            raise ValueError("Format must be 'csv' or 'json'")
            
        try:
            if format == "csv":
                self.dataset.to_csv(file_path, index=False)
            else:
                self.dataset.to_json(file_path, orient="records", indent=2)
            logger.info(f"Successfully saved dataset to {file_path}")
        except Exception as e:
            logger.error(f"Error saving dataset: {str(e)}")
            raise