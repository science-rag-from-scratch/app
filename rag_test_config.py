"""
RAGAS Test Configuration Module

This module provides configuration for RAG (Retrieval-Augmented Generation) testing
using the RAGAS framework. It includes settings for evaluation metrics, test data,
pipeline execution, and result reporting.

Author: AI Assistant
"""

import os
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_similarity,
    answer_correctness,
)


class RagTestConfig:
    """
    Configuration class for RAGAS testing framework.
    
    Attributes:
        evaluation_metrics (List[str]): List of evaluation metrics to use
        test_data_source (str): Source of test data ('file', 'generate', or 'hybrid')
        test_data_path (Optional[str]): Path to test data file
        data_generation_params (Dict[str, Any]): Parameters for synthetic data generation
        pipeline_params (Dict[str, Any]): Parameters for test execution pipeline
        reporting_params (Dict[str, Any]): Parameters for result reporting
    """
    
    def __init__(
        self,
        evaluation_metrics: List[str] = None,
        test_data_source: str = "generate",
        test_data_path: Optional[str] = None,
        data_generation_params: Dict[str, Any] = None,
        pipeline_params: Dict[str, Any] = None,
        reporting_params: Dict[str, Any] = None
    ):
        """
        Initialize RAGAS test configuration.
        
        Args:
            evaluation_metrics: List of metrics to use in evaluation
            test_data_source: Source of test data ('file', 'generate', or 'hybrid')
            test_data_path: Path to test data file (if using file source)
            data_generation_params: Parameters for synthetic data generation
            pipeline_params: Parameters for test execution pipeline
            reporting_params: Parameters for result reporting
        """
        # Set default evaluation metrics
        self.evaluation_metrics = evaluation_metrics or [
            "answer_relevancy",
            "faithfulness",
            "context_recall",
            "context_precision",
            "answer_similarity",
            "answer_correctness"
        ]
        
        # Validate test data source
        valid_sources = ["file", "generate", "hybrid"]
        if test_data_source not in valid_sources:
            raise ValueError(f"Invalid test_data_source. Must be one of: {valid_sources}")
        self.test_data_source = test_data_source
        
        # Set test data path
        self.test_data_path = test_data_path
        
        # Set default data generation parameters
        self.data_generation_params = data_generation_params or {
            "num_samples": 100,
            "question_types": ["factual", "comparative", "explanatory"],
            "difficulty_distribution": [0.5, 0.3, 0.2],  # Easy, medium, hard
            "seed": 42
        }
        
        # Set default pipeline parameters
        self.pipeline_params = pipeline_params or {
            "batch_size": 10,
            "max_workers": 4,
            "timeout": 300,  # seconds
            "retry_attempts": 3,
            "retry_delay": 5  # seconds
        }
        
        # Set default reporting parameters
        self.reporting_params = reporting_params or {
            "output_dir": "ragas_results",
            "formats": ["csv", "json", "html"],
            "include_plots": True,
            "plot_config": {
                "figsize": (10, 6),
                "style": "seaborn",
                "palette": "viridis"
            }
        }
    
    def get_metric_objects(self) -> List[Any]:
        """
        Get metric objects for evaluation.
        
        Returns:
            List of RAGAS metric objects
        """
        metric_map = {
            "answer_relevancy": answer_relevancy,
            "faithfulness": faithfulness,
            "context_recall": context_recall,
            "context_precision": context_precision,
            "answer_similarity": answer_similarity,
            "answer_correctness": answer_correctness
        }
        
        selected_metrics = []
        for metric_name in self.evaluation_metrics:
            if metric_name in metric_map:
                selected_metrics.append(metric_map[metric_name])
            else:
                print(f"Warning: Metric '{metric_name}' not recognized, skipping")
        
        return selected_metrics
    
    def validate_config(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Check evaluation metrics
            if not isinstance(self.evaluation_metrics, list) or not self.evaluation_metrics:
                raise ValueError("evaluation_metrics must be a non-empty list")
            
            # Check test data source
            if self.test_data_source not in ["file", "generate", "hybrid"]:
                raise ValueError("Invalid test_data_source")
            
            # Check test data path if using file source
            if self.test_data_source in ["file", "hybrid"] and not self.test_data_path:
                raise ValueError("test_data_path required when using file or hybrid data source")
            
            # Check data generation parameters
            if not isinstance(self.data_generation_params, dict):
                raise ValueError("data_generation_params must be a dictionary")
            
            # Check pipeline parameters
            if not isinstance(self.pipeline_params, dict):
                raise ValueError("pipeline_params must be a dictionary")
            
            # Check reporting parameters
            if not isinstance(self.reporting_params, dict):
                raise ValueError("reporting_params must be a dictionary")
            
            # Check output directory
            output_dir = self.reporting_params.get("output_dir")
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            return True
        
        except Exception as e:
            print(f"Configuration validation error: {str(e)}")
            return False
    
    def save_config(self, filepath: str) -> None:
        """
        Save configuration to a file.
        
        Args:
            filepath: Path to save configuration file
        """
        try:
            config_dict = {
                "evaluation_metrics": self.evaluation_metrics,
                "test_data_source": self.test_data_source,
                "test_data_path": self.test_data_path,
                "data_generation_params": self.data_generation_params,
                "pipeline_params": self.pipeline_params,
                "reporting_params": self.reporting_params
            }
            
            df = pd.DataFrame([config_dict])
            df.to_json(filepath, orient="records", indent=2)
            print(f"Configuration saved to {filepath}")
        
        except Exception as e:
            print(f"Error saving configuration: {str(e)}")
    
    @classmethod
    def load_config(cls, filepath: str) -> 'RagTestConfig':
        """
        Load configuration from a file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            RagTestConfig object
        """
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Configuration file not found: {filepath}")
            
            df = pd.read_json(filepath, orient="records")
            config_dict = df.iloc[0].to_dict()
            
            return cls(
                evaluation_metrics=config_dict.get("evaluation_metrics"),
                test_data_source=config_dict.get("test_data_source"),
                test_data_path=config_dict.get("test_data_path"),
                data_generation_params=config_dict.get("data_generation_params"),
                pipeline_params=config_dict.get("pipeline_params"),
                reporting_params=config_dict.get("reporting_params")
            )
        
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Create default configuration
    config = RagTestConfig()
    
    # Validate configuration
    if config.validate_config():
        print("Configuration is valid")
    else:
        print("Configuration validation failed")
    
    # Save configuration
    config.save_config("ragas_test_config.json")
    
    # Load configuration from file
    loaded_config = RagTestConfig.load_config("ragas_test_config.json")
    print("Loaded configuration:", loaded_config.evaluation_metrics)