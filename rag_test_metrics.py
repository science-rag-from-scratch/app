"""
RAG Testing Module using RAGAS Framework

This module implements comprehensive RAG (Retrieval-Augmented Generation) system testing
using the RAGAS evaluation framework. It provides functionality to configure evaluation metrics,
execute tests, and generate performance reports.

Key Features:
- Configurable evaluation metrics (faithfulness, answer relevancy, context recall, context precision)
- Support for custom test data or sample data generation
- Detailed result reporting with visualization options
- Robust error handling and logging
- Type safety with comprehensive type hints

Requirements:
- ragas>=0.1.0
- pandas>=1.5.0
- numpy>=1.24.0
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    AnswerSimilarity,
    AspectCritic
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGTester:
    """
    A comprehensive testing class for RAG systems using RAGAS metrics.
    
    This class provides methods to:
    - Configure evaluation metrics
    - Generate or load test datasets
    - Execute evaluation pipelines
    - Generate detailed reports
    """
    
    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        custom_metrics: Optional[Dict[str, callable]] = None,
        sample_size: int = 10,
        random_seed: int = 42
    ):
        """
        Initialize the RAGTester with configuration parameters.
        
        Args:
            metrics: List of metric names to use. Defaults to standard RAGAS metrics.
            custom_metrics: Dictionary of custom metrics {name: metric_function}
            sample_size: Number of samples to generate if using sample data
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.sample_size = sample_size
        self.metrics_config = self._configure_metrics(metrics, custom_metrics)
        self.test_data = None
        self.results = None
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
    def _configure_metrics(
        self, 
        metrics: Optional[List[str]], 
        custom_metrics: Optional[Dict[str, callable]]
    ) -> Dict[str, callable]:
        """
        Configure evaluation metrics with default and custom options.
        
        Args:
            metrics: List of metric names to include
            custom_metrics: Dictionary of custom metrics to add
            
        Returns:
            Dictionary of configured metrics {name: metric_function}
        """
        # Default metrics
        default_metrics = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_recall": context_recall,
            "context_precision": context_precision
        }
        
        # Use custom metrics if provided
        if custom_metrics:
            default_metrics.update(custom_metrics)
            
        # Filter metrics based on user input
        if metrics is None:
            logger.info("Using default metrics")
            return default_metrics
            
        # Validate requested metrics
        invalid_metrics = [m for m in metrics if m not in default_metrics]
        if invalid_metrics:
            raise ValueError(f"Invalid metrics requested: {invalid_metrics}")
            
        return {name: default_metrics[name] for name in metrics}
    
    def generate_sample_data(self) -> pd.DataFrame:
        """
        Generate sample test data for demonstration purposes.
        
        Returns:
            DataFrame with columns: question, ground_truth, contexts, answer
        """
        logger.info(f"Generating {self.sample_size} sample test cases")
        
        # Sample questions
        questions = [
            "What is the capital of France?",
            "Who wrote 'Romeo and Juliet'?",
            "Explain photosynthesis",
            "What is machine learning?",
            "Describe the water cycle"
        ] * (self.sample_size // 5 + 1)
        
        # Sample ground truths
        ground_truths = [
            "The capital of France is Paris.",
            "William Shakespeare wrote 'Romeo and Juliet'.",
            "Photosynthesis is the process by which plants convert light energy into chemical energy.",
            "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.",
            "The water cycle is the continuous movement of water on, above, and below the surface of the Earth."
        ] * (self.sample_size // 5 + 1)
        
        # Sample contexts (retrieved documents)
        contexts = [
            [["Paris is the capital city of France."]],
            [["Shakespeare was an English playwright and poet."]],
            [["Plants use sunlight to convert carbon dioxide and water into glucose and oxygen."]],
            [["Machine learning algorithms build models based on sample data to make predictions."]],
            [["The water cycle involves evaporation, condensation, precipitation, and collection."]]
        ] * (self.sample_size // 5 + 1)
        
        # Sample answers
        answers = [
            "Paris is the capital of France.",
            "The famous play 'Romeo and Juliet' was written by William Shakespeare.",
            "Photosynthesis is how plants make food using sunlight.",
            "Machine learning is about teaching computers to learn from data.",
            "The water cycle describes how water moves through the environment."
        ] * (self.sample_size // 5 + 1)
        
        # Create DataFrame
        sample_df = pd.DataFrame({
            "question": questions[:self.sample_size],
            "ground_truth": ground_truths[:self.sample_size],
            "contexts": contexts[:self.sample_size],
            "answer": answers[:self.sample_size]
        })
        
        self.test_data = sample_df
        logger.info("Sample data generated successfully")
        return sample_df
    
    def load_test_data(self, data: pd.DataFrame) -> None:
        """
        Load custom test data for evaluation.
        
        Args:
            data: DataFrame with required columns:
                  - 'question': User questions
                  - 'ground_truth': Expected answers
                  - 'contexts': Retrieved context documents
                  - 'answer': RAG system answers
                  
        Raises:
            ValueError: If required columns are missing
        """
        required_columns = {"question", "ground_truth", "contexts", "answer"}
        missing_columns = required_columns - set(data.columns)
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        self.test_data = data
        logger.info(f"Loaded {len(data)} test cases")
    
    def evaluate_rag_system(self) -> pd.DataFrame:
        """
        Execute the RAG evaluation pipeline using configured metrics.
        
        Returns:
            DataFrame containing evaluation results with metrics as columns
            
        Raises:
            RuntimeError: If test data hasn't been loaded or generated
        """
        if self.test_data is None:
            raise RuntimeError("Test data not loaded. Call generate_sample_data() or load_test_data() first")
            
        logger.info("Starting RAG evaluation")
        
        try:
            # Execute evaluation
            result_df = evaluate(
                self.test_data,
                metrics=list(self.metrics_config.values()),
                raise_exceptions=True
            )
            
            self.results = result_df
            logger.info("Evaluation completed successfully")
            return result_df
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise RuntimeError(f"Evaluation failed: {str(e)}")
    
    def generate_report(
        self, 
        output_format: str = "console",
        output_path: Optional[str] = None,
        include_details: bool = True
    ) -> Union[str, pd.DataFrame]:
        """
        Generate evaluation report in specified format.
        
        Args:
            output_format: Output format ('console', 'csv', 'json', 'dataframe')
            output_path: File path to save results (if applicable)
            include_details: Whether to include detailed test case results
            
        Returns:
            Report data as string, DataFrame, or None (if saved to file)
            
        Raises:
            ValueError: If invalid output format specified
            RuntimeError: If evaluation hasn't been run
        """
        if self.results is None:
            raise RuntimeError("Evaluation not run. Call evaluate_rag_system() first")
            
        if output_format not in ["console", "csv", "json", "dataframe"]:
            raise ValueError("Invalid output format. Use 'console', 'csv', 'json', or 'dataframe'")
            
        # Prepare report data
        report_data = self.results.copy()
        
        # Add overall summary
        summary = pd.DataFrame({
            "metric": list(self.metrics_config.keys()),
            "score": [self.results[m].mean() for m in self.metrics_config.keys()],
            "max_score": [1.0] * len(self.metrics_config)
        })
        
        if output_format == "console":
            # Console report
            report_str = "\n=== RAG Evaluation Summary ===\n"
            report_str += summary.to_string(index=False)
            
            if include_details:
                report_str += "\n\n=== Detailed Results ===\n"
                report_str += report_data.to_string()
                
            print(report_str)
            return report_str
            
        elif output_format == "csv":
            # Save to CSV
            if output_path is None:
                output_path = "rag_evaluation_results.csv"
            report_data.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
            
        elif output_format == "json":
            # Save to JSON
            if output_path is None:
                output_path = "rag_evaluation_results.json"
            report_data.to_json(output_path, orient="records")
            logger.info(f"Results saved to {output_path}")
            
        elif output_format == "dataframe":
            return summary if not include_details else report_data
    
    def add_custom_metric(
        self, 
        name: str, 
        metric: callable, 
        description: Optional[str] = None
    ) -> None:
        """
        Add a custom evaluation metric to the tester.
        
        Args:
            name: Unique name for the metric
            metric: Callable metric function following RAGAS interface
            description: Optional description of the metric
        """
        if not callable(metric):
            raise TypeError("Metric must be a callable function")
            
        if name in self.metrics_config:
            logger.warning(f"Replacing existing metric '{name}'")
            
        self.metrics_config[name] = metric
        logger.info(f"Added custom metric '{name}'")


def main():
    """
    Example usage of the RAGTester class.
    
    This function demonstrates:
    1. Initialization with default metrics
    2. Sample data generation
    3. Evaluation execution
    4. Report generation
    """
    # Initialize tester with default metrics
    tester = RAGTester(sample_size=5)
    
    # Generate sample data
    sample_data = tester.generate_sample_data()
    print(f"Generated {len(sample_data)} test cases")
    
    # Run evaluation
    results = tester.evaluate_rag_system()
    print("\nEvaluation Results:")
    print(results)
    
    # Generate console report
    report = tester.generate_report(output_format="console")
    
    # Save detailed results to CSV
    tester.generate_report(output_format="csv", output_path="rag_results.csv")


if __name__ == "__main__":
    main()