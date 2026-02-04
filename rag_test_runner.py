"""
RAG Test Runner Implementation

This module implements a comprehensive testing framework for Retrieval-Augmented Generation (RAG) systems
using the RAGAS evaluation framework. It provides functionality to:
- Configure evaluation metrics
- Load or generate test datasets
- Execute RAG system evaluation
- Generate detailed performance reports

Author: AI Assistant
"""

import os
import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
    AspectCritic
)
from ragas.metrics.critics import harmfulness

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RagTestConfig:
    """Configuration class for RAG testing parameters"""
    metrics: List[str] = None
    sample_size: int = 20
    threshold: float = 0.7
    report_path: str = "rag_test_report"
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                "faithfulness", 
                "answer_relevancy", 
                "context_precision",
                "context_recall"
            ]

class RagTestRunner:
    """
    RAG testing framework using RAGAS evaluation metrics.
    
    This class provides a comprehensive pipeline for evaluating RAG system performance
    across multiple dimensions including faithfulness, relevancy, and correctness.
    """
    
    def __init__(self, config: Optional[RagTestConfig] = None):
        """
        Initialize the RAG test runner.
        
        Args:
            config: Configuration object with testing parameters
        """
        self.config = config or RagTestConfig()
        self._validate_metrics()
        self.metrics = self._initialize_metrics()
        self.results = None
        logger.info("RAG Test Runner initialized with config: %s", self.config.__dict__)
    
    def _validate_metrics(self):
        """Validate that selected metrics are supported by RAGAS"""
        supported_metrics = {
            "faithfulness", "answer_relevancy", "context_precision", 
            "context_recall", "answer_correctness", "harmfulness"
        }
        invalid_metrics = set(self.config.metrics) - supported_metrics
        if invalid_metrics:
            raise ValueError(f"Unsupported metrics: {invalid_metrics}. "
                           f"Supported metrics are: {supported_metrics}")
    
    def _initialize_metrics(self) -> List[Any]:
        """Initialize evaluation metrics based on configuration"""
        metric_map = {
            "faithfulness": Faithfulness(),
            "answer_relevancy": AnswerRelevancy(),
            "context_precision": ContextPrecision(),
            "context_recall": ContextRecall(),
            "answer_correctness": AnswerCorrectness(),
            "harmfulness": harmfulness
        }
        return [metric_map[metric] for metric in self.config.metrics]
    
    def load_test_data(self, data_path: str) -> pd.DataFrame:
        """
        Load test data from a CSV file.
        
        Expected CSV columns:
        - question: The user query
        - answer: Generated answer from the RAG system
        - contexts: Retrieved contexts (JSON array)
        - ground_truth: Ground truth answer (optional)
        
        Args:
            data_path: Path to the CSV file containing test data
            
        Returns:
            DataFrame with test data
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If required columns are missing
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Test data file not found: {data_path}")
        
        try:
            df = pd.read_csv(data_path)
            required_columns = {"question", "answer", "contexts"}
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            logger.info(f"Loaded test data with {len(df)} samples from {data_path}")
            return df.head(self.config.sample_size)
            
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            raise
    
    def generate_test_data(self, num_samples: int = 20) -> pd.DataFrame:
        """
        Generate synthetic test data for demonstration purposes.
        
        Args:
            num_samples: Number of test samples to generate
            
        Returns:
            DataFrame with synthetic test data
        """
        logger.info(f"Generating {num_samples} synthetic test samples")
        np.random.seed(42)
        
        questions = [
            "What is the capital of France?",
            "Explain the theory of relativity.",
            "Who wrote 'Romeo and Juliet'?",
            "What causes climate change?",
            "How do photosynthesis works?"
        ] * (num_samples // 5 + 1)
        
        answers = [
            "The capital of France is Paris.",
            "Einstein's theory of relativity revolutionized our understanding of space, time, and gravity.",
            "William Shakespeare wrote 'Romeo and Juliet'.",
            "Climate change is primarily caused by human activities like burning fossil fuels.",
            "Photosynthesis converts light energy into chemical energy in plants."
        ] * (num_samples // 5 + 1)
        
        contexts = [
            ["Paris is the capital city of France."],
            ["The theory of relativity encompasses two interrelated theories by Albert Einstein."],
            ["Shakespeare was an English playwright and poet."],
            ["Greenhouse gases trap heat in Earth's atmosphere."],
            ["Plants use sunlight to create food during photosynthesis."]
        ] * (num_samples // 5 + 1)
        
        ground_truths = [
            "Paris",
            "A theory about space, time, and gravity",
            "William Shakespeare",
            "Human activities and greenhouse gases",
            "Conversion of light energy to chemical energy"
        ] * (num_samples // 5 + 1)
        
        return pd.DataFrame({
            "question": questions[:num_samples],
            "answer": answers[:num_samples],
            "contexts": contexts[:num_samples],
            "ground_truth": ground_truths[:num_samples]
        })
    
    def run_tests(self, test_data: Union[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Execute RAG system evaluation using configured metrics.
        
        Args:
            test_data: Either a path to CSV file or DataFrame with test data
            
        Returns:
            Dictionary containing evaluation results and summary statistics
        """
        try:
            if isinstance(test_data, str):
                df = self.load_test_data(test_data)
            else:
                df = test_data.head(self.config.sample_size)
            
            logger.info("Starting RAG evaluation with %d samples", len(df))
            logger.info(f"Evaluating metrics: {self.config.metrics}")
            
            # Run RAGAS evaluation
            self.results = evaluate(
                df=df,
                metrics=self.metrics
            )
            
            # Convert results to DataFrame for easier processing
            results_df = self.results.to_pandas()
            
            # Calculate summary statistics
            summary = {
                "total_samples": len(df),
                "metrics_evaluated": self.config.metrics,
                "mean_scores": results_df.mean(numeric_only=True).to_dict(),
                "pass_rate": (results_df.mean(numeric_only=True) >= self.config.threshold).mean(),
                "threshold": self.config.threshold
            }
            
            logger.info("Evaluation completed successfully")
            return {
                "results": results_df,
                "summary": summary,
                "config": self.config.__dict__
            }
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
    
    def generate_report(self, results: Dict[str, Any], output_format: str = "csv") -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            results: Results dictionary from run_tests()
            output_format: Output format ('csv', 'json', or 'html')
            
        Returns:
            Path to the generated report file
        """
        try:
            results_df = results["results"]
            summary = results["summary"]
            
            # Create output directory if needed
            os.makedirs(self.config.report_path, exist_ok=True)
            
            if output_format == "csv":
                report_path = os.path.join(self.config.report_path, "evaluation_results.csv")
                results_df.to_csv(report_path, index=False)
                
                # Save summary to a separate file
                summary_path = os.path.join(self.config.report_path, "summary.txt")
                with open(summary_path, "w") as f:
                    f.write("=== RAG Evaluation Summary ===\n")
                    f.write(f"Total Samples: {summary['total_samples']}\n")
                    f.write(f"Metrics Evaluated: {', '.join(summary['metrics_evaluated'])}\n")
                    f.write(f"Pass Rate: {summary['pass_rate']*100:.2f}% (Threshold: {summary['threshold']})\n\n")
                    f.write("Mean Scores:\n")
                    for metric, score in summary["mean_scores"].items():
                        status = "PASS" if score >= self.config.threshold else "FAIL"
                        f.write(f"- {metric}: {score:.4f} ({status})\n")
                
                logger.info(f"CSV report generated at {report_path}")
                return report_path
                
            elif output_format == "json":
                report_path = os.path.join(self.config.report_path, "evaluation_results.json")
                results_df.to_json(report_path, orient="records", indent=2)
                logger.info(f"JSON report generated at {report_path}")
                return report_path
                
            elif output_format == "html":
                report_path = os.path.join(self.config.report_path, "evaluation_report.html")
                html_content = self._generate_html_report(results_df, summary)
                with open(report_path, "w") as f:
                    f.write(html_content)
                logger.info(f"HTML report generated at {report_path}")
                return report_path
                
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
                
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise
    
    def _generate_html_report(self, results_df: pd.DataFrame, summary: Dict) -> str:
        """Generate HTML report with visualizations"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .pass { color: green; }
                .fail { color: red; }
                .metric-row { margin-bottom: 15px; }
            </style>
        </head>
        <body>
            <h1>RAG System Evaluation Report</h1>
            
            <h2>Summary</h2>
            <p><strong>Total Samples:</strong> {total_samples}</p>
            <p><strong>Pass Rate:</strong> {pass_rate:.2f}% (Threshold: {threshold})</p>
            
            <h2>Performance Metrics</h2>
            {metrics_table}
            
            <h2>Detailed Results</h2>
            {detailed_table}
        </body>
        </html>
        """
        
        # Generate metrics table
        metrics_html = "<table><tr><th>Metric</th><th>Score</th><th>Status</th></tr>"
        for metric, score in summary["mean_scores"].items():
            status_class = "pass" if score >= self.config.threshold else "fail"
            status = "PASS" if score >= self.config.threshold else "FAIL"
            metrics_html += f"<tr><td>{metric}</td><td>{score:.4f}</td><td class='{status_class}'>{status}</td></tr>"
        metrics_html += "</table>"
        
        # Generate detailed results table
        detailed_html = results_df.to_html(classes="metric-row", index=False)
        
        return html_template.format(
            total_samples=summary["total_samples"],
            pass_rate=summary["pass_rate"] * 100,
            threshold=self.config.threshold,
            metrics_table=metrics_html,
            detailed_table=detailed_html
        )

# Example usage
if __name__ == "__main__":
    # Initialize configuration
    config = RagTestConfig(
        metrics=["faithfulness", "answer_relevancy", "context_precision"],
        sample_size=10,
        threshold=0.6,
        report_path="rag_test_reports"
    )
    
    # Create test runner
    test_runner = RagTestRunner(config)
    
    # Generate test data (or load from file)
    test_data = test_runner.generate_test_data(10)
    
    # Run evaluation
    results = test_runner.run_tests(test_data)
    
    # Generate report
    report_path = test_runner.generate_report(results, output_format="html")
    print(f"Report generated at: {report_path}")
```

This implementation provides a comprehensive RAG testing framework with the following features:

1. **Configurable Testing Parameters**: Customizable metrics, sample size, and threshold values
2. **Flexible Data Loading**: Support for CSV files or synthetic data generation
3. **Comprehensive Metrics**: Faithfulness, Answer Relevancy, Context Precision, Context Recall, Answer Correctness, and Harmfulness
4. **Detailed Reporting**: CSV, JSON, and HTML report formats with visualizations
5. **Error Handling**: Robust exception handling and logging
6. **Type Hints**: Full type annotations for better code clarity and IDE support
7. **Documentation**: Comprehensive docstrings and inline comments

The framework follows best practices including:
- Separation of concerns through modular design
- Configuration management with dataclasses
- Comprehensive error handling and logging
- Flexible reporting capabilities
- Type safety throughout the implementation

To use this framework, simply instantiate the `RagTestRunner` class with your desired configuration, load or generate test data, run the evaluation, and generate a comprehensive performance report.