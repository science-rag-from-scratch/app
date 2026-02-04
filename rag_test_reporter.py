"""
RAG Testing Reporter using RAGAS Framework

This module implements a comprehensive testing and reporting system for RAG applications using the RAGAS framework.
It provides functionality to evaluate RAG system performance across multiple metrics and generate detailed reports.

Author: AI Assistant
Date: 2023-11-15
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

import pandas as pd
import numpy as np
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
    answer_similarity
)
from ragas.dataset import Dataset
from datasets import Dataset as HFDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Configuration for evaluation metrics."""
    faithfulness: bool = True
    answer_relevancy: bool = True
    context_precision: bool = True
    context_recall: bool = True
    answer_correctness: bool = True
    answer_similarity: bool = False
    
    def get_metrics(self) -> List:
        """Return list of enabled metrics."""
        metrics = []
        if self.faithfulness:
            metrics.append(faithfulness)
        if self.answer_relevancy:
            metrics.append(answer_relevancy)
        if self.context_precision:
            metrics.append(context_precision)
        if self.context_recall:
            metrics.append(context_recall)
        if self.answer_correctness:
            metrics.append(answer_correctness)
        if self.answer_similarity:
            metrics.append(answer_similarity)
        return metrics


class RAGTestReporter:
    """
    RAG testing and reporting system using RAGAS framework.
    
    This class provides a comprehensive solution for evaluating RAG system performance
    through multiple metrics and generating detailed reports.
    """
    
    def __init__(
        self,
        metrics_config: Optional[EvaluationMetrics] = None,
        output_dir: str = "rag_test_results",
        report_format: str = "json"
    ):
        """
        Initialize the RAG Test Reporter.
        
        Args:
            metrics_config: Configuration for evaluation metrics. If None, uses default.
            output_dir: Directory to save test results and reports.
            report_format: Format for the report ('json' or 'csv').
        """
        self.metrics_config = metrics_config or EvaluationMetrics()
        self.output_dir = output_dir
        self.report_format = report_format.lower()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize results storage
        self.results: Dict[str, Any] = {}
        self.dataset: Optional[HFDataset] = None
        
    def load_test_data(self, data_path: str) -> None:
        """
        Load test data from a file.
        
        Args:
            data_path: Path to the test data file (JSON or CSV).
            
        Raises:
            FileNotFoundError: If the data file doesn't exist.
            ValueError: If the file format is unsupported.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Test data file not found: {data_path}")
            
        file_ext = os.path.splitext(data_path)[1].lower()
        
        try:
            if file_ext == ".json":
                with open(data_path, 'r') as f:
                    data = json.load(f)
                self.dataset = Dataset.from_dict(data)
            elif file_ext == ".csv":
                df = pd.read_csv(data_path)
                self.dataset = Dataset.from_pandas(df)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
            logger.info(f"Successfully loaded test data from {data_path}")
            
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            raise
            
    def generate_test_data(
        self,
        questions: List[str],
        ground_truths: List[str],
        answers: List[str],
        contexts: List[List[str]],
        sample_size: Optional[int] = None
    ) -> None:
        """
        Generate test data from provided components.
        
        Args:
            questions: List of questions
            ground_truths: List of ground truth answers
            answers: List of generated answers
            contexts: List of context lists for each question
            sample_size: Number of samples to use (if None, use all)
        """
        if not (len(questions) == len(ground_truths) == len(answers) == len(contexts)):
            raise ValueError("All input lists must have the same length")
            
        if sample_size is not None and sample_size > len(questions):
            sample_size = len(questions)
            
        # Create dataset
        data = {
            "question": questions[:sample_size] if sample_size else questions,
            "ground_truth": ground_truths[:sample_size] if sample_size else ground_truths,
            "answer": answers[:sample_size] if sample_size else answers,
            "contexts": contexts[:sample_size] if sample_size else contexts
        }
        
        self.dataset = Dataset.from_dict(data)
        logger.info(f"Generated test dataset with {len(self.dataset)} samples")
        
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run the RAG evaluation using configured metrics.
        
        Returns:
            Dictionary containing evaluation results.
            
        Raises:
            RuntimeError: If no test data is available.
        """
        if self.dataset is None:
            raise RuntimeError("No test data available for evaluation")
            
        metrics = self.metrics_config.get_metrics()
        if not metrics:
            logger.warning("No evaluation metrics enabled")
            return {}
            
        try:
            logger.info(f"Starting evaluation with {len(metrics)} metrics")
            results = evaluate(
                dataset=self.dataset,
                metrics=metrics,
                raise_exceptions=True
            )
            
            # Store results
            self.results = results.to_dict()
            
            # Save detailed results
            self._save_detailed_results()
            
            logger.info("Evaluation completed successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
            
    def _save_detailed_results(self) -> None:
        """Save detailed evaluation results to files."""
        if not self.results:
            logger.warning("No results to save")
            return
            
        # Save as CSV
        csv_path = os.path.join(self.output_dir, "detailed_results.csv")
        df = pd.DataFrame(self.results)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved detailed results to {csv_path}")
        
        # Save as JSON
        json_path = os.path.join(self.output_dir, "detailed_results.json")
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Saved detailed results to {json_path}")
        
    def generate_report(self) -> str:
        """
        Generate a performance report based on evaluation results.
        
        Returns:
            Path to the generated report file.
        """
        if not self.results:
            logger.warning("No evaluation results available for reporting")
            return ""
            
        report_data = {
            "summary": self._generate_summary(),
            "metric_scores": self._get_metric_scores(),
            "sample_results": self._get_sample_results(),
            "recommendations": self._generate_recommendations()
        }
        
        report_path = os.path.join(self.output_dir, "performance_report")
        
        if self.report_format == "json":
            report_path += ".json"
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
        else:
            report_path += ".csv"
            df = pd.DataFrame([report_data])
            df.to_csv(report_path, index=False)
            
        logger.info(f"Generated performance report: {report_path}")
        return report_path
        
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the evaluation results."""
        if not self.results:
            return {}
            
        # Calculate overall score if multiple metrics are present
        metric_names = [m.name for m in self.metrics_config.get_metrics()]
        if len(metric_names) > 1:
            scores = [self.results.get(name, 0) for name in metric_names]
            overall_score = np.mean(scores)
        else:
            overall_score = next(iter(self.results.values())) if self.results else 0
            
        return {
            "total_samples": len(self.dataset) if self.dataset else 0,
            "metrics_evaluated": len(metric_names),
            "overall_score": overall_score,
            "best_metric": max(self.results.items(), key=lambda x: x[1]) if self.results else ("N/A", 0),
            "worst_metric": min(self.results.items(), key=lambda x: x[1]) if self.results else ("N/A", 0)
        }
        
    def _get_metric_scores(self) -> Dict[str, float]:
        """Extract metric scores from results."""
        return self.results
        
    def _get_sample_results(self) -> List[Dict[str, Any]]:
        """Extract sample evaluation results."""
        if self.dataset is None:
            return []
            
        sample_size = min(5, len(self.dataset))
        sample_results = []
        
        for i in range(sample_size):
            sample = {
                "question": self.dataset["question"][i],
                "ground_truth": self.dataset["ground_truth"][i],
                "answer": self.dataset["answer"][i],
                "contexts": self.dataset["contexts"][i]
            }
            sample_results.append(sample)
            
        return sample_results
        
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations based on results."""
        if not self.results:
            return ["No recommendations available without evaluation results"]
            
        recommendations = []
        
        # Analyze each metric
        if "faithfulness" in self.results and self.results["faithfulness"] < 0.7:
            recommendations.append(
                "Low faithfulness detected: Improve the faithfulness of generated answers "
                "by ensuring they are based solely on provided context."
            )
            
        if "answer_relevancy" in self.results and self.results["answer_relevancy"] < 0.7:
            recommendations.append(
                "Low answer relevancy: Improve answer relevance by ensuring responses "
                "directly address the user's question."
            )
            
        if "context_recall" in self.results and self.results["context_recall"] < 0.7:
            recommendations.append(
                "Low context recall: Improve context retrieval to include more relevant "
                "information in the context window."
            )
            
        if "answer_correctness" in self.results and self.results["answer_correctness"] < 0.7:
            recommendations.append(
                "Low answer correctness: Improve factual accuracy by enhancing retrieval "
                "or response generation processes."
            )
            
        # Default recommendation if no specific issues
        if not recommendations:
            recommendations.append(
                "Overall performance is good. Consider fine-tuning retrieval or generation "
                "models for further improvements."
            )
            
        return recommendations
        
    def visualize_results(self) -> None:
        """Generate visualizations of the evaluation results."""
        if not self.results:
            logger.warning("No results to visualize")
            return
            
        try:
            import matplotlib.pyplot as plt
            
            # Create bar chart of metric scores
            plt.figure(figsize=(10, 6))
            metrics = list(self.results.keys())
            scores = list(self.results.values())
            
            plt.bar(metrics, scores, color='skyblue')
            plt.xlabel('Metrics')
            plt.ylabel('Scores')
            plt.title('RAG System Performance Metrics')
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
            
            # Add score labels on bars
            for i, v in enumerate(scores):
                plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
                
            plt.tight_layout()
            viz_path = os.path.join(self.output_dir, "performance_visualization.png")
            plt.savefig(viz_path)
            plt.close()
            
            logger.info(f"Saved visualization to {viz_path}")
            
        except ImportError:
            logger.warning("Matplotlib not installed. Skipping visualization.")
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")


def main():
    """Example usage of the RAGTestReporter."""
    try:
        # Initialize the reporter with custom metrics
        reporter = RAGTestReporter(
            metrics_config=EvaluationMetrics(
                faithfulness=True,
                answer_relevancy=True,
                context_precision=True,
                context_recall=True,
                answer_correctness=True,
                answer_similarity=False
            ),
            output_dir="rag_test_results",
            report_format="json"
        )
        
        # Example test data
        questions = [
            "What is the capital of France?",
            "Explain the theory of relativity.",
            "Who wrote 'Romeo and Juliet'?",
            "What is photosynthesis?",
            "Describe the process of DNA replication."
        ]
        
        ground_truths = [
            "The capital of France is Paris.",
            "The theory of relativity is a theory in physics that describes gravity as a geometric property of space and time.",
            "William Shakespeare wrote 'Romeo and Juliet'.",
            "Photosynthesis is the process by which green plants use sunlight to synthesize foods from carbon dioxide and water.",
            "DNA replication is the process by which a double-stranded DNA molecule is copied to produce two identical DNA molecules."
        ]
        
        answers = [
            "Paris is the capital city of France.",
            "Einstein's theory of relativity explains that gravity is not a force but a curvature of spacetime caused by mass and energy.",
            "The famous play 'Romeo and Juliet' was written by William Shakespeare.",
            "Photosynthesis is the process where plants use sunlight to convert carbon dioxide and water into glucose and oxygen.",
            "DNA replication is a semi-conservative process where each strand of the original DNA molecule serves as a template for a new complementary strand."
        ]
        
        contexts = [
            ["Paris is the capital and most populous city of France."],
            ["The theory of relativity usually encompasses two interrelated theories by Albert Einstein: special relativity and general relativity."],
            ["William Shakespeare was an English playwright, poet and actor, widely regarded as the greatest writer in the English language and the world's greatest dramatist."],
            ["Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy."],
            ["DNA replication is the biological process of producing two identical replicas of DNA from one original DNA molecule."]
        ]
        
        # Generate test data
        reporter.generate_test_data(questions, ground_truths, answers, contexts)
        
        # Run evaluation
        results = reporter.run_evaluation()
        
        # Generate report
        report_path = reporter.generate_report()
        
        # Generate visualizations
        reporter.visualize_results()
        
        print(f"Evaluation completed. Results saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Error in RAG testing: {str(e)}")
        raise


if __name__ == "__main__":
    main()