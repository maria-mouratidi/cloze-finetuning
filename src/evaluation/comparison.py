"""
Model comparison utilities.

This module provides functionality to compare different models
on the same datasets and generate comparative statistics.
"""

import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

from .perplexity import PerplexityEvaluator

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Results from comparing two models."""
    model1_name: str
    model2_name: str
    dataset_name: str
    model1_results: Dict[str, float]
    model2_results: Dict[str, float]
    
    def get_improvement(self) -> float:
        """Calculate percentage improvement in mean perplexity."""
        ppl1 = self.model1_results['mean_perplexity']
        ppl2 = self.model2_results['mean_perplexity']
        return ((ppl1 - ppl2) / ppl1) * 100


class ModelComparator:
    """
    Compare multiple models on the same datasets.
    
    This class facilitates the comparison between pre-trained and
    fine-tuned models as done in the original research.
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize the comparator.
        
        Args:
            device: Device for evaluation
        """
        self.evaluator = PerplexityEvaluator(device=device)
        self.results = {}
        
    def compare_models(self, model_paths: Dict[str, str], 
                      datasets: Dict[str, Any]) -> Dict[str, ComparisonResult]:
        """
        Compare multiple models on multiple datasets.
        
        Args:
            model_paths: Dictionary mapping model names to paths
            datasets: Dictionary mapping dataset names to datasets
            
        Returns:
            Dictionary of comparison results
        """
        comparisons = {}
        
        # Evaluate each model on each dataset
        for dataset_name, dataset in datasets.items():
            logger.info(f"Evaluating on dataset: {dataset_name}")
            
            dataset_results = {}
            for model_name, model_path in model_paths.items():
                logger.info(f"Evaluating model: {model_name}")
                
                results = self.evaluator.evaluate_dataset(model_path, dataset)
                dataset_results[model_name] = results
                
                # Store for later use
                key = f"{model_name}_{dataset_name}"
                self.results[key] = results
            
            # Create pairwise comparisons
            model_names = list(model_paths.keys())
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model1, model2 = model_names[i], model_names[j]
                    
                    comparison = ComparisonResult(
                        model1_name=model1,
                        model2_name=model2,
                        dataset_name=dataset_name,
                        model1_results=dataset_results[model1],
                        model2_results=dataset_results[model2]
                    )
                    
                    comp_key = f"{model1}_vs_{model2}_{dataset_name}"
                    comparisons[comp_key] = comparison
                    
                    # Log improvement
                    improvement = comparison.get_improvement()
                    logger.info(f"{model2} vs {model1} on {dataset_name}: "
                              f"{improvement:.2f}% improvement")
        
        return comparisons
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics across all evaluations.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {}
        
        summary = {
            'total_evaluations': len(self.results),
            'models_datasets': list(self.results.keys()),
            'mean_perplexities': {
                key: result['mean_perplexity'] 
                for key, result in self.results.items()
            }
        }
        
        return summary


class ExperimentRunner:
    """
    Runs the complete experimental pipeline as in the original research.
    
    This includes:
    1. Special token vs no special token comparison
    2. Fine-tuned vs pre-trained model comparison on:
       - Cloze data (same structure as training)
       - GPT-2 original data (original training data)
       - Wikipedia data (generalization test)
    """
    
    def __init__(self, device: str = "auto"):
        """Initialize the experiment runner."""
        self.comparator = ModelComparator(device=device)
        self.experiment_results = {}
    
    def run_special_token_experiment(self, model_with_tokens: str,
                                   model_without_tokens: str,
                                   test_datasets: Dict[str, Any]) -> Dict[str, ComparisonResult]:
        """
        Run the special token comparison experiment.
        
        Args:
            model_with_tokens: Path to model trained with special tokens
            model_without_tokens: Path to model trained without special tokens
            test_datasets: Test datasets for evaluation
            
        Returns:
            Comparison results
        """
        logger.info("Running special token experiment")
        
        models = {
            'with_tokens': model_with_tokens,
            'without_tokens': model_without_tokens
        }
        
        results = self.comparator.compare_models(models, test_datasets)
        self.experiment_results['special_tokens'] = results
        
        return results
    
    def run_finetuning_experiment(self, pretrained_model: str,
                                 finetuned_model: str,
                                 evaluation_datasets: Dict[str, Any]) -> Dict[str, ComparisonResult]:
        """
        Run the main fine-tuning evaluation experiment.
        
        Args:
            pretrained_model: Path to pre-trained model (e.g., "gpt2")
            finetuned_model: Path to fine-tuned model
            evaluation_datasets: Datasets for evaluation (cloze, gpt2, wikipedia)
            
        Returns:
            Comparison results
        """
        logger.info("Running fine-tuning experiment")
        
        models = {
            'pretrained': pretrained_model,
            'finetuned': finetuned_model
        }
        
        results = self.comparator.compare_models(models, evaluation_datasets)
        self.experiment_results['finetuning'] = results
        
        return results
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get a summary of all experiments run."""
        return {
            'experiments': list(self.experiment_results.keys()),
            'results': self.experiment_results,
            'summary_stats': self.comparator.get_summary_statistics()
        }