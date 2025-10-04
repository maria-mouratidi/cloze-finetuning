"""
Perplexity evaluation utilities.

This module implements perplexity calculation for language models,
based on the methodology used in the original notebooks.
"""

import torch
import math
import logging
from typing import List, Union, Dict, Any
from pathlib import Path

from transformers import GPT2LMHeadModel, GPT2Tokenizer

logger = logging.getLogger(__name__)


class PerplexityEvaluator:
    """
    Evaluates perplexity of language models on given texts.
    
    This class provides methods to calculate perplexity scores
    for both pre-trained and fine-tuned models.
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize the evaluator.
        
        Args:
            device: Device to run evaluation on ("auto", "cpu", "cuda")
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initialized PerplexityEvaluator on device: {self.device}")
    
    def calculate_perplexity(self, model_path: str, encodings: torch.Tensor) -> float:
        """
        Calculate perplexity for a model on given encodings.
        
        This method implements the perplexity calculation as used in the
        original notebooks, following HuggingFace documentation.
        
        Args:
            model_path: Path to model or model name (e.g., "gpt2")
            encodings: Tokenized input tensors
            
        Returns:
            Perplexity score
        """
        try:
            # Load model and tokenizer
            model = GPT2LMHeadModel.from_pretrained(model_path)
            tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            
            # Move model to device
            model.to(self.device)
            model.eval()
            
            # Ensure encodings are on the correct device
            encodings = encodings.to(self.device)
            
            max_length = model.config.n_positions
            seq_len = encodings.size(1)
            
            nlls = []
            prev_end_loc = 0
            
            # Calculate negative log-likelihoods
            for begin_loc in range(0, seq_len, max_length):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - prev_end_loc
                
                input_ids = encodings[:, begin_loc:end_loc]
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100
                
                with torch.no_grad():
                    outputs = model(input_ids, labels=target_ids)
                    neg_log_likelihood = outputs.loss * trg_len
                
                nlls.append(neg_log_likelihood)
                prev_end_loc = end_loc
                
                if end_loc == seq_len:
                    break
            
            # Calculate perplexity
            total_nll = torch.stack(nlls).sum()
            perplexity = torch.exp(total_nll / seq_len).item()
            
            return perplexity
            
        except Exception as e:
            logger.error(f"Error calculating perplexity: {e}")
            raise
    
    def evaluate_texts(self, model_path: str, texts: List[str]) -> List[float]:
        """
        Calculate perplexity for a list of texts.
        
        Args:
            model_path: Path to model or model name
            texts: List of text strings to evaluate
            
        Returns:
            List of perplexity scores
        """
        # Load tokenizer for encoding
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        perplexities = []
        
        for i, text in enumerate(texts):
            if i % 100 == 0:
                logger.info(f"Evaluating text {i+1}/{len(texts)}")
            
            # Tokenize text
            encoding = tokenizer(text, return_tensors="pt")
            
            # Calculate perplexity
            perplexity = self.calculate_perplexity(model_path, encoding.input_ids)
            perplexities.append(perplexity)
        
        logger.info(f"Completed evaluation of {len(texts)} texts")
        return perplexities
    
    def evaluate_dataset(self, model_path: str, dataset: Any) -> Dict[str, float]:
        """
        Evaluate perplexity on a dataset.
        
        Args:
            model_path: Path to model or model name
            dataset: Dataset object or list of texts
            
        Returns:
            Dictionary with evaluation metrics
        """
        if hasattr(dataset, 'to_list'):
            # Pandas DataFrame
            texts = dataset['text'].to_list()
        elif hasattr(dataset, '__iter__'):
            # List or similar iterable
            texts = list(dataset)
        else:
            raise ValueError("Unsupported dataset format")
        
        perplexities = self.evaluate_texts(model_path, texts)
        
        # Calculate summary statistics
        import statistics
        results = {
            'mean_perplexity': statistics.mean(perplexities),
            'median_perplexity': statistics.median(perplexities),
            'min_perplexity': min(perplexities),
            'max_perplexity': max(perplexities),
            'std_perplexity': statistics.stdev(perplexities) if len(perplexities) > 1 else 0,
            'num_samples': len(perplexities),
            'perplexities': perplexities
        }
        
        logger.info(f"Dataset evaluation complete: mean PPL = {results['mean_perplexity']:.2f}")
        return results


def calculate_perplexity_huggingface(model_path: str, encodings: torch.Tensor) -> float:
    """
    Standalone function for perplexity calculation.
    
    This function replicates the calcperplexity function from the original notebooks.
    
    Args:
        model_path: Path to model or model name
        encodings: Tokenized input tensors
        
    Returns:
        Perplexity score
    """
    evaluator = PerplexityEvaluator()
    return evaluator.calculate_perplexity(model_path, encodings)