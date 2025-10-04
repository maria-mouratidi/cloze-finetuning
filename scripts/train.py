#!/usr/bin/env python3
"""
Main training script for Cloze fine-tuning project.

This script handles the complete training pipeline from data loading
to model fine-tuning and evaluation.
"""

import argparse
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils import setup_logging, ProjectConfig, ensure_directories
from src.data import ClozeDataLoader, DataPreprocessor
from src.models import ClozeTrainer
from src.evaluation import PerplexityEvaluator


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train GPT-2 on Cloze data')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--datasets', nargs='+', default=['cloze', 'peelle'],
                       help='Datasets to use for training')
    parser.add_argument('--output-dir', type=str, default='outputs/gpt2_cloze_model',
                       help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=6,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Training batch size')
    parser.add_argument('--max-samples', type=int, default=2000,
                       help='Maximum samples per dataset')
    parser.add_argument('--special-tokens', action='store_true', default=True,
                       help='Use special tokens (EOS) in training')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(log_level=args.log_level, log_file='training.log')
    logger.info("Starting Cloze fine-tuning training script")
    
    try:
        # Load configuration
        if args.config:
            config = ProjectConfig.from_yaml(args.config)
        else:
            config = ProjectConfig()
        
        # Override config with command line arguments
        config.update(
            datasets_to_use=args.datasets,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            max_samples_per_dataset=args.max_samples,
            use_special_tokens=args.special_tokens
        )
        
        logger.info(f"Configuration: {config}")
        
        # Ensure directories exist
        ensure_directories([config.output_dir, config.logs_dir])
        
        # Load and prepare data
        logger.info("Loading and preparing data...")
        data_loader = ClozeDataLoader(data_dir=config.data_dir)
        preprocessor = DataPreprocessor(add_special_tokens=config.use_special_tokens)
        
        # Combine datasets
        combined_data = data_loader.combine_datasets(
            config.datasets_to_use, 
            sample_size=config.max_samples_per_dataset
        )
        
        # Split data
        train_data, eval_data = preprocessor.train_test_split(
            combined_data, 
            train_size=config.train_test_split
        )
        
        # Prepare texts
        train_texts = preprocessor.prepare_texts(train_data)
        eval_texts = preprocessor.prepare_texts(eval_data)
        
        logger.info(f"Prepared {len(train_texts)} training texts and {len(eval_texts)} evaluation texts")
        
        # Set up training
        training_config = config.get_training_config()
        trainer = ClozeTrainer(training_config)
        
        # Train model
        logger.info("Starting model training...")
        train_result = trainer.train(train_texts, eval_texts)
        
        logger.info("Training completed successfully!")
        logger.info(f"Training results: {train_result}")
        
        # Quick evaluation
        logger.info("Running quick evaluation...")
        evaluator = PerplexityEvaluator()
        
        # Evaluate on a subset of eval data
        sample_texts = eval_texts[:100]  # Quick evaluation on first 100 samples
        eval_results = evaluator.evaluate_texts(training_config.output_dir, sample_texts)
        
        logger.info(f"Mean perplexity on evaluation set: {sum(eval_results)/len(eval_results):.2f}")
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()