#!/usr/bin/env python3
"""
Complete pipeline script that runs training and evaluation.

This script combines the training and evaluation pipelines to reproduce
the complete experimental workflow from the original research.
"""

import argparse
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils import setup_logging, ProjectConfig, ensure_directories


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(description='Run complete Cloze fine-tuning pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--datasets', nargs='+', default=['cloze', 'peelle'],
                       help='Datasets to use for training')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and only run evaluation')
    parser.add_argument('--finetuned-model', type=str,
                       help='Path to existing fine-tuned model (if skipping training)')
    parser.add_argument('--gpt2-data-path', type=str,
                       help='Path to GPT-2 original data file')
    parser.add_argument('--output-dir', type=str, default='outputs/full_pipeline',
                       help='Output directory for all results')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(log_level=args.log_level, log_file='pipeline.log')
    logger.info("Starting complete Cloze fine-tuning pipeline")
    
    try:
        # Load configuration
        if args.config:
            config = ProjectConfig.from_yaml(args.config)
        else:
            config = ProjectConfig()
        
        # Update configuration
        config.update(
            datasets_to_use=args.datasets,
            output_dir=args.output_dir
        )
        
        # Ensure directories exist
        ensure_directories([args.output_dir, f"{args.output_dir}/models", f"{args.output_dir}/evaluation"])
        
        model_path = None
        
        # Training phase
        if not args.skip_training:
            logger.info("=" * 60)
            logger.info("PHASE 1: MODEL TRAINING")
            logger.info("=" * 60)
            
            from train import main as train_main
            
            # Set up training arguments
            train_args = [
                '--datasets'] + args.datasets + [
                '--output-dir', f"{args.output_dir}/models/gpt2_cloze_finetuned",
                '--log-level', args.log_level
            ]
            
            if args.config:
                train_args.extend(['--config', args.config])
            
            # Override sys.argv for training script
            original_argv = sys.argv
            sys.argv = ['train.py'] + train_args
            
            try:
                train_main()
                model_path = f"{args.output_dir}/models/gpt2_cloze_finetuned"
                logger.info(f"Training completed. Model saved to: {model_path}")
            finally:
                sys.argv = original_argv
        else:
            if not args.finetuned_model:
                logger.error("--finetuned-model must be provided when --skip-training is used")
                return
            model_path = args.finetuned_model
            logger.info(f"Skipping training. Using existing model: {model_path}")
        
        # Evaluation phase
        logger.info("=" * 60)
        logger.info("PHASE 2: MODEL EVALUATION")
        logger.info("=" * 60)
        
        from evaluate import main as evaluate_main
        
        # Set up evaluation arguments
        eval_args = [
            '--finetuned-model', model_path,
            '--output-dir', f"{args.output_dir}/evaluation",
            '--log-level', args.log_level,
            '--create-plots'
        ]
        
        if args.gpt2_data_path:
            eval_args.extend(['--gpt2-data-path', args.gpt2_data_path])
        
        # Override sys.argv for evaluation script
        original_argv = sys.argv
        sys.argv = ['evaluate.py'] + eval_args
        
        try:
            evaluate_main()
            logger.info("Evaluation completed successfully")
        finally:
            sys.argv = original_argv
        
        # Generate final summary
        logger.info("=" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"Complete pipeline finished successfully!")
        logger.info(f"Results saved in: {args.output_dir}")
        logger.info(f"Trained model: {model_path}")
        logger.info(f"Evaluation results: {args.output_dir}/evaluation")
        logger.info(f"Plots: {args.output_dir}/evaluation/plots")
        
        # Save pipeline configuration
        config.save_yaml(f"{args.output_dir}/pipeline_config.yaml")
        logger.info(f"Pipeline configuration saved: {args.output_dir}/pipeline_config.yaml")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()