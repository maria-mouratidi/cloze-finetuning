#!/usr/bin/env python3
"""
Evaluation script for comparing models and generating results.

This script runs the complete evaluation pipeline including perplexity
comparison between pre-trained and fine-tuned models on multiple datasets.
"""

import argparse
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils import setup_logging, ProjectConfig, ensure_directories
from src.data import ClozeDataLoader, load_gpt2_data, load_wikipedia_data
from src.evaluation import ModelComparator, ExperimentRunner, create_comparison_plots


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--pretrained-model', type=str, default='gpt2',
                       help='Pre-trained model name or path')
    parser.add_argument('--finetuned-model', type=str, required=True,
                       help='Path to fine-tuned model')
    parser.add_argument('--datasets', nargs='+', default=['cloze', 'gpt2', 'wikipedia'],
                       help='Datasets to evaluate on')
    parser.add_argument('--max-samples', type=int, default=1000,
                       help='Maximum samples per dataset for evaluation')
    parser.add_argument('--output-dir', type=str, default='outputs/evaluation',
                       help='Output directory for results')
    parser.add_argument('--create-plots', action='store_true', default=True,
                       help='Create comparison plots')
    parser.add_argument('--gpt2-data-path', type=str,
                       help='Path to GPT-2 original data file')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(log_level=args.log_level, log_file='evaluation.log')
    logger.info("Starting model evaluation script")
    
    try:
        # Load configuration
        config = ProjectConfig()
        config.update(output_dir=args.output_dir)
        
        logger.info(f"Evaluating models: {args.pretrained_model} vs {args.finetuned_model}")
        
        # Ensure directories exist
        ensure_directories([args.output_dir, f"{args.output_dir}/plots"])
        
        # Prepare datasets
        logger.info("Loading evaluation datasets...")
        datasets = {}
        
        # Load Cloze data
        if 'cloze' in args.datasets:
            data_loader = ClozeDataLoader(data_dir=config.data_dir)
            cloze_data = data_loader.combine_datasets(['cloze', 'peelle'], sample_size=args.max_samples)
            datasets['cloze'] = cloze_data['text'].tolist()
            logger.info(f"Loaded Cloze dataset: {len(datasets['cloze'])} samples")
        
        # Load GPT-2 data
        if 'gpt2' in args.datasets and args.gpt2_data_path:
            gpt2_data = load_gpt2_data(args.gpt2_data_path, max_samples=args.max_samples)
            datasets['gpt2'] = gpt2_data['text'].tolist()
            logger.info(f"Loaded GPT-2 dataset: {len(datasets['gpt2'])} samples")
        
        # Load Wikipedia data
        if 'wikipedia' in args.datasets:
            wiki_data = load_wikipedia_data(max_samples=args.max_samples)
            if not wiki_data.empty:
                datasets['wikipedia'] = wiki_data['text'].tolist()
                logger.info(f"Loaded Wikipedia dataset: {len(datasets['wikipedia'])} samples")
        
        if not datasets:
            logger.error("No datasets loaded. Please check your configuration.")
            return
        
        # Set up experiment runner
        experiment_runner = ExperimentRunner()
        
        # Run fine-tuning evaluation experiment
        logger.info("Running fine-tuning evaluation experiment...")
        results = experiment_runner.run_finetuning_experiment(
            pretrained_model=args.pretrained_model,
            finetuned_model=args.finetuned_model,
            evaluation_datasets=datasets
        )
        
        # Print results summary
        logger.info("Evaluation Results Summary:")
        logger.info("=" * 50)
        
        for key, comparison in results.items():
            improvement = comparison.get_improvement()
            logger.info(f"{comparison.dataset_name} dataset:")
            logger.info(f"  {comparison.model1_name} mean PPL: {comparison.model1_results['mean_perplexity']:.2f}")
            logger.info(f"  {comparison.model2_name} mean PPL: {comparison.model2_results['mean_perplexity']:.2f}")
            logger.info(f"  Improvement: {improvement:.2f}%")
            logger.info("")
        
        # Create plots if requested
        if args.create_plots:
            logger.info("Creating comparison plots...")
            
            for key, comparison in results.items():
                plot_dir = f"{args.output_dir}/plots/{comparison.dataset_name}"
                ensure_directories([plot_dir])
                
                figures = create_comparison_plots(comparison, save_dir=plot_dir)
                logger.info(f"Saved plots for {comparison.dataset_name} comparison")
        
        # Save results summary
        import json
        summary = experiment_runner.get_experiment_summary()
        
        with open(f"{args.output_dir}/evaluation_results.json", 'w') as f:
            # Convert non-serializable objects to serializable format
            serializable_summary = {}
            for key, value in summary.items():
                if key == 'results':
                    serializable_summary[key] = {
                        k: {
                            'model1_name': v.model1_name,
                            'model2_name': v.model2_name,
                            'dataset_name': v.dataset_name,
                            'improvement_percent': v.get_improvement(),
                            'model1_mean_ppl': v.model1_results['mean_perplexity'],
                            'model2_mean_ppl': v.model2_results['mean_perplexity']
                        } for k, v in value['finetuning'].items()
                    }
                else:
                    serializable_summary[key] = value
            
            json.dump(serializable_summary, f, indent=2)
        
        logger.info(f"Evaluation results saved to {args.output_dir}/evaluation_results.json")
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        raise


if __name__ == "__main__":
    main()