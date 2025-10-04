"""
Visualization utilities for model comparison results.

This module provides functions to create plots and visualizations
for perplexity comparisons and statistical analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional, Any

from .comparison import ComparisonResult

logger = logging.getLogger(__name__)


def setup_plotting_style():
    """Set up matplotlib and seaborn styling."""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12


def create_perplexity_histogram(perplexities: List[float], 
                               title: str = "Perplexity Distribution",
                               bins: int = 100,
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a histogram of perplexity values.
    
    Args:
        perplexities: List of perplexity values
        title: Plot title
        bins: Number of histogram bins
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    setup_plotting_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram
    ax.hist(perplexities, bins=bins, alpha=0.7, density=True)
    ax.set_xlabel('Perplexity')
    ax.set_ylabel('Density')
    ax.set_title(title)
    
    # Add statistics text
    mean_ppl = np.mean(perplexities)
    median_ppl = np.median(perplexities)
    std_ppl = np.std(perplexities)
    
    stats_text = f"Mean: {mean_ppl:.2f}\nMedian: {median_ppl:.2f}\nStd: {std_ppl:.2f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    return fig


def create_comparison_histogram(perplexities1: List[float], 
                               perplexities2: List[float],
                               labels: Tuple[str, str],
                               title: str = "Model Comparison",
                               bins: int = 100,
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Create overlapping histograms comparing two sets of perplexities.
    
    Args:
        perplexities1: First set of perplexity values
        perplexities2: Second set of perplexity values
        labels: Labels for the two sets
        title: Plot title
        bins: Number of histogram bins
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    setup_plotting_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create overlapping histograms
    ax.hist(perplexities1, bins=bins, alpha=0.6, label=labels[0], density=True)
    ax.hist(perplexities2, bins=bins, alpha=0.6, label=labels[1], density=True)
    
    ax.set_xlabel('Perplexity')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    
    # Add statistics
    stats1 = f"{labels[0]}: μ={np.mean(perplexities1):.2f}, σ={np.std(perplexities1):.2f}"
    stats2 = f"{labels[1]}: μ={np.mean(perplexities2):.2f}, σ={np.std(perplexities2):.2f}"
    
    ax.text(0.02, 0.98, f"{stats1}\n{stats2}", transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison plot to {save_path}")
    
    return fig


def create_quantile_table(perplexities: List[float]) -> pd.DataFrame:
    """
    Create a quantile table for perplexity values.
    
    Args:
        perplexities: List of perplexity values
        
    Returns:
        DataFrame with quantile statistics
    """
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    
    data = {
        'Quantile': quantiles,
        'Perplexity': [np.quantile(perplexities, q) for q in quantiles]
    }
    
    df = pd.DataFrame(data)
    df['Quantile'] = df['Quantile'].apply(lambda x: f"{x*100:.0f}%")
    
    return df


def stats_plot(perplexity_list: List[float], title: str) -> plt.Figure:
    """
    Create a statistics plot as in the original notebooks.
    
    Args:
        perplexity_list: List of perplexity values
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    return create_perplexity_histogram(perplexity_list, title=title)


def stats_plot_comparison(perplexity_list1: List[float], 
                         perplexity_list2: List[float],
                         bigtitle: str,
                         title1: str, 
                         title2: str) -> plt.Figure:
    """
    Create a comparison statistics plot as in the original notebooks.
    
    Args:
        perplexity_list1: First perplexity list
        perplexity_list2: Second perplexity list
        bigtitle: Main title
        title1: Label for first dataset
        title2: Label for second dataset
        
    Returns:
        Matplotlib figure
    """
    return create_comparison_histogram(
        perplexity_list1, perplexity_list2,
        labels=(title1, title2),
        title=bigtitle
    )


def create_comparison_plots(comparison_result: ComparisonResult,
                           save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
    """
    Create comprehensive comparison plots for a comparison result.
    
    Args:
        comparison_result: Result from model comparison
        save_dir: Optional directory to save plots
        
    Returns:
        Dictionary of created figures
    """
    figures = {}
    
    # Extract data
    ppl1 = comparison_result.model1_results['perplexities']
    ppl2 = comparison_result.model2_results['perplexities']
    
    # Individual histograms
    title1 = f"{comparison_result.model1_name} on {comparison_result.dataset_name}"
    fig1 = stats_plot(ppl1, title1)
    figures['model1_hist'] = fig1
    
    title2 = f"{comparison_result.model2_name} on {comparison_result.dataset_name}"
    fig2 = stats_plot(ppl2, title2)
    figures['model2_hist'] = fig2
    
    # Comparison histogram
    comp_title = f"Comparison: {comparison_result.model1_name} vs {comparison_result.model2_name}"
    fig3 = stats_plot_comparison(
        ppl1, ppl2, comp_title,
        comparison_result.model1_name,
        comparison_result.model2_name
    )
    figures['comparison'] = fig3
    
    # Save plots if directory provided
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        base_name = f"{comparison_result.model1_name}_vs_{comparison_result.model2_name}_{comparison_result.dataset_name}"
        
        fig1.savefig(f"{save_dir}/{base_name}_model1.png", dpi=300, bbox_inches='tight')
        fig2.savefig(f"{save_dir}/{base_name}_model2.png", dpi=300, bbox_inches='tight')
        fig3.savefig(f"{save_dir}/{base_name}_comparison.png", dpi=300, bbox_inches='tight')
        
        logger.info(f"Saved comparison plots to {save_dir}")
    
    return figures


def create_summary_table(comparison_results: Dict[str, ComparisonResult]) -> pd.DataFrame:
    """
    Create a summary table of all comparison results.
    
    Args:
        comparison_results: Dictionary of comparison results
        
    Returns:
        DataFrame with summary statistics
    """
    data = []
    
    for key, result in comparison_results.items():
        row = {
            'Comparison': key,
            'Model 1': result.model1_name,
            'Model 2': result.model2_name,
            'Dataset': result.dataset_name,
            'Model 1 Mean PPL': result.model1_results['mean_perplexity'],
            'Model 2 Mean PPL': result.model2_results['mean_perplexity'],
            'Improvement (%)': result.get_improvement(),
            'Model 1 Std': result.model1_results['std_perplexity'],
            'Model 2 Std': result.model2_results['std_perplexity']
        }
        data.append(row)
    
    return pd.DataFrame(data)