"""
Path utilities and directory management.
"""

from pathlib import Path
from typing import List, Dict, Union
import os


def ensure_directories(paths: Union[List[str], Dict[str, str], str]) -> None:
    """
    Ensure that directories exist, creating them if necessary.
    
    Args:
        paths: Single path, list of paths, or dictionary mapping names to paths
    """
    if isinstance(paths, str):
        Path(paths).mkdir(parents=True, exist_ok=True)
    elif isinstance(paths, list):
        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)
    elif isinstance(paths, dict):
        for path in paths.values():
            Path(path).mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError("paths must be string, list, or dictionary")


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    current_file = Path(__file__)
    # Navigate up from src/utils/paths.py to project root
    project_root = current_file.parent.parent.parent
    return project_root.resolve()


def get_data_path(relative_path: str = "") -> Path:
    """
    Get path to data directory.
    
    Args:
        relative_path: Relative path within data directory
        
    Returns:
        Path to data location
    """
    project_root = get_project_root()
    data_path = project_root / "datasets"
    
    if relative_path:
        data_path = data_path / relative_path
    
    return data_path


def get_output_path(relative_path: str = "") -> Path:
    """
    Get path to output directory.
    
    Args:
        relative_path: Relative path within output directory
        
    Returns:
        Path to output location
    """
    project_root = get_project_root()
    output_path = project_root / "outputs"
    
    if relative_path:
        output_path = output_path / relative_path
    
    return output_path


def get_model_path(model_name: str) -> Path:
    """
    Get path for saving/loading a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Path to model directory
    """
    return get_output_path("models") / model_name


def get_results_path(experiment_name: str) -> Path:
    """
    Get path for saving experiment results.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Path to results directory
    """
    return get_output_path("results") / experiment_name


def get_plots_path(plot_type: str = "") -> Path:
    """
    Get path for saving plots.
    
    Args:
        plot_type: Type/category of plots
        
    Returns:
        Path to plots directory
    """
    plots_path = get_output_path("plots")
    
    if plot_type:
        plots_path = plots_path / plot_type
    
    return plots_path


def list_available_datasets() -> List[str]:
    """
    List available datasets in the data directory.
    
    Returns:
        List of dataset names
    """
    data_path = get_data_path()
    
    if not data_path.exists():
        return []
    
    # Look for subdirectories that contain CSV files
    datasets = []
    for item in data_path.iterdir():
        if item.is_dir():
            # Check if directory contains CSV files
            csv_files = list(item.glob("*.csv"))
            if csv_files:
                datasets.append(item.name)
    
    return sorted(datasets)


def list_available_models() -> List[str]:
    """
    List available trained models.
    
    Returns:
        List of model names
    """
    models_path = get_output_path("models")
    
    if not models_path.exists():
        return []
    
    # Look for directories containing model files
    models = []
    for item in models_path.iterdir():
        if item.is_dir():
            # Check for common model files
            model_files = ['pytorch_model.bin', 'config.json', 'tokenizer.json']
            if any((item / f).exists() for f in model_files):
                models.append(item.name)
    
    return sorted(models)


def clean_filename(filename: str) -> str:
    """
    Clean a filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename
    """
    # Remove/replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove multiple consecutive underscores
    while '__' in filename:
        filename = filename.replace('__', '_')
    
    # Remove leading/trailing underscores
    filename = filename.strip('_')
    
    return filename


def get_unique_filename(base_path: Path, extension: str = "") -> Path:
    """
    Get a unique filename by appending a number if necessary.
    
    Args:
        base_path: Base path for the file
        extension: File extension (with or without dot)
        
    Returns:
        Unique file path
    """
    if extension and not extension.startswith('.'):
        extension = '.' + extension
    
    full_path = base_path.with_suffix(extension) if extension else base_path
    
    if not full_path.exists():
        return full_path
    
    # Append number to make unique
    counter = 1
    while True:
        stem = base_path.stem
        new_name = f"{stem}_{counter}"
        new_path = base_path.with_name(new_name).with_suffix(extension)
        
        if not new_path.exists():
            return new_path
        
        counter += 1