"""
Project configuration management.
"""

import yaml
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List


@dataclass
class ProjectConfig:
    """
    Main project configuration class.
    
    This class centralizes all configuration parameters for the project.
    """
    
    # Project metadata
    project_name: str = "cloze-finetuning"
    version: str = "1.0.0"
    author: str = "Maria Mouratidi"
    
    # Paths
    data_dir: str = "datasets"
    output_dir: str = "outputs"
    models_dir: str = "models" 
    logs_dir: str = "logs"
    plots_dir: str = "plots"
    
    # Model configuration
    base_model: str = "gpt2"
    use_special_tokens: bool = True
    max_length: int = 512
    
    # Training configuration
    num_epochs: int = 6
    batch_size: int = 1
    learning_rate: float = 5e-5
    warmup_steps: int = 200
    weight_decay: float = 0.01
    random_seed: int = 42
    
    # Data configuration
    datasets_to_use: List[str] = None
    max_samples_per_dataset: int = 2000
    train_test_split: float = 0.9
    
    # Evaluation configuration
    evaluation_datasets: List[str] = None
    create_plots: bool = True
    save_results: bool = True
    
    # Computing configuration
    device: str = "auto"
    num_workers: int = 4
    
    def __post_init__(self):
        """Set default values for list fields."""
        if self.datasets_to_use is None:
            self.datasets_to_use = ["cloze", "peelle", "provo"]
        
        if self.evaluation_datasets is None:
            self.evaluation_datasets = ["cloze", "gpt2", "wikipedia"]
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'ProjectConfig':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            ProjectConfig instance
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, config_path: str) -> 'ProjectConfig':
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            ProjectConfig instance
        """
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def save_yaml(self, config_path: str):
        """
        Save configuration to YAML file.
        
        Args:
            config_path: Path to save YAML file
        """
        config_dict = asdict(self)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def save_json(self, config_path: str):
        """
        Save configuration to JSON file.
        
        Args:
            config_path: Path to save JSON file
        """
        config_dict = asdict(self)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def get_training_config(self):
        """
        Get training-specific configuration.
        
        Returns:
            Dictionary with training parameters
        """
        from ..models.config import TrainingConfig
        
        return TrainingConfig(
            model_name=self.base_model,
            use_special_tokens=self.use_special_tokens,
            output_dir=f"{self.output_dir}/{self.base_model}_finetuned",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            max_length=self.max_length,
            train_test_split=self.train_test_split,
            random_seed=self.random_seed,
            logging_dir=f"{self.logs_dir}/training"
        )
    
    def get_paths(self) -> Dict[str, Path]:
        """
        Get all project paths as Path objects.
        
        Returns:
            Dictionary mapping path names to Path objects
        """
        return {
            'data': Path(self.data_dir),
            'output': Path(self.output_dir),
            'models': Path(self.models_dir),
            'logs': Path(self.logs_dir),
            'plots': Path(self.plots_dir)
        }
    
    def update(self, **kwargs):
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")


def load_config(config_path: Optional[str] = None) -> ProjectConfig:
    """
    Load project configuration.
    
    Args:
        config_path: Optional path to configuration file.
                    If not provided, looks for default config files.
                    
    Returns:
        ProjectConfig instance
    """
    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if config_file.suffix == '.yaml' or config_file.suffix == '.yml':
            return ProjectConfig.from_yaml(config_path)
        elif config_file.suffix == '.json':
            return ProjectConfig.from_json(config_path)
        else:
            raise ValueError("Configuration file must be YAML or JSON")
    
    # Look for default configuration files
    default_configs = ['config.yaml', 'config.yml', 'config.json']
    
    for config_name in default_configs:
        config_file = Path(config_name)
        if config_file.exists():
            return load_config(str(config_file))
    
    # Return default configuration if no file found
    return ProjectConfig()