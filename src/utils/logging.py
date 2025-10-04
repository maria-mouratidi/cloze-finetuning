"""
Logging configuration and utilities.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional


def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 log_dir: str = "logs") -> logging.Logger:
    """
    Set up logging configuration for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file name
        log_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'standard',
                'stream': sys.stdout
            }
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['console'],
                'level': log_level,
                'propagate': False
            }
        }
    }
    
    # Add file handler if log file is specified
    if log_file:
        log_path = Path(log_dir) / log_file
        log_config['handlers']['file'] = {
            'class': 'logging.FileHandler',
            'level': log_level,
            'formatter': 'detailed',
            'filename': str(log_path),
            'mode': 'a'
        }
        log_config['loggers']['']['handlers'].append('file')
    
    # Apply configuration
    logging.config.dictConfig(log_config)
    
    # Get logger
    logger = logging.getLogger('cloze_finetuning')
    logger.info(f"Logging initialized with level: {log_level}")
    
    if log_file:
        logger.info(f"Log file: {log_path}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f'cloze_finetuning.{name}')


class LoggingContext:
    """
    Context manager for temporary logging configuration.
    """
    
    def __init__(self, log_level: str):
        """
        Initialize the logging context.
        
        Args:
            log_level: Temporary log level
        """
        self.log_level = log_level
        self.original_level = None
        self.logger = logging.getLogger()
    
    def __enter__(self):
        """Enter the context."""
        self.original_level = self.logger.level
        self.logger.setLevel(getattr(logging, self.log_level.upper()))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        self.logger.setLevel(self.original_level)