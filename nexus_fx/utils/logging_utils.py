"""
Logging utilities for NEXUS-FX.
"""

import logging
import sys
from datetime import datetime
from typing import Dict, Any
import json


def setup_logger(name: str = 'nexus_fx', level: int = logging.INFO) -> logging.Logger:
    """
    Set up logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger


class MetricsLogger:
    """
    Logs training metrics to file and/or console.
    """
    
    def __init__(self, log_file: str = 'metrics.jsonl'):
        self.log_file = log_file
        self.logger = setup_logger('metrics')
    
    def log(self, metrics: Dict[str, Any], step: int, epoch: int = 0) -> None:
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Global step number
            epoch: Epoch number
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'epoch': epoch,
            **metrics
        }
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Log to console
        metrics_str = ', '.join(f'{k}: {v:.4f}' for k, v in metrics.items() if isinstance(v, (int, float)))
        self.logger.info(f"Step {step} - {metrics_str}")
    
    def log_summary(self, summary: str) -> None:
        """Log a summary string"""
        self.logger.info(summary)
