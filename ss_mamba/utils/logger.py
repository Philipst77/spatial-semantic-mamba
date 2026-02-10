"""
Logging Utilities
=================

Logging and experiment tracking for SS-Mamba.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def get_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Get configured logger.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'{name}_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


class TensorboardLogger:
    """
    Tensorboard logger for experiment tracking.
    
    Args:
        log_dir: Directory for tensorboard logs
        experiment_name: Name of experiment
    """
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: Optional[str] = None,
    ):
        if SummaryWriter is None:
            raise ImportError("tensorboard is required for TensorboardLogger")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if experiment_name:
            log_path = Path(log_dir) / f'{experiment_name}_{timestamp}'
        else:
            log_path = Path(log_dir) / timestamp
        
        self.writer = SummaryWriter(log_path)
        self.step = 0
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log scalar value."""
        step = step if step is not None else self.step
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: Optional[int] = None):
        """Log multiple scalars."""
        step = step if step is not None else self.step
        self.writer.add_scalars(tag, values, step)
    
    def log_histogram(self, tag: str, values, step: Optional[int] = None):
        """Log histogram."""
        step = step if step is not None else self.step
        self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, image, step: Optional[int] = None):
        """Log image."""
        step = step if step is not None else self.step
        self.writer.add_image(tag, image, step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = ''):
        """Log dictionary of metrics."""
        step = step if step is not None else self.step
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                tag = f'{prefix}/{key}' if prefix else key
                self.writer.add_scalar(tag, value, step)
    
    def increment_step(self):
        """Increment global step."""
        self.step += 1
    
    def close(self):
        """Close writer."""
        self.writer.close()


class ExperimentTracker:
    """
    Experiment tracker for managing training runs.
    
    Tracks hyperparameters, metrics, and model checkpoints.
    """
    
    def __init__(
        self,
        experiment_name: str,
        base_dir: str = 'experiments',
    ):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.base_dir / f'{experiment_name}_{timestamp}'
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoints_dir = self.run_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        self.logger = get_logger(experiment_name, str(self.run_dir))
        self.tb_logger = TensorboardLogger(str(self.run_dir / 'tensorboard'))
        
        self.metrics_history = []
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        import yaml
        config_path = self.run_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        self.logger.info(f"Config saved to {config_path}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int, phase: str = 'train'):
        """Log metrics for a training step."""
        self.metrics_history.append({
            'step': step,
            'phase': phase,
            **metrics
        })
        self.tb_logger.log_metrics(metrics, step, prefix=phase)
        
        metrics_str = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items() if isinstance(v, float)])
        self.logger.info(f"[{phase}] Step {step}: {metrics_str}")
    
    def save_checkpoint(self, state: Dict, filename: str):
        """Save model checkpoint."""
        import torch
        path = self.checkpoints_dir / filename
        torch.save(state, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def close(self):
        """Close all loggers."""
        self.tb_logger.close()
