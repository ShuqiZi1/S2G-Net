# src/evaluation/tracking/logger.py
"""
CSV logging utilities for training and validation metrics tracking
"""
import csv
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional


class CSVLogger:
    """Thread-safe CSV logger for epoch-wise metrics"""
    
    def __init__(self, filepath: Path, fieldnames: list):
        """
        Initialize CSV logger
        
        Args:
            filepath: Path to CSV file
            fieldnames: List of column names
        """
        self.filepath = Path(filepath)
        self.fieldnames = fieldnames
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Write header if file doesn't exist
        if not self.filepath.exists():
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
    
    def write(self, data: Dict[str, Any]):
        """Write a row of data to CSV"""
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            # Only write fields that exist in fieldnames
            filtered_data = {k: v for k, v in data.items() if k in self.fieldnames}
            writer.writerow(filtered_data)


class MetricsTracker:
    """Unified metrics tracking for training experiments"""
    
    def __init__(self, results_dir: Path, model_name: str):
        """
        Initialize metrics tracker
        
        Args:
            results_dir: Base directory for results
            model_name: Name of the model (e.g., 'mamba_gps', 'bilstm')
        """
        self.results_dir = Path(results_dir)
        self.model_name = model_name
        
        # Create loggers for train and validation metrics
        train_fields = ['epoch', 'loss', 'r2', 'mse', 'mae', 'mape', 'kappa', 'msle', 'lr']
        val_fields = ['epoch', 'loss', 'r2', 'mse', 'mae', 'mape', 'kappa', 'msle']
        
        self.train_logger = CSVLogger(
            self.results_dir / 'train_metrics.csv', 
            train_fields
        )
        self.val_logger = CSVLogger(
            self.results_dir / 'val_metrics.csv', 
            val_fields
        )
        
        # Runtime tracking
        self.start_time = None
        self.epoch_times = []
        
    def log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict, lr: float):
        """Log metrics for one epoch"""
        
        # Prepare train data
        train_data = {
            'epoch': epoch,
            'lr': lr,
            **train_metrics
        }
        
        # Prepare validation data  
        val_data = {
            'epoch': epoch,
            **val_metrics
        }
        
        # Write to CSV
        self.train_logger.write(train_data)
        self.val_logger.write(val_data)
        
    def start_training(self):
        """Mark start of training"""
        self.start_time = time.time()
        
    def end_epoch(self):
        """Mark end of epoch for timing"""
        if self.start_time is not None:
            epoch_time = time.time() - self.start_time
            self.epoch_times.append(epoch_time)
            self.start_time = time.time()  # Reset for next epoch
            
    def get_epoch_rate(self) -> float:
        """Calculate epochs per second"""
        if not self.epoch_times:
            return 0.0
        total_time = sum(self.epoch_times)
        return len(self.epoch_times) / total_time if total_time > 0 else 0.0