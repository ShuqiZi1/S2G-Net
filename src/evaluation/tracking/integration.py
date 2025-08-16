# src/tracking/integration.py
"""
Integration utilities for adding tracking to existing training scripts
with minimal code changes
"""
import json
import time
import torch
from pathlib import Path
from typing import Dict, Any, Optional

from .logger import MetricsTracker
from .runtime import RuntimeTracker, measure_inference_speed
from .params import analyze_model_complexity


class TrainingTracker:
    """
    Unified tracking wrapper that can be easily integrated into existing training loops
    """
    
    def __init__(self, results_dir: Path, model_name: str, config: Dict):
        """
        Initialize training tracker
        
        Args:
            results_dir: Directory to save results
            model_name: Name of the model for logging
            config: Training configuration dictionary
        """
        self.results_dir = Path(results_dir)
        self.model_name = model_name
        self.config = config
        
        # Create tracking components
        self.metrics_tracker = MetricsTracker(self.results_dir, model_name)
        self.runtime_tracker = RuntimeTracker()
        
        # Training state
        self.current_epoch = 0
        self.best_val_r2 = -float('inf')
        self.best_epoch = 0
        self.epoch_start_time = None
        
    def start_training(self):
        """Call at the beginning of training"""
        self.metrics_tracker.start_training()
        self.runtime_tracker.start()
        print(f"🔍 Tracking enabled for {self.model_name}")
        print(f"📁 Results will be saved to: {self.results_dir}")
        
    def start_epoch(self, epoch: int):
        """Call at the beginning of each epoch"""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        
    def end_epoch(self, train_metrics: Dict, val_metrics: Dict, learning_rate: float):
        """
        Call at the end of each epoch with metrics
        
        Args:
            train_metrics: Dictionary of training metrics
            val_metrics: Dictionary of validation metrics  
            learning_rate: Current learning rate
        """
        # Track best validation R²
        val_r2 = val_metrics.get('r2', -float('inf'))
        if val_r2 > self.best_val_r2:
            self.best_val_r2 = val_r2
            self.best_epoch = self.current_epoch
            
        # Log metrics to CSV
        self.metrics_tracker.log_epoch(
            self.current_epoch, 
            train_metrics, 
            val_metrics, 
            learning_rate
        )
        
        # Track epoch timing
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            self.metrics_tracker.epoch_times.append(epoch_time)
        
        # Print progress
        print(f"Epoch {self.current_epoch}: "
              f"train_loss={train_metrics.get('loss', 0):.4f}, "
              f"val_loss={val_metrics.get('loss', 0):.4f}, "
              f"val_r2={val_r2:.4f}, "
              f"lr={learning_rate:.2e}")
    
    def end_training(self, model: torch.nn.Module, test_loader = None):
        """
        Call at the end of training
        
        Args:
            model: Trained model
            test_loader: Optional test dataloader for inference timing
        """
        self.runtime_tracker.stop()
        
        # Analyze model complexity
        model_info = analyze_model_complexity(model)
        
        # Measure inference speed if test loader provided
        inference_time = 0.0
        if test_loader is not None:
            try:
                device = next(model.parameters()).device
                inference_time = measure_inference_speed(model, test_loader, device)
            except Exception as e:
                print(f"⚠️  Could not measure inference speed: {e}")
        
        # Calculate epoch rate
        epoch_rate = self.metrics_tracker.get_epoch_rate()
        
        # Compile runtime statistics
        runtime_stats = {
            'train_hours': self.runtime_tracker.get_training_hours(),
            'peak_vram_GB': self.runtime_tracker.get_peak_vram_gb(),
            'inference_ms_per_patient': inference_time,
            'epoch_per_sec': epoch_rate,
            'total_epochs': self.current_epoch + 1,
            'best_epoch': self.best_epoch,
            'best_val_r2': self.best_val_r2,
            'total_params': model_info['total_params'],
            'model_size_mb': model_info['model_size_mb'],
        }
        
        # Save runtime statistics
        runtime_file = self.results_dir / 'runtime.json'
        with open(runtime_file, 'w') as f:
            json.dump(runtime_stats, f, indent=2)
        
        print(f"✅ Training completed!")
        print(f"📊 Best validation R²: {self.best_val_r2:.4f} at epoch {self.best_epoch}")
        print(f"⏱️  Training time: {runtime_stats['train_hours']:.2f} hours")
        print(f"🔧 Model parameters: {model_info['params_formatted']}")
        print(f"💾 Peak VRAM: {runtime_stats['peak_vram_GB']:.1f} GB")
        if inference_time > 0:
            print(f"⚡ Inference speed: {inference_time:.1f} ms/patient")


def extract_metrics_from_lightning_logs(lightning_logs: Dict) -> Dict[str, float]:
    """
    Extract standard metrics from PyTorch Lightning logs
    
    Args:
        lightning_logs: Dictionary from Lightning's log output
        
    Returns:
        Standardized metrics dictionary
    """
    metrics = {}
    
    # Map common metric names
    metric_mapping = {
        'train_loss': ['train_loss', 'loss'],
        'val_loss': ['val_loss', 'validation_loss'],
        'r2': ['r2', 'val_r2', 'test_r2'],
        'mse': ['mse', 'val_mse', 'test_mse'],
        'mae': ['mae', 'val_mae', 'test_mae'],
        'mape': ['mape', 'val_mape', 'test_mape'],
        'kappa': ['kappa', 'val_kappa', 'test_kappa'],
        'msle': ['msle', 'val_msle', 'test_msle']
    }
    
    for standard_name, possible_names in metric_mapping.items():
        for name in possible_names:
            if name in lightning_logs:
                metrics[standard_name] = float(lightning_logs[name])
                break
    
    return metrics


def get_learning_rate_from_scheduler(scheduler) -> float:
    """
    Extract current learning rate from scheduler
    
    Args:
        scheduler: PyTorch learning rate scheduler
        
    Returns:
        Current learning rate
    """
    try:
        if hasattr(scheduler, 'get_last_lr'):
            return scheduler.get_last_lr()[0]
        elif hasattr(scheduler, 'get_lr'):
            lr_list = scheduler.get_lr()
            return lr_list[0] if lr_list else 0.0
        else:
            return 0.0
    except:
        return 0.0


# Minimal integration functions for quick patching of existing code

def quick_track_epoch(results_dir: Path, model_name: str, epoch: int, 
                     train_dict: Dict, val_dict: Dict, lr: float = 0.0):
    """
    Quick function to track a single epoch - can be called directly from existing code
    
    Args:
        results_dir: Directory to save CSV files
        model_name: Model name for file naming
        epoch: Current epoch number
        train_dict: Training metrics
        val_dict: Validation metrics
        lr: Learning rate
    """
    from .logger import CSVLogger
    
    # Initialize loggers
    train_fields = ['epoch', 'loss', 'r2', 'mse', 'mae', 'mape', 'kappa', 'msle', 'lr']
    val_fields = ['epoch', 'loss', 'r2', 'mse', 'mae', 'mape', 'kappa', 'msle']
    
    train_logger = CSVLogger(results_dir / 'train_metrics.csv', train_fields)
    val_logger = CSVLogger(results_dir / 'val_metrics.csv', val_fields)
    
    # Log data
    train_data = {'epoch': epoch, 'lr': lr, **train_dict}
    val_data = {'epoch': epoch, **val_dict}
    
    train_logger.write(train_data)
    val_logger.write(val_data)


def quick_save_runtime(results_dir: Path, model: torch.nn.Module, 
                      train_hours: float = 0.0, inference_ms: float = 0.0):
    """
    Quick function to save runtime statistics
    
    Args:
        results_dir: Directory to save files
        model: PyTorch model
        train_hours: Training time in hours
        inference_ms: Inference time per sample in ms
    """
    from .params import analyze_model_complexity
    
    # Get model info
    model_info = analyze_model_complexity(model)
    
    # Get VRAM usage
    peak_vram = 0.0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    
    runtime_stats = {
        'train_hours': train_hours,
        'peak_vram_GB': peak_vram,
        'inference_ms_per_patient': inference_ms,
        'total_params': model_info['total_params'],
        'model_size_mb': model_info['model_size_mb'],
    }
    
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(results_dir) / 'runtime.json', 'w') as f:
        json.dump(runtime_stats, f, indent=2)