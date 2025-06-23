# src/evaluation/tracking/runtime.py
"""
Runtime and resource usage tracking utilities
"""
import time
import json
import torch
import psutil
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import contextmanager


class RuntimeTracker:
    """Track training time, memory usage, and inference speed"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.peak_memory = 0
        self.inference_times = []
        
    def start(self):
        """Start timing"""
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
    def stop(self):
        """Stop timing and record peak memory"""
        self.end_time = time.time()
        if torch.cuda.is_available():
            self.peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            
    def get_training_hours(self) -> float:
        """Get total training time in hours"""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) / 3600
        
    def get_peak_vram_gb(self) -> float:
        """Get peak VRAM usage in GB"""
        return self.peak_memory
        
    def time_inference(self, model, dataloader, device, num_samples: int = 100):
        """
        Time model inference speed
        
        Args:
            model: The model to time
            dataloader: Dataloader for inference
            device: Device to run on
            num_samples: Number of samples to time
            
        Returns:
            Average inference time per sample in milliseconds
        """
        model.eval()
        times = []
        sample_count = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if sample_count >= num_samples:
                    break
                    
                # Move batch to device if needed
                if isinstance(batch, (list, tuple)):
                    batch_size = len(batch[0]) if hasattr(batch[0], '__len__') else 1
                else:
                    batch_size = 1
                
                # Time the inference
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
                start_time = time.time()
                
                # Run inference (adapt based on model type)
                try:
                    if hasattr(model, 'inference'):
                        _ = model.inference(*batch, device=device)
                    else:
                        _ = model(*batch)
                except:
                    # Fallback for different model interfaces
                    _ = model.forward(*batch)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
                end_time = time.time()
                
                # Record time per sample
                batch_time_ms = (end_time - start_time) * 1000
                per_sample_time = batch_time_ms / batch_size
                times.append(per_sample_time)
                
                sample_count += batch_size
                
        return sum(times) / len(times) if times else 0.0
        
    def save_runtime_stats(self, filepath: Path, additional_stats: Dict = None):
        """Save runtime statistics to JSON"""
        stats = {
            'train_hours': self.get_training_hours(),
            'peak_vram_GB': self.get_peak_vram_gb(),
            'inference_ms_per_patient': sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0.0,
        }
        
        if additional_stats:
            stats.update(additional_stats)
            
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)


@contextmanager
def track_runtime():
    """Context manager for runtime tracking"""
    tracker = RuntimeTracker()
    tracker.start()
    try:
        yield tracker
    finally:
        tracker.stop()


def measure_inference_speed(model, test_loader, device, num_samples: int = 100) -> float:
    """
    Standalone function to measure inference speed
    
    Returns:
        Average inference time per patient in milliseconds
    """
    model.eval()
    times = []
    sample_count = 0
    
    with torch.no_grad():
        for batch in test_loader:
            if sample_count >= num_samples:
                break
                
            # Determine batch size
            if isinstance(batch, (list, tuple)):
                # Handle different batch formats
                if hasattr(batch[0], 'shape'):
                    batch_size = batch[0].shape[0]
                elif hasattr(batch[0], '__len__'):
                    batch_size = len(batch[0])
                else:
                    batch_size = 1
            else:
                batch_size = 1
            
            # Time the forward pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            start_time = time.time()
            
            # Move to device and run inference
            try:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs, labels = batch[0], batch[1]
                    if hasattr(inputs, 'to'):
                        inputs = inputs.to(device)
                    _ = model(inputs)
                else:
                    # Handle single input case
                    if hasattr(batch, 'to'):
                        batch = batch.to(device)
                    _ = model(batch)
            except Exception as e:
                print(f"Warning: Could not measure inference speed: {e}")
                break
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            end_time = time.time()
            
            # Record time per sample
            batch_time_ms = (end_time - start_time) * 1000
            per_sample_time = batch_time_ms / batch_size
            times.append(per_sample_time)
            
            sample_count += batch_size
            
    return sum(times) / len(times) if times else 0.0