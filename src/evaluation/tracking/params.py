# src/evaluation/tracking/params.py
"""
Model parameter counting and size calculation utilities
"""
import os
import torch
from pathlib import Path
from typing import Dict, Any


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params
    }


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Calculate model size in memory (MB)
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return round(size_mb, 2)


def get_checkpoint_size_mb(checkpoint_path: Path) -> float:
    """
    Get checkpoint file size in MB
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        File size in MB
    """
    if not checkpoint_path.exists():
        return 0.0
    
    size_bytes = os.path.getsize(checkpoint_path)
    return round(size_bytes / (1024 ** 2), 2)


def format_parameter_count(param_count: int) -> str:
    """
    Format parameter count for display (e.g., 1.2M, 15.3K)
    
    Args:
        param_count: Number of parameters
        
    Returns:
        Formatted string
    """
    if param_count >= 1_000_000:
        return f"{param_count / 1_000_000:.1f}M"
    elif param_count >= 1_000:
        return f"{param_count / 1_000:.1f}K"
    else:
        return str(param_count)


def analyze_model_complexity(model: torch.nn.Module, checkpoint_path: Path = None) -> Dict[str, Any]:
    """
    Comprehensive model complexity analysis
    
    Args:
        model: PyTorch model
        checkpoint_path: Optional path to checkpoint file
        
    Returns:
        Dictionary with all complexity metrics
    """
    param_info = count_parameters(model)
    
    analysis = {
        'total_params': param_info['total_params'],
        'trainable_params': param_info['trainable_params'],
        'params_formatted': format_parameter_count(param_info['total_params']),
        'model_size_mb': get_model_size_mb(model),
    }
    
    if checkpoint_path and checkpoint_path.exists():
        analysis['checkpoint_size_mb'] = get_checkpoint_size_mb(checkpoint_path)
    
    return analysis


def save_model_info(model: torch.nn.Module, save_path: Path, checkpoint_path: Path = None):
    """
    Save model information to JSON file
    
    Args:
        model: PyTorch model
        save_path: Path to save JSON file
        checkpoint_path: Optional path to checkpoint file
    """
    import json
    
    info = analyze_model_complexity(model, checkpoint_path)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=2)


def compare_models(models: Dict[str, torch.nn.Module]) -> Dict[str, Dict]:
    """
    Compare multiple models
    
    Args:
        models: Dictionary of model_name -> model
        
    Returns:
        Comparison dictionary
    """
    comparison = {}
    
    for name, model in models.items():
        comparison[name] = analyze_model_complexity(model)
    
    return comparison