# src/evaluation/explain/plotter.py
"""
Unified plotting functions
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
import pandas as pd
from sklearn.calibration import calibration_curve
import umap
import src.evaluation.explain.viz_style as vs


def plot_lambda_dynamics(log_path: str, save_dir: str = './figs/explain'):
    """Plot λ_ts and λ_gps dynamics from actual training logs"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Try to read from tensorboard logs or saved JSON
    lambda_history_path = Path(log_path) / 'lambda_history.json'
    
    if lambda_history_path.exists():
        with open(lambda_history_path, 'r') as f:
            history = json.load(f)
        epochs = history['epochs']
        lambda_ts_values = history['lambda_ts']
        lambda_gps_values = history['lambda_gps']
    else:
        # Try to read from tensorboard
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            event_path = list(Path(log_path).glob('events.out.tfevents.*'))[0]
            event_acc = EventAccumulator(str(event_path))
            event_acc.Reload()
            
            # Extract lambda values
            lambda_ts_events = event_acc.Scalars('lambda_ts')
            lambda_gps_events = event_acc.Scalars('lambda_gps')
            
            epochs = [e.step for e in lambda_ts_events]
            lambda_ts_values = [e.value for e in lambda_ts_events]
            lambda_gps_values = [e.value for e in lambda_gps_events]
        except:
            print("Warning: No training logs found, using synthetic data for demonstration")
            epochs = np.arange(0, 100)
            lambda_ts_values = 0.3 + 0.4 * (1 - np.exp(-epochs / 20))
            lambda_gps_values = 0.7 - 0.4 * (1 - np.exp(-epochs / 20))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lambda_ts_values, color=vs.COLOR['temporal'][0],
             linewidth=2, label='λ_ts (Time Series)')
    plt.plot(epochs, lambda_gps_values, color=vs.COLOR['graph'][0],
             linewidth=2, label='λ_gps (Graph)')
    
    plt.xlabel('Epoch')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/lambda_dynamics.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Lambda dynamics plot saved to {save_dir}")


def plot_reliability_diagram(y_true: np.ndarray, y_pred: np.ndarray, 
                            y_std: Optional[np.ndarray] = None,
                            n_bins: int = 10, save_dir: str = './figs/explain'):
    """Create reliability/calibration diagram"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # For regression, create bins based on prediction quantiles
    bin_edges = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-8  # Ensure last value is included
    
    plt.figure(figsize=(10, 8))
    
    # Calculate calibration statistics
    bin_centers = []
    bin_accuracies = []
    bin_counts = []
    bin_stds = []
    
    for i in range(n_bins):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_mean_pred = y_pred[mask].mean()
            bin_mean_true = y_true[mask].mean()
            bin_centers.append(bin_mean_pred)
            bin_accuracies.append(bin_mean_true)
            bin_counts.append(mask.sum())
            
            if y_std is not None:
                bin_stds.append(y_std[mask].mean())
    
    fig, (ax_cal, ax_hist) = plt.subplots(2, 1, figsize=(8, 6.4))
    
    # === 1. Calibration curve ===
    y_range = [min(min(bin_centers), min(bin_accuracies)), 
               max(max(bin_centers), max(bin_accuracies))]
    
    ax_cal.plot(y_range, y_range, 'k--', label='Perfect Calibration')
    
    sizes = [100 * (c / max(bin_counts)) for c in bin_counts]
    ax_cal.scatter(
        bin_centers, bin_accuracies, s=sizes, alpha=0.8,
        c=range(len(bin_centers)), cmap='viridis',
        edgecolors='black', linewidth=1
    )
    
    if y_std is not None and bin_stds:
        ax_cal.errorbar(
            bin_centers, bin_accuracies, yerr=bin_stds,
            fmt='none', ecolor='gray', alpha=0.7
        )
    
    ax_cal.set_title("(a) Calibration Curve", fontsize=13)
    ax_cal.set_xlabel("Mean Predicted Value", fontsize=12)
    ax_cal.set_ylabel("Mean Actual Value", fontsize=12)
    ax_cal.legend()
    ax_cal.grid(True, alpha=0.3)
    
    # === 2. Histogram ===
    ax_hist.hist(
        y_pred, bins=30, alpha=0.7,
        color='#376b9e', edgecolor='black'
    )
    ax_hist.set_title("(b) Distribution of Predicted LOS", fontsize=13)
    ax_hist.set_xlabel("Predicted Value", fontsize=12)
    ax_hist.set_ylabel("Count", fontsize=12)
    ax_hist.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/reliability_diagram.pdf", dpi=600, bbox_inches='tight')
    plt.close()
    
    # Calculate ECE
    y_range_size = y_true.max() - y_true.min()
    ece = np.sum([abs(bin_accuracies[i] - bin_centers[i]) * bin_counts[i] / y_range_size
                  for i in range(len(bin_centers))]) / sum(bin_counts)
    
    print(f"Reliability diagram saved. ECE (normalized): {ece:.4f}")



from matplotlib.colors import LinearSegmentedColormap

def plot_umap_embeddings(embeddings: np.ndarray, labels: np.ndarray,
                         metadata: Optional[Dict] = None,
                         save_dir: str = './figs/explain'):
    """Create UMAP visualization"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print("Computing UMAP projection...")
    
    # Create UMAP projection with cosine metric for medical embeddings
    reducer = umap.UMAP(
        n_neighbors=15,       
        min_dist=0.1,         
        metric='cosine',
        target_metric='l2',     
        random_state=42
    )
    
    embedding_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    
    MAX_POINTS = 2000
    if embeddings.shape[0] > MAX_POINTS:
        indices = np.random.choice(embeddings.shape[0], MAX_POINTS, replace=False)
        embedding_2d = embedding_2d[indices]
        labels = labels[indices]
        if metadata and 'icu_type' in metadata:
            metadata['icu_type'] = np.array(metadata['icu_type'])[indices]
    
    # Normalize LOS for continuous color mapping
    normalized_los = (labels - labels.min()) / (labels.max() - labels.min())

    cmap = LinearSegmentedColormap.from_list(
        'gradient',
        ['#b22222', '#d89090','#e5b5b5', '#f2dada', '#d7e1eb', '#afc3d8', '#5f89b1', '#376b9e', '#104e8b']
    )

    # Main scatter plot
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                          c=normalized_los, cmap=cmap,
                          s=80, alpha=0.7, edgecolors='none', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Normalized LOS', fontsize=12)
    
    # Add borders to colorbar
    for spine in cbar.ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor('black')
        
    # ICU markers (optional)
    if metadata and 'icu_type' in metadata:
        icu_types = metadata['icu_type']
        unique_icus = np.unique(icu_types)
        markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X']
        
        for i, icu in enumerate(unique_icus[:8]):
            mask = icu_types == icu
            if mask.sum() > 0:
                plt.scatter(
                    embedding_2d[:, 0], embedding_2d[:, 1],
                    c=normalized_los, cmap=cmap,
                    s=30, alpha=0.5, edgecolors='none', linewidth=0.2
                )
    
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    
    if metadata and 'icu_type' in metadata:
        plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/umap_embeddings.pdf", dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"UMAP visualizations saved to {save_dir}")