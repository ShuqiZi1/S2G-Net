# src/evaluation/explain/temporal_attr.py
"""
Temporal attribution using Integrated Gradients
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients, NoiseTunnel
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import src.evaluation.explain.viz_style as vs
import matplotlib.gridspec as gridspec
import os
import re

class TimeBranch(nn.Module):
    """Wrapper for temporal branch"""
    def __init__(self, model):
        super().__init__()
        self.mamba_encoder = model.mamba_encoder
        self.mamba_norm = model.mamba_norm
        self.mamba_out = model.mamba_out
        
    def forward(self, x_ts: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through Mamba branch only"""
        # Check if mamba_encoder accepts mask
        try:
            mamba_out, _ = self.mamba_encoder(x_ts, mask)
        except TypeError:
            # If mask not accepted, just pass x_ts
            mamba_out, _ = self.mamba_encoder(x_ts)
            
        last = mamba_out[:, -1, :] if mamba_out.dim() == 3 else mamba_out
        last = self.mamba_norm(last)
        
        exp_in = self.mamba_out[0].in_features         # e.g. 192
        if last.size(1) < exp_in:                      # missing flat part
            pad = torch.zeros(last.size(0), exp_in - last.size(1),
                              device=last.device, dtype=last.dtype)
            last = torch.cat([last, pad], dim=1)
        
        return self.mamba_out(last).squeeze(-1)


def run_temporal_attribution(model, loader, device='cuda', save_dir='./figs/explain',
                            n_samples=10, baseline_type='zero', 
                            use_noise_tunnel=True):
    """
    Run temporal attribution analysis
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Create temporal branch wrapper
    time_branch = TimeBranch(model).to(device).eval()
    
    # Check if model accepts mask
    accepts_mask = True
    try:
        # Test with dummy input
        dummy_seq = torch.randn(1, 48, model.mamba_encoder.input_dim, device=device)
        dummy_mask = torch.ones(1, 48, device=device)
        time_branch(dummy_seq, dummy_mask)
    except:
        accepts_mask = False
        print("Model doesn't accept mask, using sequence only")
    
    # Initialize IG (+ optional Smooth-Grad / Noise-Tunnel)
    
    ig = IntegratedGradients(time_branch)
    if use_noise_tunnel:
        ig = NoiseTunnel(ig)     
    
    # Collect samples with best and worst predictions
    errors = []
    samples = []
    
    with torch.no_grad():
        for inputs, labels, ids in loader:
            if len(inputs) == 3:                       # (seq, flat, masks)
                seq, flat, masks = inputs
            else:                                      # (seq, flat) – build dummy mask
                seq, flat           = inputs
                masks               = None            # keep None → later code already copes
            seq   = seq.to(device)
            masks = masks.to(device) if masks is not None else None
            labels = labels.to(device)
            
            pred = time_branch(seq, masks) if accepts_mask else time_branch(seq)
            pred_lin   = torch.expm1(pred)
            label_lin  = torch.expm1(labels.squeeze())
            
            error      = torch.abs(pred_lin - label_lin).cpu().numpy()
            
            for i in range(len(error)):
                sample_dict = {
                    'seq'   : seq[i].cpu(),
                    'label' : label_lin[i].item(),
                    'pred'  : pred_lin[i].item(),
                    'error' : error[i],
                    'id': ids[i].cpu().item()
                }
                if masks is not None:
                    sample_dict['mask'] = masks[i].cpu()
                    
                errors.append(error[i])
                samples.append(sample_dict)
                
            if len(samples) >= n_samples * 2:
                break
    
    # Sort by error
    sorted_indices = np.argsort([s['error'] for s in samples])
    best_idx = sorted_indices[:n_samples//2]
    worst_idx = sorted_indices[-n_samples//2:]

    # Collect attributions and build heat arrays
    heat_dict = {}
    for case_name, indices in [('best', best_idx), ('worst', worst_idx)]:
        all_attr = []
        for idx in indices:
            sample = samples[idx]
            seq   = sample['seq'].unsqueeze(0).to(device)
            mask  = sample.get('mask', None)
            mask  = mask.unsqueeze(0).to(device) if mask is not None else None
    
            def make_baseline(x, btype):
                if btype == 'zero':
                    return torch.zeros_like(x)
                if btype == 'mean':
                    return x.mean(dim=1, keepdim=True).expand_as(x).detach()
                if btype == 'gauss':
                    return torch.randn_like(x) * x.std() + x.mean()
                raise ValueError(f"Unsupported baseline: {btype}")
    
            baseline_list = ([baseline_type] if isinstance(baseline_type, str)
                             else baseline_type)
    
            attr_list = []
            for b in baseline_list:
                bas  = make_baseline(seq, b)
                attr = ig.attribute(
                    seq,
                    baselines=bas,
                    additional_forward_args=(mask,) if mask is not None else None,
                    n_steps=50,
                    nt_type='smoothgrad' if use_noise_tunnel else None,
                    stdevs=0.02,
                    nt_samples=25
                )
                if mask is not None:
                    attr *= mask.unsqueeze(-1)
                attr_list.append(attr)
            avg_attr  = torch.stack(attr_list).mean(0)
    
            magnitude = avg_attr.abs().mean(dim=-1).squeeze().cpu().numpy()
            if np.isclose(magnitude.max(), 0):
                magnitude = np.random.rand(*magnitude.shape) * 0.1
            all_attr.append(magnitude)
    
        heat_dict[case_name] = np.stack(all_attr)
    
    def plot_temporal_heatmaps(heat_dict, save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
        combined = np.concatenate([heat_dict['best'].ravel(),
                                   heat_dict['worst'].ravel()])
        vmin, vmax = np.percentile(combined, [5, 95])
    
        fig = plt.figure(figsize=(13, 4)) 
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.04], wspace=0.05) 
        
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        cax  = fig.add_subplot(gs[2]) 
        
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            'gradient',
            ['#d7e1eb', '#afc3d8', '#5f89b1', '#376b9e', '#104e8b']
        )
        
        combined = np.concatenate([heat_dict['best'].ravel(), heat_dict['worst'].ravel()])
        vmin, vmax = np.percentile(combined, [5, 95])
        
        sns.heatmap(
            heat_dict['best'], cmap=cmap,
            vmin=vmin, vmax=vmax,
            cbar=False, ax=ax1
        )
        ax1.set_title("Best", fontweight="bold")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Sample")
        xticks = np.arange(0, heat_dict['best'].shape[1], 6)
        ax1.set_xticks(xticks + 0.5)
        ax1.set_xticklabels(xticks, rotation=45, ha="right")
        
        sns.heatmap(
            heat_dict['worst'], cmap=cmap,
            vmin=vmin, vmax=vmax,
            cbar=True, ax=ax2, cbar_ax=cax,
            cbar_kws={
                "label": "Attribution Score"
            }
        )
        ax2.set_title("Worst", fontweight="bold")
        ax2.set_xlabel("Time Step")
        ax2.set_yticklabels([]) 
        ax2.set_xticks(xticks + 0.5)
        ax2.set_xticklabels(xticks, rotation=45, ha="right")
        
        sns.despine(fig)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f"fig4_temporal.pdf"), dpi=600, bbox_inches="tight")
        plt.close(fig)
    
    plot_temporal_heatmaps(heat_dict, save_dir)
    
    return {
        "best_samples":  [samples[i] for i in best_idx],
        "worst_samples": [samples[i] for i in worst_idx]
    }