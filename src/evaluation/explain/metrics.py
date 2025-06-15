# src/evaluation/explain/metrics.py
"""
Utility metrics for explanation quality
"""
import torch
import numpy as np
from scipy.stats import kendalltau

# --------------------------------------------------
def insertion_curve(model, x, attr, steps=50):
    """
    Compute insertion curve scores – gradually *adding* most important
    features back to zero-masked input.
    """
    keep = torch.argsort(attr.view(-1), descending=True)
    scores = []
    baseline = torch.zeros_like(x)
    for k in torch.linspace(0, len(keep), steps).long():
        mask = torch.zeros_like(attr, dtype=torch.bool)
        mask.view(-1)[keep[:k]] = True
        inp = torch.where(mask, x, baseline)
        with torch.no_grad():
            scores.append(model(inp.unsqueeze(0)).item())
    return np.asarray(scores)

# --------------------------------------------------
def deletion_curve(model, x, attr, steps=50):
    """
    Deletion curve – progressively *removing* important features.
    """
    keep = torch.argsort(attr.view(-1), descending=True)
    scores = []
    for k in torch.linspace(0, len(keep), steps).long():
        mask = torch.ones_like(attr, dtype=torch.bool)
        mask.view(-1)[keep[:k]] = False
        inp = torch.where(mask, x, torch.zeros_like(x))
        with torch.no_grad():
            scores.append(model(inp.unsqueeze(0)).item())
    return np.asarray(scores)

# --------------------------------------------------
def area_under_curve(curve):
    """Simple trapezoidal AUC (higher means better for insertion)."""
    x = np.linspace(0, 1, len(curve))
    return np.trapz(curve, x)

# --------------------------------------------------
def attribution_stability(attr_a, attr_b, k_ratio=0.2):
    """
    Kendall τ on top-k features between two attribution vectors.
    """
    k = int(k_ratio * attr_a.numel())
    idx_a = torch.argsort(attr_a.view(-1), descending=True)[:k]
    idx_b = torch.argsort(attr_b.view(-1), descending=True)[:k]
    # rank union of indices
    union = torch.unique(torch.cat([idx_a, idx_b])).cpu().numpy()
    rank_a = np.full(len(union), fill_value=len(union))
    rank_b = rank_a.copy()
    for r, i in enumerate(idx_a.cpu().numpy()):
        rank_a[np.where(union == i)[0][0]] = r
    for r, i in enumerate(idx_b.cpu().numpy()):
        rank_b[np.where(union == i)[0][0]] = r
    tau, _ = kendalltau(rank_a, rank_b)
    return tau
