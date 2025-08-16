# src/evaluation/ablation/ablate.py

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from copy import deepcopy
import time
from pathlib import Path
import random
import re
import os
from scipy import stats
from src.evaluation.metrics import get_loss_function, get_metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
    cohen_kappa_score
)

# Load ablation output path from paths.json
with open("paths.json") as f:
    _path_config = json.load(f)
ABLATE_RESULT_DIR = Path(_path_config.get("ablate_results", "results/ablations/ablation_analysis"))
ABLATE_RESULT_DIR.mkdir(parents=True, exist_ok=True)

data_dir = _path_config["data_dir"]
flat_info_path = os.path.join(data_dir, "flat_info.json")

with open(flat_info_path, "r") as f:
    flat_info = json.load(f)
    
DEFAULT_FEATURE_GROUPS = {
    "remove_physio": [
        flat_info['columns'].index(c) for c in [
            "gender", "age", "height", "weight", "nullheight"
        ]
    ],
    "remove_vitals": [
        flat_info['columns'].index(c) for c in [
            "eyes", "motor", "verbal"
        ]
    ],
    "remove_ethnicity": [
        i for i, c in enumerate(flat_info['columns']) if c.startswith("ethnicity_")
    ]
}
class StaticOnlyModel(nn.Module):
    """Ablation model that uses only static (flat) features.

    The original GraphGPS and Mamba branches are disabled.  
    We keep the trained flat encoder `flat_fc` (if it exists)  
    but build a brand-new output head that expects exactly 
    `flat_dim` inputs, avoiding any dimension mismatch.
    """
    def __init__(self, original_model):
        super().__init__()

        # Re-use the trained flat feature encoder
        self.flat_fc = getattr(original_model, "flat_fc", None)

        # Determine the dimensionality after flat_fc
        if self.flat_fc is not None:
            if isinstance(self.flat_fc, nn.Sequential):
                # find the last Linear layer inside the Sequential
                linear_layers = [m for m in self.flat_fc if isinstance(m, nn.Linear)]
                if linear_layers:
                    self.flat_dim = linear_layers[-1].out_features
                else:
                    # fallback: use original_model.flat_dim
                    self.flat_dim = original_model.flat_dim
            elif isinstance(self.flat_fc, nn.Linear):
                self.flat_dim = self.flat_fc.out_features
            else:
                self.flat_dim = original_model.flat_dim
        else:
            self.flat_dim = original_model.flat_dim

        # Number of prediction targets remains unchanged
        self.out_dim = original_model.out_dim

        # New head that takes ONLY flat features
        self.out_layer = nn.Sequential(
            nn.Linear(self.flat_dim, self.flat_dim // 2),
            nn.LayerNorm(self.flat_dim // 2),
            nn.GELU(),
            nn.Linear(self.flat_dim // 2, self.out_dim)
        )

    def forward(self, x, flat, adjs, batch_size, edge_weight):
        """Training / validation step.

        Only `flat` is used; other arguments are ignored.
        """
        flat_batch = flat[:batch_size]

        # Pass through the (possibly trained) flat encoder
        flat_out = self.flat_fc(flat_batch) if self.flat_fc else flat_batch

        y = self.out_layer(flat_out)

        # Squeeze to 1-D tensor if the last dimension is singleton
        if y.dim() > 1 and y.size(-1) == 1:
            y = y.squeeze(-1)

        # Return twice to keep (pred, pred_ts) interface
        return y, y

    @torch.no_grad()
    def inference(self, x_all, flat_all, edge_weight,
                  ts_loader, subgraph_loader, device, get_emb=False):
        """Full-graph inference for test / ablation evaluation."""
        flat_out = self.flat_fc(flat_all) if self.flat_fc else flat_all
        y = self.out_layer(flat_out)

        if y.dim() > 1 and y.size(-1) == 1:
            y = y.squeeze(-1)

        return y, y

class NoStaticWrapper(nn.Module):
    """Wrapper that zeros out static features"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x, flat, adjs, batch_size, edge_weight):
        # Zero out flat features
        flat_zero = torch.zeros_like(flat)
        return self.model(x, flat_zero, adjs, batch_size, edge_weight)
        
    def inference(self, x_all, flat_all, edge_weight, ts_loader, subgraph_loader, device, get_emb=False):
        # Zero out flat features for inference
        flat_zero = torch.zeros_like(flat_all)
        return self.model.inference(x_all, flat_zero, edge_weight, ts_loader, subgraph_loader, device, get_emb)


class EdgeDropWrapper(nn.Module):
    """Randomly remove a proportion of edges while keeping the graph undirected
    and reproducible.  Works for both training minibatches and full-graph inference."""
    def __init__(self, model, drop_rate: float = 0.3, seed: int = 42):
        super().__init__()
        self.model = model
        self.drop_rate = drop_rate
        # Single generator shared by all calls – no manual_seed in forward()
        self.gen = torch.Generator(device='cpu').manual_seed(seed)
        self.drop_rate = drop_rate
        self.seed = seed          # logged for reproducibility

    def _make_mask(self, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        num_nodes = int(edge_index.max()) + 1
        undirected_id = torch.minimum(src, dst) * num_nodes + torch.maximum(src, dst)
        uniq_id, inv  = torch.unique(undirected_id, return_inverse=True)
        # 1) sample CPU-side with our seeded Generator
        keep_cpu = torch.rand(uniq_id.numel(), generator=self.gen) > self.drop_rate
        # 2) move mask to same device as edge_index
        keep = keep_cpu.to(edge_index.device)
        # 3) expand back to full edge list
        return keep[inv]

    def forward(self, x, flat, adjs, batch_size, edge_weight):
        new_adjs = []
        new_edge_weight = []
        for idx, (edge_index, e_id, size) in enumerate(adjs):
            keep = self._make_mask(edge_index)
            new_e_id = e_id[keep] if e_id is not None else None
            new_adjs.append((edge_index[:, keep], new_e_id, size))

            if edge_weight is not None and e_id is not None:
                new_edge_weight.append(edge_weight[new_e_id])
        if edge_weight is not None:
            edge_weight = torch.cat(new_edge_weight, dim=0)
        return self.model(x, flat, new_adjs, batch_size, edge_weight)

    def inference(self, x_all, flat_all, edge_weight,
                  ts_loader, subgraph_loader, device, get_emb=False):
        full_ei = subgraph_loader.edge_index.to(device)
        keep = self._make_mask(full_ei)
        sparse_ei = full_ei[:, keep]
        sparse_w = edge_weight[keep] if edge_weight is not None else None


        # Build fresh NeighborSampler (PyG ≥2.5 supports edge_index arg)
        from torch_geometric.loader import NeighborSampler
        sub_sparse = NeighborSampler(
            edge_index=sparse_ei, sizes=[-1],
            batch_size=subgraph_loader.batch_size, shuffle=False)

        return self.model.inference(
            x_all, flat_all, sparse_w,
            ts_loader, sub_sparse, device, get_emb)



class SeqCropWrapper(nn.Module):
    def __init__(self, model, max_len: int = 24, keep_tail: bool = True):
        super().__init__()
        self.model   = model
        self.max_len = max_len
        self.keep_tail = keep_tail
    def _crop(self, seq):
        """Keep the **last** max_len steps by default (more clinical)."""
        if self.keep_tail:
            return seq[:, -self.max_len:, :]
        else:                       # legacy: keep the first window
            return seq[:, :self.max_len, :]
    def forward(self, x, flat, adjs, batch_size, edge_weight):
        return self.model(self._crop(x), flat, adjs, batch_size, edge_weight)
    def inference(self, x_all, flat_all, edge_weight,
                  ts_loader, subgraph_loader, device, get_emb=False):
        class _CropLoader:
            def __iter__(self_inner):
                for (seq, flat, masks), lbl, ids in ts_loader:
                    seq   = seq[:, :self.max_len, :]
                    masks = masks[:, :self.max_len] if masks is not None else None
                    yield (seq, flat, masks), lbl, ids
        return self.model.inference(self._crop(x_all), flat_all,
                                    edge_weight, _CropLoader(),
                                    subgraph_loader, device, get_emb)

class FeatureGroupZeroWrapper(nn.Module):
    """Zero out specified flat feature group in both forward and inference."""
    def __init__(self, model: nn.Module, drop_cols: list):
        super().__init__()
        self.model = model
        self.drop_cols = drop_cols
    def forward(self, x, flat, adjs, batch_size, edge_weight=None):
        flat_mod = flat.clone()
        flat_mod[:, self.drop_cols] = 0.0
        return self.model(x, flat_mod, adjs, batch_size, edge_weight)
    def inference(self, x_all, flat_all, edge_weight, ts_loader, subgraph_loader, device, get_emb=False):
        flat_mod = flat_all.clone()
        flat_mod[:, self.drop_cols] = 0.0
        return self.model.inference(x_all, flat_mod, edge_weight, ts_loader, subgraph_loader, device, get_emb)

     
def apply_ablation(batch, model, config, ablation_type: str):
    m = re.match(r'drop_edges_(\d+)', ablation_type)
    if m:
        rate = int(m.group(1)) / 100.0
        return batch, EdgeDropWrapper(model, drop_rate=rate)

    if ablation_type == 'baseline':
        return batch, model
    # Time-window variants
    if ablation_type == 'last6h':
        return batch, SeqCropWrapper(model, max_len=6)
    if ablation_type == 'last24h':
        return batch, SeqCropWrapper(model, max_len=24)
    if ablation_type == 'full48h':
        return batch, SeqCropWrapper(model, max_len=48)
    # Static-only or no-static
    if ablation_type == 'static_only':
        return batch, StaticOnlyModel(model)
    if ablation_type == 'no_static':
        return batch, NoStaticWrapper(model)
    # Feature-group removals (indices must be provided in config)
    if ablation_type in {'remove_physio', 'remove_vitals', 'remove_ethnicity'}:
        feature_groups = config.get('feature_groups', DEFAULT_FEATURE_GROUPS)
        cols = feature_groups.get(ablation_type, [])
        return batch, FeatureGroupZeroWrapper(model, cols)

    raise ValueError(f"Unknown ablation: {ablation_type}")

DEFAULT_ABLATIONS = [
    'baseline',
    'last6h', 'last24h', 'full48h',
    'remove_physio', 'remove_vitals', 'remove_ethnicity',
    'static_only', 'no_static',
    'drop_edges_30','drop_edges_50','drop_edges_70'
]

def finetune(model, loader, device, epochs=3, lr=1e-4):
    """Fine-tune the new heads of StaticOnly / NoStatic wrappers.
    Handles loaders that yield (inputs, labels, ids)  OR  (inputs, labels)."""

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        print("[INFO] Nothing to fine-tune (all params frozen).")
        return

    model.train()
    optim    = torch.optim.Adam(params, lr=lr)
    loss_fn  = torch.nn.MSELoss()

    for _ in range(epochs):
        for batch in loader:
            # --- Parse mini-batch ------------------------------------------
            if not isinstance(batch, (list, tuple)):
                print("[INFO] Loader did not return a tuple – skip fine-tune.")
                return

            if len(batch) == 3:          # (inputs, labels, ids)
                inputs, labels, _ = batch
            elif len(batch) == 2:        # (inputs, labels)
                inputs, labels     = batch
            else:
                raise RuntimeError("Unexpected batch structure")

            # Unpack seq / flat
            if isinstance(inputs, (list, tuple)):
                seq  = inputs[0].to(device)
                flat = inputs[1].to(device)
            else:  # fallback – not expected for LSTM loaders
                continue

            labels = labels.to(device).float()

            preds, _ = model(
                seq, flat,
                adjs=[(torch.tensor([[0],[0]], device=device), None, None)],
                batch_size=seq.size(0),
                edge_weight=None
            )

            loss = loss_fn(preds.squeeze(), labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
            
METRICS = ['kappa', 'mad', 'mape', 'mse', 'msle', 'r2']


def set_global_seed(seed: int):
    """Set global random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def run_ablation_experiment(model, loader, subgraph_loader, ts_loader,
                           dataset, config, ablation_type: str,
                           device='cuda') -> Dict[str, Any]:
    """Run a single ablation and return metrics dict (does not save files)."""
    # Apply ablation
    if ablation_type == 'baseline':
        ablated = model
    else:
        ablated = deepcopy(model)
        _, ablated = apply_ablation(None, ablated, config, ablation_type)
    ablated.to(device).eval()

    # Finetune static-only/no-static heads
    if ablation_type in {'static_only', 'no_static'}:
        finetune(ablated, loader, device, epochs=5, lr=1e-4)
        ablated.eval()

    # Inference
    start = time.time()
    with torch.no_grad():
        x = dataset.data.x.to(device)
        flat = dataset.data.flat.to(device)
        edge_weight = getattr(dataset.data, 'edge_attr', None)
        edge_weight = edge_weight.to(device) if edge_weight is not None else None

        if hasattr(ablated, 'inference'):
            out, out_ts = ablated.inference(x, flat, edge_weight,
                                           ts_loader, subgraph_loader, device)
        else:
            # fallback forward
            preds = []
            batch_size = 512
            for i in range(0, x.size(0), batch_size):
                end = min(i + batch_size, x.size(0))
                p, _ = ablated(x[i:end], flat[i:end], [(torch.tensor([[0],[0]], device=device), None, None)], end-i, edge_weight)
                preds.append(p)
            out = torch.cat(preds, dim=0)
            out_ts = out

        mask = getattr(dataset.data, 'test_mask', torch.ones_like(dataset.data.y, dtype=torch.bool))
        truth = dataset.data.y[mask]
        pred = out[mask]
        if config.get('log1p_label', True):
            truth = torch.expm1(truth)
            pred  = torch.expm1(pred)

        truth_np = torch.relu(truth).cpu().numpy()
        pred_np  = torch.relu(pred).cpu().numpy()

    # Compute metrics
    kappa = cohen_kappa_score(
        np.digitize(truth_np, bins=[1,2,3]),
        np.digitize(pred_np, bins=[1,2,3]),
        weights='quadratic'
    )
    results = get_metrics(truth_np, pred_np, verbose=False, is_cls=False)
    results['time'] = time.time() - start
    return results


def run_all_ablations(model, loader, subgraph_loader, ts_loader,
                                 dataset, config, ablations: Optional[List[str]] = None,
                                 seeds: List[int] = [21, 22, 23],
                                 device: str = 'cuda',
                                 save_dir: Path = ABLATE_RESULT_DIR) -> None:
    """Run each ablation over multiple seeds, compute statistics and significance."""
    if ablations is None:
        ablations = DEFAULT_ABLATIONS
    all_results = {abl: {m: [] for m in METRICS} for abl in ablations}

    # Run experiments
    for seed in seeds:
        set_global_seed(seed)
        for abl in ablations:
            res = run_ablation_experiment(
                model, loader, subgraph_loader, ts_loader,
                dataset, config, abl, device
            )
            for m in METRICS:
                all_results[abl][m].append(float(res[m]))

    # Summarize: mean, std
    summary = {}
    for abl in ablations:
        summary[abl] = {}
        for m in METRICS:
            vals = all_results[abl][m]
            summary[abl][f'{m}_mean'] = float(np.mean(vals))
            summary[abl][f'{m}_std']  = float(np.std(vals, ddof=1))

    # Save summary
    out_path = save_dir / 'ablations_summary_multi_seed.json'
    with open(out_path, 'w') as f:
        json.dump({
            'summary': summary,
            'raw': all_results,
            'seeds': seeds
        }, f, indent=2)
    print(f"Saved multi-seed summary to {out_path}")
    
    print("="*40)
    print("Ablation results by seed:")
    for abl in ablations:
        print(f"\nAblation: {abl}")
        for i, seed in enumerate(seeds):
            metric_str = ", ".join(
                [f"{m}: {all_results[abl][m][i]:.4f}" for m in METRICS]
            )
            print(f"  Seed {seed}: {metric_str}")

    print("\nAblation summary (mean ± std):")
    for abl in ablations:
        metric_str = ", ".join(
            [f"{m}: {summary[abl][f'{m}_mean']:.4f} ± {summary[abl][f'{m}_std']:.4f}" for m in METRICS]
        )
        print(f"  {abl}: {metric_str}")
    print("="*40)
    
    import matplotlib.pyplot as plt

    def plot_bar_with_error(summary, ablations, metric, save_path=None):
        means = [summary[abl][f'{metric}_mean'] for abl in ablations]
        stds = [summary[abl][f'{metric}_std'] for abl in ablations]
    
        plt.figure(figsize=(12, 6))
        plt.bar(ablations, means, yerr=stds, capsize=4, alpha=0.7)
        plt.ylabel(f"{metric} (mean ± std)")
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Ablation study: {metric}")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
    
    plot_bar_with_error(summary, ablations, 'kappa', save_path=save_dir / 'ablation_kappa.png')
    
    plot_bar_with_error(summary, ablations, 'mse', save_path=save_dir / 'ablation_mse.png')

    def plot_boxplot_all(all_results, ablations, metric, seeds, save_path=None):
        data = [all_results[abl][metric] for abl in ablations]
        plt.figure(figsize=(12, 6))
        plt.boxplot(data, labels=ablations, showmeans=True)
        plt.ylabel(f"{metric}")
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Ablation study (boxplot): {metric} across seeds")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    plot_boxplot_all(all_results, ablations, 'mse', seeds, save_path=save_dir / 'ablation_mse_boxplot.png')


