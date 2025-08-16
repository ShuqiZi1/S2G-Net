"""
Extended PyG data readers for GraphGPS model
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import warnings
from typing import Tuple, List, Optional, Dict, Any
from src.dataloader.ts_reader import collect_ts_flat_labels


def read_txt(path, node=True):
    """Read raw txt file into lists"""
    with open(path, "r") as f:
        content = f.read()
    if node:
        return [int(n) for n in content.split('\n') if n != '']
    else:
        return [float(n) for n in content.split('\n') if n != '']


def _sample_mask(idx, l):
    """Create sample mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask


def get_edge_index(us, vs, scores=None, timestamps=None):
    """
    Return edge data according to pytorch-geometric's specified formats.
    Optionally includes timestamp information for dynamic graphs.
    """
    both_us = np.concatenate([us, vs])  # both directions
    both_vs = np.concatenate([vs, us])  # both directions
    edge = np.stack([both_us, both_vs], 0)
    edge_index = torch.tensor(edge, dtype=torch.long)

    # Process edge attributes
    if scores is None:
        num_edges = edge_index.shape[1]
        scores = np.random.rand(num_edges, 1)
    else:
        scores = np.concatenate([scores, scores])[:, None]

    edge_attr = torch.tensor(scores).float()

    # Process timestamps if provided
    edge_time = None
    if timestamps is not None:
        timestamps = np.concatenate([timestamps, timestamps])
        edge_time = torch.tensor(timestamps).float()

    return edge_index, edge_attr, edge_time


def define_node_masks(N, train_n, val_n):
    """Define node masks according to train / val / test split"""
    idx_train = range(train_n)
    idx_val = range(train_n, train_n + val_n)
    idx_test = range(train_n + val_n, N)
    train_mask = torch.BoolTensor(_sample_mask(idx_train, N))
    val_mask = torch.BoolTensor(_sample_mask(idx_val, N))
    test_mask = torch.BoolTensor(_sample_mask(idx_test, N))
    return train_mask, val_mask, test_mask, idx_train, idx_val, idx_test


def read_graph_edge_list(graph_dir, version, with_timestamps=False, with_types=True):
    """
    Extended version of read_graph_edge_list that also reads edge types
    
    Args:
        graph_dir: Directory containing graph files
        version: Graph version/type (e.g., 'gps_k10_10_')
        with_timestamps: Whether to read timestamps
        with_types: Whether to read edge types
        
    Returns:
        u_list, v_list: Lists of source and target nodes
        scores: Edge scores/weights
        timestamps: Edge timestamps (optional)
        types: Edge types (optional)
    """
    from pathlib import Path
    
    # If version doesn't end with '_', add it
    if not version.endswith('_'):
        version = version + '_'
    
    u_path = Path(graph_dir) / f"{version}u.txt"
    v_path = Path(graph_dir) / f"{version}v.txt"
    scores_path = Path(graph_dir) / f"{version}scores.txt"
    types_path = Path(graph_dir) / f"{version}types.txt"
    
    # Read node indices
    u_list = read_txt(u_path)
    v_list = read_txt(v_path)
    
    # Read scores if available
    scores = None
    if os.path.exists(scores_path):
        scores = read_txt(scores_path, node=False)
    
    # Read edge types if available
    types = None
    if with_types and os.path.exists(types_path):
        types = read_txt(types_path, node=False)
    
    # Read timestamps if requested
    timestamps = None
    if with_timestamps:
        time_path = Path(graph_dir) / f"{version}timestamps.txt"
        if os.path.exists(time_path):
            timestamps = read_txt(time_path, node=False)
        else:
            # Generate default timestamps
            timestamps = np.linspace(0, 1, len(u_list))
    
    return u_list, v_list, scores, timestamps, types


# Placeholder for get_class_weights function
def get_class_weights(labels):
    """
    Placeholder for the function that computes class weights
    
    In your actual implementation, you would import this from your existing code or
    implement it here.
    """
    warnings.warn("Using placeholder get_class_weights function. "
                 "Replace with your actual implementation.")
    
    # Simple placeholder implementation
    if len(np.unique(labels)) <= 10:  # Classification
        counts = np.bincount(labels.astype(int))
        weights = len(labels) / (len(counts) * counts)
        return torch.tensor(weights, dtype=torch.float)
    else:
        return False


class GraphDataset(Dataset):
    """
    Extended Dataset class for GraphGPS model with edge type support
    """
    def __init__(self, config):
        super().__init__()
        data_dir = config['data_dir']
        task     = config['task']
        use_time = config.get('use_time_encoding', False)
        with_types = config.get('with_edge_types', True)

        # ---- 1. Load time series, flat features, labels ----
        seq, flat, labels, info, N, train_n, val_n = collect_ts_flat_labels(
            data_dir, config['ts_mask'], task, config['add_diag'],
            debug=config.get('debug', 0),
            use_time_encoding=use_time
        )

        # Store info
        self.info = info

        # true T and feature dimension
        T = seq.shape[1] if seq.ndim == 3 else 1
        d_ts = seq.shape[-1]

        # ---- 2. Construct node features ----
        x_hist = torch.from_numpy(seq.astype(np.float32))  # [N, T, d_ts]
        x_last = x_hist[:, -1, :]  # [N, d_ts]

        if config.get('flatten_ts', False):
            x_flatten = seq.reshape(N, -1)
            x = torch.from_numpy(x_flatten.astype(np.float32))
        else:
            x = x_hist

        # ---- 3. Flatten plane features ----
        flat = torch.from_numpy(flat.astype(np.float32)) if flat.size else None

        # ---- 4. Labels ----
        y = torch.from_numpy(labels.astype(np.float32))

        # ---- 5. Edges with types ----
        if config.get('random_g', False):
            rand_u = np.random.randint(0, N, size=N * 2)
            rand_v = np.random.randint(0, N, size=N * 2)
            out = get_edge_index(rand_u, rand_v)
            edge_index, edge_attr = out[:2]
            edge_types = None
        else:
            us, vs, scores, _, types = read_graph_edge_list(
                config['graph_dir'], config['g_version'], 
                with_timestamps=False, with_types=with_types
            )
            
            # Process edge attributes
            if with_types and types is not None:
                # Convert types to one-hot encoding
                unique_types = np.unique(types)
                n_types = len(unique_types)
                
                # Create edge_attr with scores and one-hot types
                edge_attr_list = []
                
                # Add scores if available
                if scores is not None:
                    edge_attr_list.append(np.asarray(scores).reshape(-1, 1))
                
                # Add one-hot types
                type_onehot = np.zeros((len(types), n_types))
                for i, t in enumerate(unique_types):
                    type_onehot[types == t, i] = 1
                
                edge_attr_list.append(type_onehot)
                edge_attr = torch.from_numpy(
                    np.hstack(edge_attr_list).astype(np.float32)
                ).float()
                
                # Record edge dimension for the model
                self.edge_dim = edge_attr.shape[1]
                config['edge_dim'] = self.edge_dim
            else:
                # Use just scores as edge attributes
                out = get_edge_index(us, vs, scores)
                _, edge_attr = out[:2]
                self.edge_dim = 1
                config['edge_dim'] = 1
            
            # Build bi-directional edges **without** destroying multi-dim edge_attr
            # Forward+reverse: [u->v] ∪ [v->u]
            us = np.asarray(us)
            vs = np.asarray(vs)
            both_us = np.concatenate([us, vs])
            both_vs = np.concatenate([vs, us])
            edge_index = torch.tensor(
                np.stack([both_us, both_vs], axis=0), dtype=torch.long
            )  # [2, 2E]

            if with_types and types is not None:
                # edge_attr (E, F) already built as [score | one-hot(type)]
                # duplicate for reverse edges → (2E, F)
                if not isinstance(edge_attr, torch.Tensor):
                    edge_attr = torch.from_numpy(edge_attr).float()
                edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
                self.edge_dim = int(edge_attr.size(1))
            else:
                # fall back to 1-D scores; still make it bi-directional
                if scores is None:
                    scores = np.ones(len(us), dtype=np.float32)
                scores_2 = np.concatenate([scores, scores])[:, None]  # (2E, 1)
                edge_attr = torch.from_numpy(scores_2.astype(np.float32))
                self.edge_dim = 1
            config['edge_dim'] = self.edge_dim

        # ---- 6. Batch vector ----
        batch = torch.zeros(N, dtype=torch.long)   # Single graph -> all zeros

        # ---- 7. Build Data object ----
        data = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            flat=flat, batch=batch, y=y
        )
        
        if edge_attr is not None:
            w = edge_attr if edge_attr.dim() == 1 else edge_attr[:, 0]
            w = w.clone().detach()

            w = w - w.min()
            if w.max() > 0:
                w = w / w.max()

            N = int(edge_index.max()) + 1 
            attn_bias = torch.zeros((N, N), dtype=torch.float32)

            src, dst = edge_index
            bias_val = (w + 1e-6).log()

            attn_bias[src, dst] = bias_val
            attn_bias[dst, src] = bias_val

            data.attn_bias = attn_bias

        data.ts_seq = x_hist
        data.edge_tuple = (edge_index, edge_attr)

        # ---- 8. Training / validation / test masks ----
        data.train_mask, data.val_mask, data.test_mask, \
            self.idx_train, self.idx_val, self.idx_test = define_node_masks(N, train_n, val_n)

        # Class weights (for classification tasks)
        self.class_weights = get_class_weights(labels[:train_n]) if task == 'ihm' else None
        self.data = data
        
        self.seq_full    = torch.from_numpy(seq.astype(np.float32))   # [N, T, d_ts]
        self.edge_index  = edge_index
        self.edge_attr   = edge_attr
        self.N           = self.seq_full.size(0)

        
        # Set dimension attributes
        self.x_dim = self.data.x.size(-1) if self.data.x is not None else 0
        self.flat_dim = self.data.flat.size(-1) if self.data.flat is not None else 0
        
        print(f"GraphDataset initialized with {N} nodes, "
              f"{edge_index.shape[1]} edges, {self.edge_dim} edge features")

    def __len__(self):
        """
        Return dataset length - correctly implemented for different training modes
        
        For node-level tasks with full-graph training, return number of nodes
        For graph-level tasks or neighbor sampling, return 1
        """
        return self.data.num_graphs if getattr(self, "graph_level", False) else 1

    def __getitem__(self, idx):
        """
        Get dataset item with support for both regular batching and neighbor sampling.
        
        Args:
            idx: Index or tuple with neighborhood sampling info
            
        Returns:
            Data tuple for model processing
        """
        if isinstance(idx, tuple) and len(idx) == 3:
            return idx     # (batch_size, n_id, adjs)
    
        return (
            self.seq_full,
            self.data.flat, 
            self.edge_index,
            self.N, 
            self.edge_attr,
        )


class NeighborSamplerWrapper:
    """
    Wrapper around NeighborSampler to adapt its output for GraphGPSModel
    """
    def __init__(self, dataset, neighbor_sampler):
        self.dataset = dataset
        self.neighbor_sampler = neighbor_sampler
        
    def __iter__(self):
        for batch_size, n_id, adjs in self.neighbor_sampler:
            yield (batch_size, n_id, adjs)
            
    def __len__(self):
        return len(self.neighbor_sampler)