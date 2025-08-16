"""
dataloaders to work with graphs
"""
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from src.dataloader.ts_reader import collect_ts_flat_labels, get_class_weights


def _sample_mask(idx, l):
    """Create sample mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask


def get_edge_index(us, vs, scores=None, timestamps=None):
    """
    return edge data according to pytorch-geometric's specified formats.
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


def get_rdm_edge_index(N, factor=2, with_timestamps=False):
    """
    return random edge data, optionally with timestamps for dynamic graphs
    """
    n_edge = N * factor
    us = np.random.choice(range(N), n_edge)
    vs = np.random.choice(range(N), n_edge)

    if with_timestamps:
        timestamps = np.random.rand(n_edge)
        return get_edge_index(us, vs, timestamps=timestamps)
    else:
        return get_edge_index(us, vs)


def define_node_masks(N, train_n, val_n):
    """
    define node masks according to train / val / test split
    """
    idx_train = range(train_n)
    idx_val = range(train_n, train_n + val_n)
    idx_test = range(train_n + val_n, N)
    train_mask = torch.BoolTensor(_sample_mask(idx_train, N))
    val_mask = torch.BoolTensor(_sample_mask(idx_val, N))
    test_mask = torch.BoolTensor(_sample_mask(idx_test, N))
    return train_mask, val_mask, test_mask, idx_train, idx_val, idx_test


def read_txt(path, node=True):
    """
    read raw txt file into lists
    """
    u = open(path, "r")
    u_list = u.read()
    if node:
        return [int(n) for n in u_list.split('\n') if n != '']
    else:
        return [float(n) for n in u_list.split('\n') if n != '']


def read_graph_edge_list(graph_dir, version, with_timestamps=False):
    """
    return edge lists, edge similarity scores, and optionally timestamps from specified graph.
    """
    version2filename = {
        'default': 'k_closest_{}_k=3_adjusted_ns.txt',
        'dynamic': 'dynamic_{}_k=3_adjusted_ns.txt'
    }

    file_name = version2filename.get(version, version2filename['default'])
    u_path = Path(graph_dir) / file_name.format('u')
    v_path = Path(graph_dir) / file_name.format('v')
    scores_path = Path(graph_dir) / file_name.format('scores')

    u_list = read_txt(u_path)
    v_list = read_txt(v_path)

    scores = None
    if os.path.exists(scores_path):
        scores = read_txt(scores_path, node=False)

    timestamps = None
    if with_timestamps:
        time_path = Path(graph_dir) / file_name.format('timestamps')
        if os.path.exists(time_path):
            timestamps = read_txt(time_path, node=False)
        else:
            timestamps = np.linspace(0, 1, len(u_list))

    return u_list, v_list, scores, timestamps


class GraphDataset(Dataset):
    """
    Dataset class for graph data, extended to support dynamic graphs with time information
    """
    def __init__(self, config, us=None, vs=None):
        super().__init__()

        data_dir = config['data_dir']
        task = config['task']
        use_time_encoding = config.get('use_time_encoding', False)
        dynamic_graph = config.get('dynamic_g', False)

        # Get node features
        seq, flat, labels, info, N, train_n, val_n = collect_ts_flat_labels(
            data_dir, config['ts_mask'], task, config['add_diag'],
            debug=config.get('debug', 0), use_time_encoding=use_time_encoding)

        self.info = info

        if 'total' in info and N != info['total']:
            print(f"Adjusting N from {N} to {info['total']} to match actual data")
            N = info['total']

        if (train_n == 0 or val_n == 0) and 'train_len' in info and 'val_len' in info:
            train_n = info['train_len']
            val_n = info['val_len']
            if config.get('verbose', False):
                print(f"[INFO] Using train/val split from info.json: {train_n} train, {val_n} val")

        # Get the edges
        edge_time = None
        if config['model'] == 'lstm':
            edge_index, edge_attr, _ = get_rdm_edge_index(N, 1)
        else:
            if config['random_g']:
                edge_index, edge_attr, edge_time = get_rdm_edge_index(
                    N, factor=2, with_timestamps=dynamic_graph)
            else:
                us, vs, scores, timestamps = read_graph_edge_list(
                    config['graph_dir'], config['g_version'], with_timestamps=dynamic_graph)
                edge_index, edge_attr, edge_time = get_edge_index(us, vs, scores, timestamps)

        present_src = set(edge_index[0].tolist())
        present_dst = set(edge_index[1].tolist())
        present = present_src.union(present_dst)
        missing = set(range(N)).difference(present)
        if missing:
            loops = torch.tensor(list(missing), dtype=torch.long, device=edge_index.device)
            self_edges = torch.stack([loops, loops], dim=0)
            edge_index = torch.cat([edge_index, self_edges], dim=1)
            if edge_attr is not None:
                loop_attr = torch.ones(len(missing), 1, device=edge_attr.device)
                edge_attr = torch.cat([edge_attr, loop_attr], dim=0)
        # ————————————————————————————————

        # Record feature dimensions
        self.ts_dim = seq.shape[1] if config.get('read_lstm_emb', False) else seq.shape[2]
        self.flat_dim = flat.shape[1]
        x = seq
        if config.get('flatten', False):
            x = np.reshape(x, (len(x), -1))
            if config.get('flat_first', False) and config.get('add_diag', False):
                x = np.concatenate([x, flat], 1)
            self.x_dim = x.shape[1]
        else:
            self.x_dim = self.ts_dim

        if config.get('verbose', False):
            print(f'Dimensions of ts: {self.ts_dim}, flat features: {self.flat_dim}, x: {self.x_dim}')

        x = x[:N]
        flat = flat[:N]
        labels = labels[:N]

        # define the graph and its features
        x = torch.from_numpy(x).float()
        flat = torch.from_numpy(flat).float()
        y = torch.from_numpy(labels)
        y = y.long() if task == 'ihm' else y.float()

        # Create data object, 包含补完自环后的 edge_index / edge_attr
        data = Data(x=x, edge_index=edge_index, y=y, flat=flat, edge_attr=edge_attr)
        if edge_time is not None:
            data.edge_time = edge_time

        # define masks
        data.train_mask, data.val_mask, data.test_mask, \
            self.idx_train, self.idx_val, self.idx_test = define_node_masks(N, train_n, val_n)
        self.data = data

        # define class weights
        self.class_weights = get_class_weights(labels[:train_n]) if task == 'ihm' else False

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if hasattr(self.data, 'edge_time'):
            return (self.data.x, self.data.flat,
                    self.data.edge_index, self.data.edge_attr,
                    self.data.edge_time, self.data.y)
        else:
            return (self.data.x, self.data.flat,
                    self.data.edge_index, self.data.edge_attr,
                    self.data.y)

def read_graph_dir(graph_dir, version):
    """
    Read graph data from directory
    
    Args:
        graph_dir: Directory containing graph files
        version: Graph version/type
        
    Returns:
        u, v: Source and target node lists
        scores: Edge weights/scores
    """
    pattern = 'k_closest_{}_k=3_adjusted_ns.txt' if version == 'default' else version
    u = read_txt(Path(graph_dir) / pattern.format('u'))
    v = read_txt(Path(graph_dir) / pattern.format('v'))
    s_path = Path(graph_dir) / pattern.format('scores')
    scores = read_txt(s_path, node=False) if s_path.exists() else None
    return u, v, scores

def define_masks(N, train_n, val_n):
    """
    Define boolean masks for training, validation and testing
    
    Args:
        N: Total number of nodes
        train_n: Number of training nodes
        val_n: Number of validation nodes
        
    Returns:
        train_m, val_m, test_m: Boolean masks for each set
    """
    idx_train = range(train_n)
    idx_val   = range(train_n, train_n + val_n)
    idx_test  = range(train_n + val_n, N)
    train_m = torch.as_tensor(_sample_mask(idx_train, N))
    val_m   = torch.as_tensor(_sample_mask(idx_val, N))
    test_m  = torch.as_tensor(_sample_mask(idx_test, N))
    return train_m, val_m, test_m
    
    
class GraphGPSDataset(Dataset):
    """
    Dataset class for GraphGPS model. Returns ONE Data object with attributes:
      x, x_hist, flat, edge_index, edge_attr, batch, y,
      and boolean masks train_mask / val_mask / test_mask.
    """
    def __init__(self, config):
        super().__init__()
        data_dir = config['data_dir']
        task     = config['task']
        use_time = config.get('use_time_encoding', False)

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

        # ---- 2. Construct node single-step features x and time series x_hist ----
        x_hist = torch.from_numpy(seq.astype(np.float32))          # [N, T, d_ts]
        # Use the last time step as default node feature
        x_last = x_hist[:, -1, :]                                  # [N, d_ts]

        if config.get('flatten_ts', False):
            x_flatten = seq.reshape(N, -1)
            x = torch.from_numpy(x_flatten.astype(np.float32))
        else:
            x = x_last

        # ---- 3. Flatten plane features ----
        flat = torch.from_numpy(flat.astype(np.float32)) if flat.size else None

        # ---- 4. Labels ----
        y = torch.from_numpy(labels.astype(np.float32))
        
        # ---- 5. Edges ----
        if config['random_g']:
            rand_u = np.random.randint(0, N, size=N * 2)
            rand_v = np.random.randint(0, N, size=N * 2)
            out = get_edge_index(rand_u, rand_v)
            edge_index, edge_attr = out[:2]
        else:
            us, vs, scores = read_graph_dir(config['graph_dir'], config['g_version'])
            out = get_edge_index(us, vs, scores)
            edge_index, edge_attr = out[:2]

        # ---- 6. Batch vector ----
        batch = torch.zeros(N, dtype=torch.long)   # Single graph -> all zeros

        # ---- 7. Build Data object ----
        data = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            x_hist=x_hist, flat=flat, batch=batch, y=y
        )

        # ---- 8. Training / validation / test masks ----
        data.train_mask, data.val_mask, data.test_mask = define_masks(N, train_n, val_n)

        # Record indices for later use
        self.idx_train = range(train_n)
        self.idx_val = range(train_n, train_n + val_n)
        self.idx_test = range(train_n + val_n, N)

        # Class weights (for classification tasks)
        self.class_weights = get_class_weights(labels[:train_n]) if task == 'ihm' else None
        self.data = data
        
        # Set dimension attributes
        self.x_dim = self.data.x.size(-1) if self.data.x is not None else 0
        self.flat_dim = self.data.flat.size(-1) if self.data.flat is not None else 0

    # ---------- PyTorch style ----------
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        """
        Get dataset item with support for both regular batching and neighbor sampling.
        
        Args:
            idx: Index or tuple with neighborhood sampling info
            
        Returns:
            Data tuple for model processing
        """
        # For neighborhood sampling compatibility
        if isinstance(idx, tuple) and len(idx) == 3:
            batch_size, n_id, adjs = idx
            return (batch_size, n_id, adjs)
        
        # Regular batching
        return (self.data.x,           # [N, d_node]
                self.data.edge_index,
                self.data.edge_attr,
                self.data.batch,       # [N]
                self.data.flat,        # None or [N, d_flat]
                self.data.x_hist,      # [N, T, d_ts]
                self.data.y)           # [N] or graph label


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
