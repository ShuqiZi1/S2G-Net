"""
Dataloaders for temporal model
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from src.dataloader.convert import read_mm
from torch.utils.data import DataLoader

def slice_data(data, info, split):
    """Slice data according to the instances belonging to each split."""
    if split is None:
        return data
    elif split == 'train':
        return data[:info['train_len']]
    elif split == 'val':
        train_n = info['train_len']
        val_n = train_n + info['val_len']
        return data[train_n: val_n]
    elif split == 'test':
        val_n = info['train_len'] + info['val_len']
        test_n = val_n + info['test_len']
        return data[val_n: test_n]


def no_mask_cols(ts_info, seq):
    """do not apply temporal masks"""
    neg_mask_cols = [i for i, e in enumerate(ts_info['columns']) if 'mask' not in e]
    return seq[:, :, neg_mask_cols]


def collate_fn(x_list, task):
    """
    collect samples in each batch with padding for variable length sequences
    """
    # Get max sequence length in this batch
    max_len = max([sample[0].shape[0] for sample in x_list])
    
    # Prepare padded sequences
    padded_seqs = []
    for sample in x_list:
        seq = sample[0]
        seq_len = seq.shape[0]
        
        # Create padded sequence
        if seq_len < max_len:
            padding = np.zeros((max_len - seq_len, seq.shape[1]))
            padded_seq = np.concatenate([seq, padding], axis=0)
        else:
            padded_seq = seq
            
        padded_seqs.append(padded_seq)
    
    # Create tensors
    seq = torch.Tensor(np.stack(padded_seqs)).float()  # [bsz, seq_len, ts_dim]
    flat = torch.Tensor(np.stack([sample[1] for sample in x_list])).float()  # [bsz, flat_dim]
    inputs = (seq, flat)
    
    if task == 'los':
        labels = torch.Tensor(np.stack([sample[2] for sample in x_list])).float()  # [bsz,]
    else:
        labels = torch.Tensor(np.stack([sample[2] for sample in x_list])).long()  # [bsz,]
        
    ids = torch.Tensor(np.stack([sample[3] for sample in x_list])).long()
    
    return inputs, labels, ids
    
def collect_ts_flat_labels(data_dir, ts_mask, task, add_diag, split=None, 
                           debug=0, split_flat_and_diag=False, use_time_encoding=False):
    """
    read temporal, flat data and task labels
    """
    flat_data, flat_info = read_mm(data_dir, 'flat')
    flat = slice_data(flat_data, flat_info, split)

    ts_data, ts_info = read_mm(data_dir, 'ts')
    seq = slice_data(ts_data, ts_info, split)
    mean = seq.mean(axis=(0,1), keepdims=True)
    std  = seq.std(axis=(0,1), keepdims=True) + 1e-8
    seq = (seq - mean) / std
    seq = np.clip(seq, -3.0, 3.0)
    if not ts_mask:
        seq = no_mask_cols(ts_info, seq)
    
    # Add time encoding if requested (new feature for RWKV)
    if use_time_encoding:
        # Create normalized time steps (0 to 1) for each sequence
        seq_len = seq.shape[1]
        time_encoding = np.tile(np.linspace(0, 1, seq_len).reshape(1, seq_len, 1), 
                                (seq.shape[0], 1, 1))
        # Concatenate time encoding to sequence data
        seq = np.concatenate([seq, time_encoding], axis=2)

    if add_diag:
        diag_data, diag_info = read_mm(data_dir, 'diagnoses')
        diag = slice_data(diag_data, flat_info, split)
        if split_flat_and_diag:
            flat = (flat, diag)
        else:
            flat = np.concatenate([flat, diag], 1)

    label_data, labels_info = read_mm(data_dir, 'labels')
    labels = slice_data(label_data, flat_info, split)
    idx2col = {'ihm': 1, 'los': 3, 'multi': [1, 3]}
    label_idx = idx2col[task]
    labels = labels[:, label_idx]

    if debug:
        N = 1000
        train_n = int(N*0.5)
        val_n = int(N*0.25)
    else:
        N = flat_info['total']
        train_n = flat_info['train_len']
        val_n = flat_info['val_len']
    
    seq = seq[:N]
    flat = flat[:N]
    labels = labels[:N]
    
    return seq, flat, labels, flat_info, N, train_n, val_n


def get_class_weights(train_labels):
    """
    return class weights to handle class imbalance problems
    """
    occurences = np.unique(train_labels, return_counts=True)[1]
    class_weights = occurences.sum() / occurences
    class_weights = torch.Tensor(class_weights).float()
    return class_weights

class LstmDataset(Dataset):
    """
    Dataset class for temporal data.
    """
    def __init__(self, config, split=None):
        super().__init__()
        task = config['task']

        self.seq, self.flat, self.labels, self.ts_info, self.N, train_n, val_n = collect_ts_flat_labels(config['data_dir'], config['ts_mask'], \
                                                                                    task, config['add_diag'], split, debug=0)
        
        self.ts_dim = self.seq.shape[2]
        self.flat_dim = self.flat.shape[1]

        if split == 'train':
            self.split_n = train_n
        elif split == 'val':
            self.split_n = val_n
        elif split == 'test':
            self.split_n = self.N - train_n - val_n

        self.idx_val = range(train_n, train_n + val_n)
        self.idx_test = range(train_n + val_n, self.N)
        
        self.split_n = self.N if split is None else self.ts_info[f'{split}_len']
        all_nodes = np.arange(self.N)
        self.ids = slice_data(all_nodes, self.ts_info, split) # (N_split.)

        self.class_weights = get_class_weights(self.labels[:train_n]) if task == 'ihm' else False

    def __len__(self):
        return self.split_n
    
    def __getitem__(self, index):
        return self.seq[index], self.flat[index], self.labels[index], self.ids[index]


class MambaDataset(Dataset):
    """
    Dataset class optimized for Mamba state space model processing
    """
    def __init__(self, config, split=None):
        super().__init__()
        
        # Extract configuration parameters
        task = config['task']
        add_pos_enc = config.get('add_positional_encoding', False)
        max_seq_len = config.get('max_seq_len', 0)
        
        # Load and prepare data
        self.seq, self.flat, self.labels, self.ts_info, self.N, train_n, val_n = collect_ts_flat_labels(config['data_dir'], config['ts_mask'], \
                                                                            task, config['add_diag'], split, debug=0)
                                                                                    
        # Track dimensions for model configuration
        self.ts_dim = self.seq.shape[2]
        self.flat_dim = self.flat.shape[1]

        # Set split size
        if split == 'train':
            self.split_n = train_n
        elif split == 'val':
            self.split_n = val_n
        elif split == 'test':
            self.split_n = self.N - train_n - val_n
        else:
            self.split_n = self.N

        # Set indices for validation and test sets
        self.idx_val = range(train_n, train_n + val_n)
        self.idx_test = range(train_n + val_n, self.N)
        
        # Save IDs for each sample
        all_nodes = np.arange(self.N)
        self.ids = slice_data(all_nodes, self.ts_info, split)
        
        # Limit sequence length if specified (Mamba can handle long sequences,
        # but we might want to limit for efficiency)
        if max_seq_len > 0:
            for i in range(len(self.seq)):
                if self.seq[i].shape[0] > max_seq_len:
                    # For Mamba, we might prefer to keep the last max_seq_len elements
                    # as they are likely more relevant for prediction
                    self.seq[i] = self.seq[i][-max_seq_len:]

        # Calculate class weights for classification tasks
        self.class_weights = get_class_weights(self.labels[:train_n]) if task == 'ihm' else None

    def __len__(self):
        """Return the number of samples in the dataset"""
        return self.split_n
    
    def __getitem__(self, index):
        """Get a sample from the dataset"""
        return self.seq[index], self.flat[index], self.labels[index], self.ids[index]


def mamba_collate_fn(x_list, task):
    """
    Collate function optimized for Mamba models
    
    Args:
        x_list: List of samples
        task: Task name for label processing
        
    Returns:
        Batch data ready for Mamba processing
    """
    # Get max sequence length in this batch
    max_len = max([sample[0].shape[0] for sample in x_list])
    
    # Prepare padded sequences and attention masks
    padded_seqs = []
    attention_masks = []  # 1 for real data, 0 for padding (useful for efficient processing)
    
    for sample in x_list:
        seq = sample[0]
        seq_len = seq.shape[0]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        # This can help Mamba's state space model focus on real data
        mask = np.ones(max_len)
        if seq_len < max_len:
            mask[seq_len:] = 0
        attention_masks.append(mask)
        
        # Create padded sequence
        if seq_len < max_len:
            padding = np.zeros((max_len - seq_len, seq.shape[1]))
            padded_seq = np.concatenate([seq, padding], axis=0)
        else:
            padded_seq = seq
            
        padded_seqs.append(padded_seq)
    
    # Create tensors
    seq = torch.Tensor(np.stack(padded_seqs)).float()  # [bsz, seq_len, ts_dim]
    masks = torch.Tensor(np.stack(attention_masks)).float()  # [bsz, seq_len]
    flat = torch.Tensor(np.stack([sample[1] for sample in x_list])).float()  # [bsz, flat_dim]
    
    # For Mamba, we package inputs with attention masks
    inputs = (seq, flat, masks)
    
    # Process labels based on task
    if task == 'los':
        labels = torch.Tensor(np.stack([sample[2] for sample in x_list])).float()  # [bsz,]
    else:
        labels = torch.Tensor(np.stack([sample[2] for sample in x_list])).long()  # [bsz,]
        
    ids = torch.Tensor(np.stack([sample[3] for sample in x_list])).long()
    
    return inputs, labels, ids


def create_mamba_dataloader(config, split=None, batch_size=None, shuffle=None, num_workers=None):
    """
    Create a DataLoader for Mamba models

    Args:
        config: Configuration dictionary
        split: Data split ('train', 'val', 'test')
        batch_size: Optional override for batch size
        shuffle: Whether to shuffle the data (True for train)
        num_workers: Number of parallel data loading workers

    Returns:
        PyTorch DataLoader instance
    """
    # Use default hyperparameters values if not specified
    if batch_size is None:
        batch_size = config.get('batch_size', 32)
    if num_workers is None:
        num_workers = config.get('num_workers', 4)
    if shuffle is None:
        shuffle = (split == 'train')

    # Load dataset
    dataset = MambaDataset(config, split=split)

    # 🔧 Automatically set the model's expected input dimension
    config['mamba_d_model'] = 128
    config['mamba_indim'] = dataset.ts_dim
    print("ts_dim:")
    print(dataset.ts_dim)

    # Define collate function
    collate = lambda x: mamba_collate_fn(x, config['task'])

    # Create and return DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

        

