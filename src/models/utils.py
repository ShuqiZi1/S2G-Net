import os
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from pathlib import Path

def get_checkpoint_path(load_path: str) -> str:
    path = Path(load_path)
    if path.is_file() and path.suffix == '.ckpt':
        return str(path)
    
    model_dir = path
    chkpts = list(model_dir.glob('checkpoints/*'))
    if not chkpts:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}/checkpoints/")
    
    return str(chkpts[-1])



def define_loss_fn(config):
    """define loss function"""
    if config['classification']:
        if config['class_weights'] is False:
            return nn.CrossEntropyLoss()
        else:
            return nn.CrossEntropyLoss(weight=torch.from_numpy(config['class_weights']).float())
    else:
        return nn.MSELoss()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stack_outputs_with_mask(outputs, multi_gpu, val_nid=None):
    log_dict = {}
    for loss_type in outputs[0]:
        if multi_gpu:
            collect = []
            for output in outputs:
                for v in output[loss_type]:
                    collect.append(v)
        else:
            collect = [v[loss_type] for v in outputs]
        if 'loss' in loss_type:
            log_dict[loss_type] =  torch.Tensor(collect)
        else:
            log_dict[loss_type] =  torch.cat(collect).squeeze()
    # mask out
    if val_nid is not None:
        tmp = log_dict[loss_type]
        val_nid = val_nid.type_as(tmp)
        mask  = [n in val_nid for n in log_dict['ids']]
        anti_mask  = [n not in val_nid for n in log_dict['ids']]
        val_dict = {name: log_dict[name][mask] for name in log_dict}
        train_log_dict = {name: log_dict[name][anti_mask] for name in log_dict}
    else:
        val_dict = log_dict
        train_log_dict = log_dict
    return val_dict, train_log_dict


def collect_outputs(outputs, multi_gpu=False):
    """
    Collect and aggregate outputs from multiple batches.
    
    Args:
        outputs: List of dictionaries, each containing batch outputs
        multi_gpu: Whether using multiple GPUs
    
    Returns:
        Dictionary with concatenated tensors from all batches
    """
    
    # Handle multi-GPU case
    if multi_gpu and len(outputs) > 0 and isinstance(outputs[0], list):
        outputs = [item for sublist in outputs for item in sublist]
    
    # No outputs case
    if len(outputs) == 0:
        return {}
    
    # Initialize collection dictionary
    out = {}
    keys = outputs[0].keys()
    for k in keys:
        out[k] = []
    
    # Collect outputs from all batches
    for batch_output in outputs:
        for k in keys:
            # Add batch output to collection
            out[k].append(batch_output[k])
    
    # Process each key to concatenate tensors
    for k in out:
        if isinstance(out[k][0], torch.Tensor):
            if out[k][0].numel() == 1 and out[k][0].dim() == 0:
                # For scalar tensors (like loss values), average them
                out[k] = torch.stack(out[k]).mean()
            else:
                # For non-scalar tensors (predictions, ground truth), concatenate them
                try:
                    out[k] = torch.cat(out[k], dim=0)
                except Exception as e:
                    # Don't fall back to first tensor, raise the error
                    raise e
    
    return out


def init_weights(modules):
    """initialize model weights"""
    for m in modules:
        if isinstance(m, nn.LSTM):
            nn.init.xavier_uniform_(m.weight_ih_l0)
            nn.init.orthogonal_(m.weight_hh_l0)
            for names in m._all_weights:
                for name in filter(lambda n: 'bias' in n, names):
                    bias = getattr(m, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.0)
        elif isinstance(m, nn.Linear):
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()


def get_act_fn(name):
    """define activation function"""
    if name is None:
        act_fn = lambda x: x
    elif name == 'gelu':
        return torch.nn.GELU()
    elif name == 'relu':
        act_fn = nn.ReLU()
    elif name == 'leakyrelu':
        act_fn = nn.LeakyReLU()
    elif name == 'hardtanh':
        act_fn = nn.Hardtanh(min_val=1 / 48, max_val=100)
    return act_fn