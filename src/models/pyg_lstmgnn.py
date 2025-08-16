"""
Defining LSTM-GNN models
"""
from tqdm import tqdm
import torch
import torch.nn as nn
from src.models.lstm import define_lstm_encoder
from src.models.pyg_ns import define_ns_gnn_encoder
from src.models.utils import init_weights, get_act_fn

def _combine_last_flat(last, flat_center):
    """last: [B,H]; flat_center: [B,F] or None"""
    return torch.cat([last, flat_center], dim=-1) if flat_center is not None else last
        
class NsLstmGNN(torch.nn.Module):
    """
    model class for LSTM-GNN with node-sampling scheme.
    """
    def __init__(self, config):
        super().__init__()
        self.lstm_pooling = config['lstm_pooling']
        self.lstm_encoder = define_lstm_encoder()(config)
        self.gnn_name = config['gnn_name']
        self.gnn_encoder = define_ns_gnn_encoder(config['gnn_name'])(config)
        self.last_act = get_act_fn(config['final_act_fn'])
        in_dim = config['lstm_last_ts_dim']
        if config.get('add_flat', False):
            in_dim += config['num_flat_feats']
        
        self.lstm_out = nn.Linear(in_dim, config['out_dim'])

        self._initialize_weights()

    def _initialize_weights(self):
        init_weights(self.modules())

    def forward_lstm(self, seq, flat):
        seq = seq.permute(1, 0, 2)
        out, _ = self.lstm_encoder(seq)
        last = out[:, -1, :] if out.dim() == 3 else out           # [B,H]
        
        feat = _combine_last_flat(last, flat)                     
        lstm_y = self.last_act(self.lstm_out(feat))               
        return lstm_y

    def forward(self, x, flat, adjs, batch_size, edge_weight):
        seq = x.permute(1, 0, 2)
        out, _ = self.lstm_encoder.forward(seq)
        last = out[:, -1, :] if len(out.shape)==3 else out
        last = last[:batch_size]
        
        out = out.view(out.size(0), -1) # all_nodes, lstm_outdim
        x = out
        x = self.gnn_encoder(x, flat, adjs, edge_weight, last)
        y = self.last_act(x)        
        
        feat_center = _combine_last_flat(last, flat[:batch_size] if flat is not None else None)
        lstm_y = self.last_act(self.lstm_out(feat_center))   
        
        return y, lstm_y

    def infer_lstm_by_batch(self, ts_loader, device):
        lstm_outs = []
        lasts = []
        lstm_ys = []
        for inputs, labels, ids in ts_loader:
            seq, flat = inputs
            seq = seq.to(device)
            seq = seq.permute(1, 0, 2)
            out, _ = self.lstm_encoder.forward(seq)
            last = out[:, -1, :] if len(out.shape)==3 else out
            out = out.view(out.size(0), -1)
            feat = _combine_last_flat(last, flat.to(device))
            lstm_y = self.last_act(self.lstm_out(feat))
            lstm_outs.append(out)
            lasts.append(last)
            lstm_ys.append(lstm_y)
        lstm_outs = torch.cat(lstm_outs, dim=0) # [entire_g, dim]
        lasts = torch.cat(lasts, dim=0) # [entire_g, dim]
        lstm_ys = torch.cat(lstm_ys, dim=0)
        print('Got all LSTM output.')
        return lstm_outs, lasts, lstm_ys

    def inference(self, x_all, flat_all, edge_weight, ts_loader, subgraph_loader, device, get_emb=False):
        # first collect lstm outputs by minibatching:
        lstm_outs, last_all, lstm_ys = self.infer_lstm_by_batch(ts_loader, device)
        
        # then pass lstm outputs to gnn
        x_all = lstm_outs
        
        # Check what gnn_encoder.inference returns
        out = self.gnn_encoder.inference(x_all, flat_all, subgraph_loader, device, edge_weight, last_all, get_emb=get_emb)
        
        # If out is a tuple, we need to handle it appropriately
        if isinstance(out, tuple):
            # Apply activation to the first element of the tuple (assuming it's the main output tensor)
            processed_out = self.last_act(out[0])
            # Return the processed output along with any other elements from the original tuple
            return processed_out, lstm_ys
        else:
            # If out is a tensor, apply activation normally
            out = self.last_act(out)
            return out, lstm_ys

    def inference_w_attn(self, x_all, flat_all, edge_weight, edge_index, ts_loader, subgraph_loader, device):
        lstm_outs, last_all, lstm_ys = self.infer_lstm_by_batch(ts_loader, device)
        x_all = lstm_outs
        ret = self.gnn_encoder.inference_whole(x_all, flat_all, device, edge_weight, edge_index, last_all, get_attn=True)
        x_all, edge_index_w_self_loops, all_edge_attn = ret

        out = self.last_act(x_all)

        return out, lstm_ys, edge_index_w_self_loops, all_edge_attn

