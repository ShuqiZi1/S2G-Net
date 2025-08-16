"""
Defining GNN models (with node-sampling)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, NNConv, GCNConv
from src.models.utils import init_weights, get_act_fn


def define_ns_gnn_encoder(gnn_name):
    """
    return specified model class for GNNs with node-sampling.
    """
    if gnn_name == 'gat':
        return SamplingGAT
    elif gnn_name == 'sage':
        return SAGE
    elif gnn_name == 'mpnn':
        return SamplingMPNN
    elif gnn_name == 'gcn':
        return SamplingGCN
    else:
        raise NotImplementedError("node sampling only implemented for GAT, SAGE and MPNN models.")


def determine_fc_in_dim(config):
    """
    return dimensions of layers
    """
    flat_after = config['flat_after']
    add_lstm = config['add_last_ts']
    flat_dim = config['flat_nhid'] if config['flat_nhid'] is not None else config['num_flat_feats']
    if flat_after and add_lstm:
        in_dim = config['gnn_outdim'] + flat_dim + config['lstm_last_ts_dim']
    elif flat_after:
        in_dim = config['gnn_outdim'] + flat_dim
    else:
        in_dim = config['gnn_outdim'] + flat_dim
    return flat_after, add_lstm, in_dim, flat_dim


class NsGNN(nn.Module):
    """
    Model class for GNN with node-sampling scheme.
    """
    def __init__(self, config):
        super().__init__()
        self.gnn_encoder = define_ns_gnn_encoder(config['gnn_name'])(config)
        self.last_act = get_act_fn(config['final_act_fn'])
        self._initialize_weights()

    def _initialize_weights(self):
        init_weights(self.modules())

    def forward(self, x, flat, adjs, edge_weight):
        gnn_out = self.gnn_encoder.forward(x, flat, adjs, edge_weight)
        out = self.last_act(gnn_out)
        return out
    
    def inference(self, x_all, flat_all, subgraph_loader, device, edge_weight, last_all=None, get_attn=False, get_emb=False):
        """
        Inference wrapper for GNNs with node sampling.
    
        Returns:
            - If GNN returns a single tensor: activated output
            - If GNN returns a tuple (e.g., for GAT with attention): (activated output, ...)
        """
        out = self.gnn_encoder.inference(x_all, flat_all, subgraph_loader, device, edge_weight, last_all, get_emb=get_emb, get_attn=get_attn)
    
        if isinstance(out, tuple):
            # Apply activation only to the first element (node output)
            out = (self.last_act(out[0]), *out[1:])
        else:
            out = self.last_act(out)
    
        return out



class SAGE(torch.nn.Module):
    """
    Model class for SAGE with node sampling scheme.
    """
    def __init__(self, config):
        super().__init__()

        self.num_layers = 2
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(config['gnn_indim'], config['sage_nhid']))
        self.convs.append(SAGEConv(config['sage_nhid'], config['gnn_outdim']))
        self.main_dropout = config['main_dropout']

        self.flat_after, self.add_lstm, fc_in_dim, flat_dim = determine_fc_in_dim(config)
        if self.flat_after:
            self.flat_fc = nn.Linear(config['num_flat_feats'], flat_dim)
        if self.flat_after or self.add_lstm:
            self.out_layer = nn.Linear(fc_in_dim, config['num_cls'])
    
    def to_concat_vector(self, x, flat, last, bsz_nids=None):
        toc = [x]
        if self.flat_after:
            flat_bsz = flat[bsz_nids] if bsz_nids is not None else flat
            flat_bsz = self.flat_fc(flat_bsz)
            toc.append(flat_bsz)
        if self.add_lstm:
            if bsz_nids is not None:
                toc.append(last[bsz_nids])
            else:
                toc.append(last)
        return toc

    def forward(self, x, flat, adjs, edge_weight=None, last=None):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        

        toc = self.to_concat_vector(x, flat, last)
        if self.flat_after or self.add_lstm:
            x = torch.cat(toc, dim=1)
            x = F.dropout(x, p=self.main_dropout, training=self.training)
            x = self.out_layer(x)
        return x

    def inference(self, x_all, flat_all, subgraph_loader, device, edge_weight=None, last_all=None, get_emb=False, get_attn=False):
        cat_in_last_layer = self.flat_after or self.add_lstm
        if get_emb:
            cat_in_last_layer = False
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                elif cat_in_last_layer: # last layer
                    bsz_nids = n_id[:batch_size]
                    toc = self.to_concat_vector(x, flat_all, last_all, bsz_nids)
                    x = torch.cat(toc, dim=1)
                    x = self.out_layer(x)
                    
                xs.append(x)
            x_all = torch.cat(xs, dim=0)

        return x_all


class SamplingGAT(torch.nn.Module):
    """
    Model class for GAT with node sampling scheme.
    """
    def __init__(self, config):
        super().__init__()

        self.featdrop = config['gat_featdrop']
        self.num_layers = 2
        in2 = config['gat_nhid']*config['gat_n_heads']
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(config['gnn_indim'], config['gat_nhid'], \
            heads=config['gat_n_heads'], dropout=config['gat_attndrop']))
        self.convs.append(GATConv(in2, config['gnn_outdim'], \
            heads=config['gat_n_out_heads'], concat=False, dropout=config['gat_attndrop']))
        self.main_dropout = config['main_dropout']
        self.flat_after, self.add_lstm, fc_in_dim, flat_dim = determine_fc_in_dim(config)
        if self.flat_after or self.add_lstm:
            self.out_layer = nn.Linear(fc_in_dim, config['num_cls'])
        if self.flat_after:
            self.flat_fc = nn.Linear(config['num_flat_feats'], flat_dim)
    
    def to_concat_vector(self, x, flat, last, bsz_nids=None):
        toc = [x]
        if self.flat_after:
            flat_bsz = flat[bsz_nids] if bsz_nids is not None else flat
            flat_bsz = self.flat_fc(flat_bsz)
            toc.append(flat_bsz)
        if self.add_lstm:
            if bsz_nids is not None:
                toc.append(last[bsz_nids])
            else:
                toc.append(last)
        return toc

    def forward(self, x, flat, adjs, edge_weight=None, last=None):
        x = F.dropout(x, p=self.featdrop, training=self.training)
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.elu(x)
        toc = self.to_concat_vector(x, flat, last)
        if self.flat_after or self.add_lstm:
            x = torch.cat(toc, dim=1)
            x = F.dropout(x, p=self.main_dropout, training=self.training)
            x = self.out_layer(x)
        return x

    def inference_whole(self, x_all, flat_all, device, edge_weight=None, edge_index=None, last_all=None, get_emb=False, get_attn=False):
        cat_in_last_layer = self.flat_after or self.add_lstm
        if get_emb:
            cat_in_last_layer = False
        all_edge_attn = []
        x = x_all
        for i in range(self.num_layers):
            print('layer ',i )
            x, attn = self.convs[i].forward(x, edge_index, return_attention_weights=True)
            edge_attn = attn[1]
            all_edge_attn.append(edge_attn)
            if i != self.num_layers - 1:
                x = F.elu(x)
            elif cat_in_last_layer: # last layer
                toc = self.to_concat_vector(x, flat_all, last_all)
                x = torch.cat(toc, dim=1)
                x = self.out_layer(x)

        edge_index_w_self_loops = attn[0]

        return x, edge_index_w_self_loops, all_edge_attn

    def inference(self, x_all, flat_all, subgraph_loader, device, edge_weight=None, last_all=None, get_emb=False, get_attn=False):
        cat_in_last_layer = self.flat_after or self.add_lstm
        if get_emb:
            cat_in_last_layer = False
        all_edge_attn = []
        for i in range(self.num_layers):
            edge_index_w_self_loops = []
            edge_attn = []
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                if get_attn:
                    x, attn = self.convs[i].forward((x, x_target), edge_index, return_attention_weights=True)
                    edge_index_w_self_loops.append(attn[0])
                    edge_attn.append(attn[1])
                else:
                    x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.elu(x)
                elif cat_in_last_layer: # last layer
                    bsz_nids = n_id[:batch_size]
                    toc = self.to_concat_vector(x, flat_all, last_all, bsz_nids)
                    x = torch.cat(toc, dim=1)
                    x = self.out_layer(x)

                xs.append(x)
            x_all = torch.cat(xs, dim=0)
            if i == 1:
                if len(edge_index_w_self_loops) > 0:
                    edge_index_w_self_loops = torch.cat(edge_index_w_self_loops, dim=1)
                else:
                    edge_index_w_self_loops = torch.empty((2, 0), dtype=torch.long, device=device)

            if len(edge_attn) > 0:
                edge_attn = torch.cat(edge_attn, dim=0)
            else:
                edge_attn = torch.empty((0, self.convs[i].heads), device=device)


            all_edge_attn.append(edge_attn)

        return x_all, edge_index_w_self_loops, all_edge_attn



class SamplingMPNN(torch.nn.Module):
    """
    Model class for MPNN with node sampling scheme.
    """
    def __init__(self, config):
        super(SamplingMPNN, self).__init__()
        dim = config['mpnn_nhid']
        self.lin0 = torch.nn.Linear(config['gnn_indim'], dim)
        nn_layers = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn_layers, aggr='mean')
        self.gru = nn.GRU(dim, dim)
        self.steps = config['mpnn_step_mp']
        self.lin1 = torch.nn.Linear(dim, dim)
        self.lin2 = torch.nn.Linear(dim, config['gnn_outdim'])
        self.main_dropout = config['main_dropout']
        self.flat_after, self.add_lstm, fc_in_dim, flat_dim = determine_fc_in_dim(config)
        if self.flat_after or self.add_lstm:
            self.out_layer = nn.Linear(fc_in_dim, config['num_cls'])
        if self.flat_after:
            self.flat_fc = nn.Linear(config['num_flat_feats'], flat_dim)
        

    def to_concat_vector(self, x, flat, last, bsz_nids=None):
        toc = [x]
        if self.flat_after:
            flat_bsz = flat[bsz_nids] if bsz_nids is not None else flat
            flat_bsz = self.flat_fc(flat_bsz)
            toc.append(flat_bsz)
        if self.add_lstm:
            if bsz_nids is not None:
                toc.append(last[bsz_nids])
            else:
                toc.append(last)
        return toc

    def forward(self, x, flat, adjs, edge_weight, last=None):
        """
        Forward method for SamplingMPNN.
        Compatible with both single-layer and multi-layer neighbor sampling output.
        """
        x = F.relu(self.lin0(x))
    
        # --- Compatible unpacking for both (edge_index, e_id, size) and [(edge_index, e_id, size)] ---
        # If adjs is a tuple of length 3, treat it as a single layer
        if isinstance(adjs, tuple) and len(adjs) == 3:
            edge_index, edge_ids, size = adjs
        # If adjs is a list/tuple containing a single tuple (for single-hop)
        elif (isinstance(adjs, (list, tuple)) and len(adjs) == 1 and 
              isinstance(adjs[0], tuple) and len(adjs[0]) == 3):
            edge_index, edge_ids, size = adjs[0]
        # If adjs is a list of layers (multi-hop), use the first layer
        elif isinstance(adjs, (list, tuple)) and len(adjs) > 1:
            edge_index, edge_ids, size = adjs[0]
        else:
            raise RuntimeError(f"Unsupported adjs format: {adjs}")
    
        x_target = x[:size[1]]
        hid = x_target.unsqueeze(0)
        for step in range(self.steps):
            m = F.relu(self.conv((x, x_target), edge_index, edge_weight[edge_ids]))
            out, hid = self.gru(m.unsqueeze(0), hid)
            out = out.squeeze(0)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        
        # Concatenate additional vectors if needed (flat, last layer output)
        toc = self.to_concat_vector(out, flat, last)
        if self.flat_after or self.add_lstm:
            out = torch.cat(toc, dim=1)
            out = F.dropout(out, p=self.main_dropout, training=self.training)
            out = self.out_layer(out)
        return out


    def inference(self, x_all, flat_all, subgraph_loader, device, edge_weight, last_all=None, get_emb=False, get_attn=False):
        cat_in_last_layer = self.flat_after or self.add_lstm
        if get_emb:
            cat_in_last_layer = False
        xs = []
        x_all = F.relu(self.lin0(x_all))
        for batch_size, n_id, adj in subgraph_loader:
            edge_index, edge_ids, size = adj.to(device)
            x = x_all[n_id].to(device)
            x_target = x[:size[1]]
            hid = x_target.unsqueeze(0)
            for s in range(self.steps):
                m = F.relu(self.conv((x, x_target), edge_index, edge_weight[edge_ids]))
                out, hid = self.gru(m.unsqueeze(0), hid)
                out = out.squeeze(0)
            out = F.relu(self.lin1(out))
            out = self.lin2(out)
            # last layer
            if cat_in_last_layer:
                bsz_nids = n_id[:batch_size]
                toc = self.to_concat_vector(out, flat_all, last_all, bsz_nids)
                x = torch.cat(toc, dim=1)
                out = self.out_layer(x)
            xs.append(out)
        x_all = torch.cat(xs, dim=0)
        return x_all


class SamplingGCN(nn.Module):
    """
    Two-layer GCN with neighbor sampling, fully compatible with the existing NsGNN wrapper.
    """
    def __init__(self, config):
        super().__init__()
        self.num_layers = 2
        self.convs = nn.ModuleList([
            GCNConv(config['gnn_indim'], config['gcn_nhid']),
            GCNConv(config['gcn_nhid'], config['gnn_outdim'])
        ])
        self.dropout = config['gcn_dropout']

        # Optional flat feature / last-step LSTM concatenation (keeps parity with the other models)
        self.flat_after, self.add_lstm, fc_in_dim, flat_dim = determine_fc_in_dim(config)
        if self.flat_after:
            self.flat_fc = nn.Linear(config['num_flat_feats'], flat_dim)
        if self.flat_after or self.add_lstm:
            self.out_layer = nn.Linear(fc_in_dim, config['num_cls'])

    # Utility: gather everything that needs to be concatenated
    def _to_concat(self, x, flat, last, bsz_nids=None):
        vs = [x]
        if self.flat_after:
            f = flat[bsz_nids] if bsz_nids is not None else flat
            vs.append(self.flat_fc(f))
        if self.add_lstm:
            vs.append(last[bsz_nids] if bsz_nids is not None else last)
        return vs

    def forward(self, x, flat, adjs, edge_weight=None, last=None):
        for i, (edge_index, _, size) in enumerate(adjs):
            x = self.convs[i](x, edge_index)
            x = x[:size[1]]
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Optionally concatenate flat features and/or LSTM output
        if self.flat_after or self.add_lstm:
            x = torch.cat(self._to_concat(x, flat, last), dim=1)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.out_layer(x)
        return x

    def inference(self, x_all, flat_all, subgraph_loader, device,
                  edge_weight=None, last_all=None, get_emb=False, get_attn=False):
        """
        Mirrors the layer-by-layer, batch-wise inference routine used by SAGE/GAT.
        """
        cat_last = (self.flat_after or self.add_lstm) and not get_emb
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x = self.convs[i](x, edge_index)
                x = x[:size[1]]
                if i != self.num_layers - 1:
                    x = F.relu(x)
                elif cat_last:  # final layer before classification
                    bsz_nids = n_id[:batch_size]
                    x = torch.cat(self._to_concat(x, flat_all, last_all, bsz_nids), dim=1)
                    x = self.out_layer(x)
                xs.append(x)
            x_all = torch.cat(xs, dim=0)
        return x_all
    