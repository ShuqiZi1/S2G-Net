"""
MambaGraphGPS implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.mamba import define_mamba_encoder
from src.models.graphgps import define_graphgps_encoder
from src.models.utils import get_act_fn


class MambaGraphGPS(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg = dict(config)
        cfg.setdefault('mamba_pooling', 'mean') 
        self.mamba_encoder = define_mamba_encoder()(cfg)
        gps_cfg = dict(config)
        gps_cfg.update({
            'use_flat': False, 
            'flat_dim': 0
        })
        self.graphgps_encoder = define_graphgps_encoder()(gps_cfg)
        
        self.mamba_d_model = config.get('mamba_d_model', 256)
        self.mamba_ts_dim = config.get('mamba_last_ts_dim') or self.mamba_d_model
        
        self.gps_hidden_dim = config.get('gps_hidden_dim', 256)
        self.gps_out_dim = config.get('gps_out_dim', config.get('out_dim', 1))
        self.out_dim = config.get('out_dim', 1)
        
        self.add_flat = config.get('add_flat', True)
        self.flat_after = config.get('flat_after', True)
        self.add_last_ts = config.get('add_last_ts', True)
        
        self.mamba_to_gps = nn.Sequential(
            nn.Linear(self.mamba_d_model, self.gps_hidden_dim),
            nn.LayerNorm(self.gps_hidden_dim),
            nn.GELU(),
            nn.Linear(self.gps_hidden_dim, self.gps_hidden_dim),
            nn.LayerNorm(self.gps_hidden_dim)
        )
        
        if self.mamba_d_model != self.gps_hidden_dim:
            self.mamba_to_gps = nn.Sequential(
                nn.Linear(self.mamba_d_model, self.gps_hidden_dim),
                nn.LayerNorm(self.gps_hidden_dim),
                nn.GELU()
            )
        
        self.flat_dim = config.get('flat_dim', 0)
        if self.flat_after and self.flat_dim:
            self.flat_fc = nn.Sequential(
                nn.Linear(self.flat_dim, config.get('flat_nhid', self.flat_dim)),
                nn.LayerNorm(config.get('flat_nhid', self.flat_dim)),
                nn.GELU()
            )
            self.flat_dim = config.get('flat_nhid', self.flat_dim)
        else:
            self.flat_dim = 0
        
        fc_in_dim = self.gps_out_dim
        
        if self.add_last_ts:
            fc_in_dim += self.mamba_ts_dim
        if self.flat_after:
            fc_in_dim += self.flat_dim
        
        self.out_layer = nn.Sequential(
            nn.Linear(fc_in_dim, fc_in_dim // 2),
            nn.LayerNorm(fc_in_dim // 2),
            nn.GELU(),
            nn.Dropout(config.get('main_dropout', 0.1)),
            nn.Linear(fc_in_dim // 2, self.out_dim)
        )

        self.ts_use_flat = config.get('ts_use_flat', True) and self.flat_dim > 0
        ts_in_dim = self.mamba_ts_dim + (self.flat_dim if self.ts_use_flat else 0)

        self.mamba_out = nn.Sequential(
            nn.Linear(ts_in_dim, ts_in_dim // 2),
            nn.LayerNorm(ts_in_dim // 2),
            nn.GELU(),
            nn.Linear(ts_in_dim // 2, self.out_dim)
        )
        
        self.dropout = nn.Dropout(config.get('main_dropout', 0.1))
        self.gps_norm = nn.LayerNorm(self.gps_out_dim)
        self.mamba_norm = nn.LayerNorm(self.mamba_ts_dim)
        
        self.last_act = nn.Identity()
        if config.get('final_act_fn') == 'hardtanh':
            self.last_act = nn.Hardtanh(min_val=0.0, max_val=14.0)
        
        self._initialize_weights()
        
        self.lambda_gps  = nn.Parameter(torch.ones(1))   # GraphGPS branch
        self.lambda_ts   = nn.Parameter(torch.ones(1))   # last-timestamp branch
        self.lambda_flat = nn.Parameter(torch.ones(1))   # flat branch

    def _gps_inference(
        self,
        node_feats,               # [N_total, D] — full node features after Mamba + projection
        flat_all,                 # [N_total, Df] — optional flat features for each node
        subgraph_loader,          # NeighborSampler for inference
        device,
        edge_weight               # [E_total, *] — full edge attributes or weights (optional)
    ):
        """
        Fallback inference function for GraphGPS using subgraph sampling.
        This is used when GraphGPS has no 'inference()' method like other GNNs.
        """
        outs = []
    
        for batch_size, n_id, adj in subgraph_loader:
            # Unpack subgraph
            edge_index, edge_ids, _ = adj.to(device)
    
            # Get subgraph node features
            x_sub = node_feats[n_id].to(device)
    
            # Slice flat features if used
            flat_sub = flat_all[n_id].to(device) if flat_all is not None else None
    
            # Subset edge attributes if provided
            if edge_weight is not None and edge_ids is not None:
                edge_attr_sub = edge_weight[edge_ids].to(device)
            else:
                edge_attr_sub = None
    
            # Forward pass through GraphGPS
            y_sub = self.graphgps_encoder(
                x_sub,               # node features for the subgraph
                edge_index,          # subgraph edge index
                edge_attr=edge_attr_sub,  # subgraph edge attributes (optional)
                batch=None,          # no batch vector needed for node-level tasks
                flat=None        # optional flat input
            )
    
            # Keep only the output for target (center) nodes in this batch
            outs.append(y_sub[:batch_size])
    
        # Concatenate outputs to recover full-graph node predictions
        return torch.cat(outs, dim=0)  # [N_total, D]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, flat, adjs, batch_size, edge_weight=None):
        seq = x                                    # [B, T, d]
        # ---------- build boolean padding mask ----------
        masks = (seq.abs().sum(-1) > 0)            # pad → False
        mamba_out, _ = self.mamba_encoder(seq, masks)
        
        if mamba_out.dim() == 3:                     # (B, T, d)
            valid = masks.unsqueeze(-1)              # (B, T, 1)
            summed = (mamba_out * valid).sum(1)
            lengths = valid.sum(1).clamp_min(1)
            last_all = summed / lengths             # mean-pooling
        else:
            last_all = mamba_out 
        last_all = self.mamba_norm(last_all)                  # [N_sub, d_mamba]
        last_batch = last_all[:batch_size]
        
        gps_input = self.mamba_to_gps(last_all)               # [N_sub, 128]

        flat_all = None
        if self.flat_dim > 0 and flat is not None:
            flat_all = flat.to(x.device)                      # [N_sub, flat_dim]               

        edge_index, _, _ = adjs[0]
        edge_index = edge_index.to(x.device)
        edge_attr = None
        if edge_weight is not None and hasattr(adjs[0], "e_id"):
            edge_attr = edge_weight[adjs[0].e_id].to(x.device)

        gps_out_all = self.graphgps_encoder(
            gps_input, edge_index, edge_attr=edge_attr,
            batch=None, flat=None                             
        )                                                     # [N_sub, gps_out_dim]

        gps_out = gps_out_all[:batch_size]
        gps_out = self.gps_norm(gps_out)
        
        flat_batch = flat[:batch_size] if (self.flat_dim and flat is not None) else None
        flat_out   = self.flat_fc(flat_batch) if (self.flat_after and flat_batch is not None) else None
        
        # --- weighting ---
        lambdas  = torch.cat([self.lambda_gps, self.lambda_ts, self.lambda_flat])
        weights  = torch.softmax(lambdas, dim=0)

        w_gps, w_ts, w_flat = weights
        
        # --- build list (each branch keeps its own width) ---
        branches = [w_gps * gps_out]                 # (B, gps_out_dim = 1)
        if last_batch is not None:
            branches.append(w_ts * last_batch)       # (B, mamba_ts_dim = 192)
        if flat_out is not None:
            branches.append(w_flat * flat_out)       # (B, flat_dim = 0 or N)
        
        combined = torch.cat(branches, dim=1)        # <- width back to 193
        y = self.last_act(self.out_layer(combined))

        ts_in = torch.cat([last_batch, flat_out], dim=1) \
                if (self.ts_use_flat and flat_out is not None) else last_batch
        mamba_y = self.last_act(self.mamba_out(ts_in))
        
        return y, mamba_y

    def infer_mamba_by_batch(self, ts_loader, device):
        mamba_outs = []
        lasts = []
        mamba_ys = []
        
        for inputs, labels, ids in ts_loader:
            seq, flat, masks = inputs 
            seq = seq.to(device)
            masks = masks.to(device)
            
            with torch.no_grad():
                last, _ = self.mamba_encoder(seq, masks)
                last   = self.mamba_norm(last)
                if self.ts_use_flat and flat is not None:
                    flat_proj = self.flat_fc(flat.to(device))
                    ts_in = torch.cat([last, flat_proj], dim=1)
                else:
                    ts_in = last
                mamba_y = self.mamba_out(ts_in)
            
            mamba_outs.append(last)
            lasts.append(last)
            mamba_ys.append(mamba_y)
        
        mamba_outs = torch.cat(mamba_outs, dim=0)
        lasts = torch.cat(lasts, dim=0)
        mamba_ys = torch.cat(mamba_ys, dim=0)
        
        return mamba_outs, lasts, mamba_ys
        
    def inference(self, x_all, flat_all, edge_weight, ts_loader, subgraph_loader, device, get_emb=False):
        mamba_outs, last_all, mamba_ys = self.infer_mamba_by_batch(ts_loader, device)
        
        with torch.no_grad():
            gps_input = self.mamba_to_gps(mamba_outs)  # [N, D]
    
            # Use native GraphGPS inference if available
            if hasattr(self.graphgps_encoder, "inference"):
                gps_out = self.graphgps_encoder.inference(
                    gps_input, None, subgraph_loader, device,
                    edge_weight, last_all, get_emb=get_emb
                )
            else:
                gps_out = self._gps_inference(gps_input, None, subgraph_loader, device, edge_weight)
    
            features = [gps_out]
            if self.add_last_ts:
                features.append(last_all)

            flat_out = None
            if self.flat_after and flat_all is not None:
                flat_out = self.flat_fc(flat_all)
                features.append(flat_out)

            combined = torch.cat(features, dim=1)
            y = self.last_act(self.out_layer(combined))

            ts_in = torch.cat([last_all, flat_out], dim=1) \
                    if (self.ts_use_flat and flat_out is not None) else last_all
            mamba_ys = self.last_act(self.mamba_out(ts_in))

        return y, mamba_ys


    def inference_w_attn(self, x_all, flat_all, edge_weight, edge_index, ts_loader, subgraph_loader, device):
        mamba_outs, last_all, mamba_ys = self.infer_mamba_by_batch(ts_loader, device)
        
        with torch.no_grad():
            gps_input = self.mamba_to_gps(mamba_outs)
            
            gps_out, edge_index_w_self_loops, all_edge_attn = self.graphgps_encoder.inference_w_attn(
                gps_input, flat_all, device, edge_weight, edge_index, last_all
            )
            
            features = [gps_out]
            
            if self.add_last_ts:
                features.append(last_all)

            flat_out = None
            if self.flat_after and flat_all is not None:
                flat_out = self.flat_fc(flat_all)
                features.append(flat_out)

            combined = torch.cat(features, dim=1)
            y = self.last_act(self.out_layer(combined))

            ts_in = torch.cat([last_all, flat_out], dim=1) \
                    if (self.ts_use_flat and flat_out is not None) else last_all
            mamba_ys = self.last_act(self.mamba_out(ts_in))
        
        return y, mamba_ys, edge_index_w_self_loops, all_edge_attn