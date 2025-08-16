"""GraphGPS‑Mamba
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GENConv, global_mean_pool
from torch_geometric.utils import degree
from mamba_ssm import Mamba
from src.models.mamba import RMSNorm

class NodeWiseMamba(nn.Module):
    def __init__(self, hidden_dim: int, config: dict):
        super().__init__()
        d_state     = config.get('mamba_d_state', 16)
        d_conv      = config.get('mamba_d_conv', 4)
        expand      = config.get('mamba_expand', 1)
        self.n_layers   = config.get('node_mamba_layers', config.get('mamba_layers', 2))
        self.dropout_p  = config.get('gps_dropout', 0.1)
        self.pooling    = config.get('mamba_pooling', 'last')
        self.seq_len    = config.get('mamba_seq_len', 10)

        input_dim = config.get('gps_node_dim', hidden_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            RMSNorm(hidden_dim, eps=1e-5),
            nn.GELU()
        )
        self.pre_norms = nn.ModuleList([RMSNorm(hidden_dim, eps=1e-5) for _ in range(self.n_layers)])
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(self.n_layers)
        ])
        self.drops = nn.ModuleList([nn.Dropout(self.dropout_p) for _ in range(self.n_layers)])
        self.final_norm = RMSNorm(hidden_dim, eps=1e-5)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        self.act = nn.GELU()

    def _infer_mask(self, x: torch.Tensor) -> torch.Tensor:
        return (x.abs().sum(dim=-1) > 0)

    def forward(self, x_hist: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if x_hist.dim() == 4 and x_hist.size(0) == 1:
            x_hist = x_hist.squeeze(0)
        if x_hist.dim() == 3 and x_hist.shape[0] == self.seq_len:
            x_hist = x_hist.permute(1, 0, 2)
        assert x_hist.dim() == 3, f"x_hist must be [N, T, D], got {x_hist.shape}"

        if mask is None:
            mask = self._infer_mask(x_hist)
        else:
            if mask.dtype != torch.bool:
                mask = mask > 0
            if mask.dim() == 2 and mask.size(0) == self.seq_len and mask.size(1) == x_hist.size(0):
                mask = mask.permute(1, 0).contiguous()

        x = self.input_proj(x_hist)
        residual = x
        for pre, mamba, drop in zip(self.pre_norms, self.mamba_layers, self.drops):
            x_m = mamba(pre(residual))
            residual = residual + drop(x_m)
        x = self.final_norm(residual)

        if self.pooling == 'last':
            last_idx = mask.sum(dim=1).long().clamp(min=1) - 1
            z = x[torch.arange(x.size(0), device=x.device), last_idx, :]
        elif self.pooling == 'mean':
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
            z = (x * mask.unsqueeze(-1)).sum(dim=1) / denom
        elif self.pooling == 'max':
            x_safe = torch.nan_to_num(x, nan=-1e9)
            z = torch.max(x_safe, dim=1)[0]
        elif self.pooling == 'all':
            z = x
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")

        if z.dim() == 2:
            z = self.output_proj(z)
            return self.dropout(self.act(z))
        else:
            z = self.output_proj(z)
            return self.dropout(self.act(z))
            

class EnhancedGraphGPSBlock(nn.Module):
    def __init__(self, hidden_dim: int, config: dict):
        super().__init__()
        # Local path: GENConv
        self.conv_norm = nn.LayerNorm(hidden_dim)
        self.genconv = GENConv(hidden_dim, hidden_dim,
                               aggr='softmax', learn_t=True,
                               t=1.0, num_layers=2, norm='layer')
        # Global path: Mamba on node‑degree‑sorted sequence
        d_state = config.get('mamba_d_state', 16)
        d_conv = config.get('mamba_d_conv', 4)
        expand = config.get('mamba_expand', 1)
        self.in_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.depth_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=d_conv,
                                    groups=hidden_dim, padding='same')
        self.mamba = Mamba(d_model=hidden_dim,
                            d_state=d_state,
                            d_conv=d_conv,
                            expand=expand)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Fusion
        self.dropout = nn.Dropout(config.get('gps_dropout', 0.1))
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.GELU(),
            nn.Linear(hidden_dim*4, hidden_dim)
        )
        self.post_bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
     
    @torch.no_grad()
    def _degree_order(self, edge_index: torch.Tensor, num_nodes: int, device):
        deg = degree(edge_index[0], num_nodes=num_nodes,
                     dtype=torch.float32).to(device)
        noise = torch.rand_like(deg) * 1e-2            # jitter for permutation
        order = torch.argsort(deg + noise)             # ascending
        return order, torch.argsort(order)

     
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor = None,
                batch: torch.Tensor = None):
    
        if x.dim() == 3 and x.size(0) == 1:
            x = x.squeeze(0)
    
        if isinstance(edge_index, (list, tuple)):
            edge_index = edge_index[0]
        if edge_index.dim() == 3 and edge_index.size(0) == 1:
            edge_index = edge_index.squeeze(0)
    
        if edge_attr is not None:
            if isinstance(edge_attr, (list, tuple)):
                edge_attr = edge_attr[0]
            if edge_attr.dim() == 3 and edge_attr.size(0) == 1:
                edge_attr = edge_attr.squeeze(0)
                
        N, D = x.size()
        resid = x
        # Local GENConv
        x_loc = self.conv_norm(x)
        x_loc = self.genconv(x_loc, edge_index, edge_attr)
        # Global Mamba
        order, rev = self._degree_order(edge_index, N, x.device)
        xs = x[order]                                # [N,D]
        xs = self.in_proj(xs).T.unsqueeze(0)             # [D, N] → [1, D, N]
        xs = self.depth_conv(xs).squeeze(0).T            # [1, D, N] → [1, D, N] → [N, D]
        xs = F.silu(xs)
        y = self.mamba(xs.unsqueeze(0)).squeeze(0)   # [N,D]
        y = y * F.silu(xs)                           # gating
        x_glo = self.out_proj(y)[rev]                # restore order
        # Fusion
        x_fuse = x_loc + x_glo
        x_fuse = self.post_bn(x_fuse)
        out = resid + self.dropout(self.mlp(self.final_norm(x_fuse)))
        return out

# ---------------------------------------------------------------------------
# Full GraphGPS‑Mamba Encoder
# ---------------------------------------------------------------------------
class GraphGPSEncoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.hidden_dim = config.get('gps_hidden_dim', 128)
        self.num_blocks = config.get('gps_num_blocks', 4)
        self.dropout = config.get('gps_dropout', 0.1)
        self.use_flat = bool(config.get('use_flat', True))
        # Input dims
        node_in = config.get('gps_node_dim', self.hidden_dim)
        self.node_in = config.get('gps_node_dim', self.hidden_dim)
        self.flat_in = int(config.get('flat_dim', 0))
        flat_in = self.flat_in
        edge_in = config.get('edge_dim', 1)
        inp_dim = self.node_in + (flat_in if (self.use_flat and flat_in > 0) else 0)
        self.out_dim = int(config.get('gps_out_dim') or config.get('out_dim') or 1)
        # Encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(inp_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        self.blocks = nn.ModuleList([
            EnhancedGraphGPSBlock(self.hidden_dim, config)
            for _ in range(self.num_blocks)
        ])
        # Stand‑alone TS encoder
        self.ts_encoder = NodeWiseMamba(self.hidden_dim, config)
        self.lambda_ts = nn.Parameter(torch.ones(1))
        self.lambda_gps = nn.Parameter(torch.ones(1))

        # Fusion head
        self.mlp_in_dim = self.hidden_dim * 2
        self.final_norm = nn.LayerNorm(self.mlp_in_dim)
        self.proj1 = nn.Linear(self.mlp_in_dim, self.hidden_dim)
        self.proj2 = nn.Linear(self.hidden_dim, self.out_dim)
        self.act = nn.GELU()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, edge_index, edge_attr=None, batch=None,
                flat=None, x_hist=None):
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        elif batch.dim() == 2:  # [N,1] → [N]
            batch = batch.view(-1)

        if x.dim() == 3:                # [B_or_T, N, D]
            if x.size(0) == 1:         
                x = x.squeeze(0)        # [N, D]
            else:                      
                x = x[:, -1, :]         # [N, D]

        if flat is not None and flat.dim() == 3:
            if flat.size(0) == 1:
                flat = flat.squeeze(0)
            else:
                flat = flat[:, -1, :]

        if self.use_flat and flat is not None:
            flat = flat.to(x.device)
            if flat.dim() == 1:
                flat = flat.unsqueeze(-1)
            x = torch.cat([x, flat], dim=-1)

        # encode attrs
        h = self.node_encoder(x)
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        # GMB blocks
        for blk in self.blocks:
            h = blk(h, edge_index, edge_attr, batch)
        
        lambdas = torch.cat([self.lambda_ts, self.lambda_gps], dim=0)
        weights = torch.softmax(lambdas, dim=0)
        weight_ts = weights[0]
        weight_gps = weights[1]
        
        if x_hist is not None:
            h_ts = self.ts_encoder(x_hist)
        else:
            h_ts = torch.zeros_like(h)
        
        h_gps = h
        G_pool = global_mean_pool(h_gps, batch)
        h_gps = h_gps + G_pool[batch]
        
        h_ts = weight_ts * h_ts
        h_gps = weight_gps * h_gps
        
        z = torch.cat([h_ts, h_gps], dim=-1)
        z = self.final_norm(z)
        y = self.proj2(self.act(self.proj1(z)))
        return torch.clamp(y, 0.0, 14.0)

# Helper for external scripts      -
def define_graphgps_encoder():
    return GraphGPSEncoder