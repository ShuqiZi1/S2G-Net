"""
Mamba encoder implementation
"""
import torch
import torch.nn as nn
from mamba_ssm import Mamba

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        # x: [batch, seq, dim]
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.scale
        
def define_mamba_encoder():
    """Return the Mamba encoder implementation"""
    return MambaEncoder


class MambaEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        # Core dimensions
        self.d_model     = config.get('mamba_d_model', 256)
        self.n_layers    = config.get('mamba_layers',   4)
        self.dropout_rate= config.get('mamba_dropout',   0.15)
        self.pooling     = config.get('mamba_pooling', 'last')
        
        # Input projection
        self.input_dim = config.get('mamba_indim', 1)
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.d_model),
            RMSNorm(self.d_model, eps=1e-5),
            nn.GELU()
        )
        
        # Mamba configuration
        d_state = config.get('mamba_d_state', 64)
        d_conv = config.get('mamba_d_conv', 4)
        expand = config.get('mamba_expand', 2)
        
        
        self.layer_norms = nn.ModuleList([
            RMSNorm(self.d_model, eps=1e-5) for _ in range(self.n_layers)
        ])
        
        # Mamba layers
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=self.d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(self.n_layers)
        ])
        
        # Dropout layers
        self.drops = nn.ModuleList([
            nn.Dropout(self.dropout_rate) for _ in range(self.n_layers)
        ])
        
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(self.d_model, eps=1e-5)
        
        if self.pooling == 'last':
            self.final_cell = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model, eps=1e-5),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model, eps=1e-5)
            )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, flat=None, mask=None):
        """
        Forward pass through the Mamba model
        
        Args:
            x: [batch_size, seq_len, input_dim]
            
        Returns:
            Sequence representation based on pooling strategy
            None (placeholder for compatibility with LSTM interface)
        """
        # Check for numerical issues in input
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # Project input
        x = self.input_proj(x)
        
        residual = x
        for i, (mamba, norm, drop) in enumerate(zip(
                self.mamba_layers, self.layer_norms, self.drops)):
        
            x_mamba = mamba(norm(residual))
            residual = residual + drop(x_mamba)
        
        # Final output is the residual path
        x = self.final_norm(residual)
        
        # Apply pooling strategy
        if self.pooling == 'last':
            if mask is None:
                out = x[:, -1, :]
            else:
                # mask : [B,T]  (float 0/1) 
                last_idx = mask.sum(dim=1).long().clamp(min=1) - 1          # [B]
                out = x[torch.arange(x.size(0)), last_idx, :]               # [B,d]
        elif self.pooling == 'mean':
            if mask is None:
                out = x.mean(dim=1)
            else:
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
                out = (x * mask.unsqueeze(-1)).sum(dim=1) / denom
        elif self.pooling == 'max':
            # Max pooling
            x_safe = torch.nan_to_num(x, nan=-1e9)
            out = torch.max(x_safe, dim=1)[0]
        elif self.pooling == 'all':
            # Return full sequence
            out = x
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")
        
        # Final NaN check on output
        if torch.isnan(out).any() or torch.isinf(out).any():
            out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return out, None  # Return None as second output for API compatibility with LSTM