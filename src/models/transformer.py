# transformer.py
# Transformer LOS regression network

import math
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from src.models.utils import init_weights, get_act_fn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 14 * 24):  # 14 days × 24 h
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))  # (max_len,1,d_model)

    def forward(self, x: torch.Tensor):  # x: (S,B,E)
        return x + self.pe[: x.size(0)]

class MyBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(MyBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        # hack to work around model.eval() issue
        if not self.training:
            self.eval_momentum = 0  # set the momentum to zero when the model is validating

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum if self.training else self.eval_momentum

        if self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum if self.training else self.eval_momentum

        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            training=True, momentum=exponential_average_factor, eps=self.eps)  # set training to True so it calculates the norm of the batch


class MyBatchNorm1d(MyBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))


class EmptyModule(nn.Module):
    def forward(self, X):
        return X
        
class TransformerEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.d_model = cfg["trans_hidden_dim"]
        self.input_dim = cfg["trans_indim"]
        self.ffn_dim = cfg["trans_ffn_dim"]
        self.n_heads = cfg["trans_num_heads"]
        self.n_layers = cfg["trans_layers"]
        self.dropout = cfg["trans_dropout"]
        max_len = cfg.get("max_seq_len", 14 * 24)

        self.input_proj = nn.Linear(self.input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.ffn_dim,
            dropout=self.dropout,
            activation="relu",
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.n_layers)
        self.drop = nn.Dropout(self.dropout)
        init_weights(self.modules())

    def _causal_mask(self, size: int, device):
        mask = torch.triu(torch.ones(size, size, device=device), 1).bool()
        return mask.float().masked_fill(mask, float("-inf"))

    def forward(self, seq):  # seq: (S,B,C)
        S, _, _ = seq.shape
        x = self.input_proj(seq) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.drop(x)
        x = self.encoder(x, mask=self._causal_mask(S, seq.device))
        return x  # (S,B,E)


class TransformerLOSNet(nn.Module):
    """Causal Transformer that outputs (B, out_dim) like lstm.Net."""
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.batchnorm = cfg.get("batchnorm")
        self.momentum = 0.01 if self.batchnorm == "low_momentum" else 0.1
        self.flat_after = cfg.get("flat_after", False)
        self.pooling = cfg.get("trans_pooling", "mean")

        # backbone
        self.transformer = TransformerEncoder(cfg)

        # flat feature path (optional)
        flat_dim_out = 0
        if self.flat_after:
            flat_dim_out = cfg.get("flat_nhid") or cfg["num_flat_feats"]
            self.flat_fc = nn.Linear(cfg["num_flat_feats"], flat_dim_out)
            self.bn_flat = self._make_bn(flat_dim_out)

        # final FC dim
        seq_len = cfg.get("max_seq_len", 14 * 24)
        base_dim = self.transformer.d_model if self.pooling != "all" else self.transformer.d_model * seq_len
        fc_in_dim = base_dim + flat_dim_out

        self.out_layer = nn.Linear(fc_in_dim, cfg["out_dim"])
        self.drop = nn.Dropout(cfg["main_dropout"])
        self.last_act = get_act_fn(cfg["final_act_fn"])

        init_weights(self.modules())

    
    def _make_bn(self, size):
        if self.batchnorm in {"mybatchnorm", "low_momentum"}:
            return MyBatchNorm1d(size, momentum=self.momentum)
        elif self.batchnorm == "default":
            return nn.BatchNorm1d(size)
        else:
            return EmptyModule()

    
    def _pool(self, x):  # x: (B,E,T)
        if self.pooling == "mean":
            return x.mean(dim=2)
        if self.pooling == "max":
            return x.max(dim=2).values
        if self.pooling == "last":
            return x[:, :, -1]
        if self.pooling == "all":
            return x.flatten(start_dim=1)
        raise NotImplementedError

    
    def forward(self, seq, flat):
        x = self.transformer(seq.permute(1, 0, 2)).permute(1, 2, 0)  # (B,E,T)
        emb = self._pool(x)

        if self.flat_after:
            flat_enc = self.bn_flat(self.flat_fc(flat))
            emb = torch.cat([emb, flat_enc], dim=1)

        out = self.out_layer(self.drop(emb))
        return self.last_act(out)

    # helper
    def forward_to_embedding(self, seq, flat):
        x = self.transformer(seq.permute(1, 0, 2)).permute(1, 2, 0)
        return self._pool(x)

def define_transformer_los():
    return TransformerLOSNet
