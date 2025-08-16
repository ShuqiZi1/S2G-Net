"""
Defining RNN models
"""
import torch
import torch.nn as nn
from src.models.utils import init_weights, get_act_fn


def define_rnn_encoder():
    """Return the DynamicRNN class so the outer wrapper can instantiate it."""
    return DynamicRNN


class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rnn_encoder = DynamicRNN(config)

        # --- flat-feature branch (unchanged) ---
        self.flat_after = config['flat_after']
        outdim_key  = 'rnn_outdim' if 'rnn_outdim' in config else 'lstm_outdim'
        fc_in_dim   = config[outdim_key]
        if self.flat_after:
            flat_dim = (
                config['flat_nhid']
                if config['flat_nhid'] is not None
                else config['num_flat_feats']
            )
            self.flat_fc = nn.Linear(config['num_flat_feats'], flat_dim)
            fc_in_dim += flat_dim

        # --- head ---
        self.out_layer = nn.Linear(fc_in_dim, config['out_dim'])
        self.drop = nn.Dropout(config['main_dropout'])
        self.last_act = get_act_fn(config['final_act_fn'])

        self._initialize_weights()

    def _initialize_weights(self):
        init_weights(self.modules())

    # Convenience when you only need RNN embeddings
    def forward_to_rnn(self, seq, flat):
        # seq: (batch, seq_len, feat)  → RNN expects (seq_len, batch, feat)
        seq = seq.permute(1, 0, 2)
        out, _ = self.rnn_encoder(seq)
        out = out.view(out.size(0), -1)  # (batch, rnn_outdim)
        return out

    def forward(self, seq, flat):
        seq = seq.permute(1, 0, 2)  # (seq_len, batch, feat)
        out, _ = self.rnn_encoder(seq)
        out = out.view(out.size(0), -1)  # (batch, rnn_outdim)

        if self.flat_after:
            flat = self.flat_fc(flat)         # (batch, flat_dim)
            out = torch.cat([out, flat], dim=1)

        out = self.out_layer(self.drop(out))
        out = self.last_act(out)
        return out


class DynamicRNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        alias = {
            'rnn_indim'   : 'lstm_indim',
            'rnn_nhid'    : 'lstm_nhid',
            'rnn_layers'  : 'lstm_layers',
            'rnn_dropout' : 'lstm_dropout',
            'rnn_pooling' : 'lstm_pooling',
            'birnn'       : 'bilstm',
            'rnn_outdim'  : 'lstm_outdim'
        }
        for new_key, old_key in alias.items():
            if new_key not in config and old_key in config:
                config[new_key] = config[old_key]

        self.n_layers       = config['rnn_layers']
        self.dropout_rate   = config['rnn_dropout']
        self.pooling        = config['rnn_pooling']
        self.is_bidirectional = config['birnn']   # bool

        # hidden size per direction
        if self.is_bidirectional:
            self.num_units = config['rnn_nhid'] // 2
            self.num_dir   = 2
        else:
            self.num_units = config['rnn_nhid']
            self.num_dir   = 1

        self.drop = nn.Dropout(self.dropout_rate)

        self.rnn = nn.RNN(
            input_size  = config['rnn_indim'],
            hidden_size = self.num_units,
            num_layers  = self.n_layers,
            dropout     = self.dropout_rate if self.n_layers > 1 else 0.0,
            bidirectional = self.is_bidirectional,
            nonlinearity = config.get('rnn_nonlinearity', 'tanh')  # 'tanh' | 'relu'
        )

        # buffer for hidden state
        self.hidden = self.init_hidden(config['batch_size'])


    # fresh zero-state for every new mini-batch
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return weight.new_zeros(
            self.n_layers * self.num_dir, batch_size, self.num_units
        )

    def forward(self, rnn_input):
        """
        Args
        ----
        rnn_input : Tensor
            Shape (seq_len, batch, feat_dim)
        Returns
        -------
        rnn_out  : Tensor
            Shape depends on pooling:
            - 'all'  : (batch, seq_len, hid_dim)
            - others : (batch, hid_dim)
        attention : None (placeholder for API parity with LSTM version)
        """
        batch_size = rnn_input.size(1)
        self.hidden = self.init_hidden(batch_size)

        rnn_out, _ = self.rnn(rnn_input, self.hidden)  # (seq_len, batch, hid_dim)
        rnn_out = torch.transpose(rnn_out, 0, 1).contiguous()  # (batch, seq_len, hid_dim)

        # ----- pooling -----
        if self.pooling == 'mean':
            rnn_out = torch.mean(rnn_out, dim=1)
        elif self.pooling == 'max':
            rnn_out = torch.max(rnn_out, dim=1)[0]
        elif self.pooling == 'last':
            if self.is_bidirectional:
                # concat last step of forward dir + first step of backward dir
                rnn_out = torch.cat(
                    (rnn_out[:, -1, :self.num_units], rnn_out[:, 0, self.num_units:]),
                    dim=1
                )
            else:
                rnn_out = rnn_out[:, -1, :]
        elif self.pooling == 'all':
            pass  # keep full sequence
        else:
            raise NotImplementedError("pooling must be mean | max | last | all")

        attention = None
        return rnn_out, attention
