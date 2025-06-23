# S²G-Net Module Construction

## Overview

S²G-Net (Mamba-GPS) addresses the critical challenge of ICU Length-of-Stay prediction by effectively modeling multivariate temporal characteristics of ICU patients through the fusion of state-space models and graph neural networks. The key innovation lies in the integration of **Mamba** (for efficient long-range sequential dependencies) with **GraphGPS** (for expressive graph representation learning).

## Model Architecture

### Core Components

**S²G-Net (Mamba-GPS)** consists of three main components:

1. **Time Series Module (Mamba)**: A state-space sequence encoder designed to handle patient-level multivariate time series data over 48-hour ICU windows
2. **Graph Module (Optimized GraphGPS)**: Combines GENConv-based graph convolution for local message passing with node-wise Mamba sequence modeling
3. **Fusion and Prediction**: Integrates temporal and relational embeddings via learnable parameters, incorporating demographic and diagnostic features

```
Time Series → Mamba Encoder → Projection → GraphGPS-Mamba → Fusion → LOS Predictions
    ↓                                           ↑
Static Features ────────────────────────────────┘
```

## Baseline Models

We provide implementations of several strong baseline methods for comprehensive comparison:

### Sequential Baselines
- **BiLSTM** (`train_ns_lstm`): Bidirectional LSTM with various pooling strategies
- **Mamba** (`experiments.train_mamba_only`): State-space model with selective mechanisms and RMS normalization

### Graph Neural Networks
- **GCN** (`train_ns_gnn`): Graph Convolutional Networks
- **GAT** (`train_ns_gnn`, `train_ns_lstmgnn`): Graph Attention Networks with multi-head attention
- **SAGE** (`train_ns_gnn`): GraphSAGE with neighborhood sampling
- **MPNN** (`train_ns_gnn`, `train_ns_lstmgnn`): Message Passing Neural Networks

### Hybrid Approaches
- **LSTM-GNN** (`train_ns_lstmgnn`): Sequential LSTM followed by GNN processing
- **Dynamic LSTM-GNN** (`train_dynamic`): k-NN graph construction with LSTM-GNN
- **GraphGPS** (`experiments.train_graphgps_only`): Transformer-based approach with dynamic graph updates

## S²G-Net: Detailed Architecture

### 1. Time Series Encoding (Mamba)

Our Mamba implementation features:
- **RMS Normalization**: Improved gradient flow and training stability
- **Selective State Space**: Efficient modeling of long sequences (48 hours)
- **Multiple Pooling Strategies**: Last timestep hidden state selection
- **Residual Connections**: Enhanced information flow across layers

**Input**: Patient multivariate time series X^(i) ∈ R^(T×d) (T=48 hours)
**Output**: Temporal embedding h_ts^(i) ∈ R^(d_model)

```python
class MambaEncoder(nn.Module):
    def __init__(self, config):
        # Core Mamba layers with selective SSM
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=self.d_model, d_state=d_state, 
                  d_conv=d_conv, expand=expand)
            for _ in range(self.n_layers)
        ])
```

### 2. Graph Processing with GraphGPS-Mamba

GraphGPS-Mamba combines:
- **Local Message Passing**: GENConv for neighborhood aggregation
- **Global Attention**: Mamba-based processing of degree-ordered node sequences
- **Multi-scale Fusion**: Adaptive weighting of local and global representations

Key innovations:
- **Node-wise Mamba**: Processes node features ordered by node degree
- **Graph-Mamba Block**: Integrates local graph convolution with global Mamba processing
- **Adaptive Fusion**: Learnable combination of time-series and graph representations

### 3. S²G-Net Integration

The main S²G-Net model orchestrates the complete pipeline:

```python
class MambaGraphGPS(nn.Module):
    def forward(self, x, flat, edge_index, batch_size, edge_attr=None):
        # 1. Sequential processing
        mamba_out, _ = self.mamba_encoder(seq, masks)
        
        # 2. Project to graph space
        gps_input = self.mamba_to_gps(mamba_out)
        
        # 3. Graph processing
        gps_out = self.graphgps_encoder(gps_input, edge_index, edge_attr)
        
        # 4. Multi-modal fusion
        combined = torch.cat([gps_out, flat], dim=1)
        y = self.out_layer(combined)
        
        return y, mamba_out  # Dual predictions
```

#### Key Features:

- **Dual-Stream Architecture**: Separate pathways for sequential and graph information
- **Adaptive Projection**: Learned mapping between Mamba and GraphGPS representations  
- **Multi-Modal Fusion**: Integration of temporal, structural, and static features
- **Dual Predictions**: Both fused and sequence-only predictions for enhanced learning

## Model Configurations

### Recommended Hyperparameters

**S²G-Net (Small)**:
```python
config = {
    'mamba_d_model': 128,
    'mamba_layers': 3,
    'gps_layers': 3,  
    'gps_hidden_dim': 64,
    'mamba_dropout': 0.1,
    'gps_dropout': 0.1
}
```

**S²G-Net (Large)**:
```python
config = {
    'mamba_d_model': 512,
    'mamba_layers': 6,
    'gps_layers': 6,
    'gps_hidden_dim': 256,
    'mamba_dropout': 0.15,
    'gps_dropout': 0.15
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.