# S²G-Net Module Construction

## Overview

S²G-Net (Mamba-GPS) addresses the critical challenge of ICU Length-of-Stay prediction by effectively modeling multivariate temporal characteristics of ICU patients through the fusion of state-space models and graph neural networks. The key innovation lies in the integration of **Mamba** (for efficient long-range sequential dependencies) with **GraphGPS** (for expressive graph representation learning).

## Model Architecture

### Core Components

**S²G-Net (Mamba-GPS)** consists of three main components:

1. **Time Series Module (Mamba)**: Processes each patient’s first 48 hours of ICU data with a state-space sequence model, handling irregular measurements and capturing long-range trends for stable temporal features.
2. **Graph Module (Optimized GraphGPS)**: Learns from multi-view patient-similarity graphs built from diagnoses, clinical text, and administrative data, combining local graph message passing with global Mamba modeling to capture both close and distant patient relationships.
3. **Fusion and Prediction**: Merges temporal, relational, and static patient information using learnable weights, then predicts length-of-stay with an additional auxiliary head to boost robustness.

```
Time Series ──► Mamba Temporal Encoder ──► Temporal Embedding ─┐
                                                               │
Static Features ──► Static Feature Encoder ──► Static Embedding ├─► Weighted Fusion ─► LOS Prediction
                                                               │
Patient Similarity Graph (from multi-view construction: diagnosis, semantic, admin)
        └─► GraphGPS-Mamba Graph Encoder ──► Graph Embedding ──┘

```

## Baseline Models

We provide implementations of several strong baseline methods for comprehensive comparison:

### Sequential Baselines
- **RNN** (`train_ns_lstm`): Standard recurrent neural network baseline for sequential modeling, providing a simpler architecture for temporal feature extraction
- **BiLSTM** (`train_ns_lstm`): Bidirectional LSTM with various pooling strategies
- **Transformer** (`train_ns_transformer`): Self-attention–based encoder adapted for clinical time-series, capable of modeling long-range dependencies but with quadratic complexity in sequence length
- **Mamba** (`experiments.train_mamba_only`): State-space model with selective mechanisms and RMS normalization

### Graph Neural Networks
- **GCN** (`train_ns_gnn`): Graph Convolutional Networks
- **GAT** (`train_ns_gnn`): Graph Attention Networks with multi-head attention
- **SAGE** (`train_ns_gnn`): GraphSAGE with neighborhood sampling
- **MPNN** (`train_ns_gnn`): Message Passing Neural Networks

### Hybrid Approaches
- **LSTM-GNN** (`train_ns_lstmgnn`): Sequential LSTM followed by GNN processing
- **Dynamic LSTM-GNN** (`train_dynamic`): k-NN graph construction with LSTM-GNN
- **GraphGPS** (`experiments.train_graphgps_only`): Transformer-based approach with dynamic graph updates

### Traditional Machine Learning Model
- **XGBoost** (`train_xgb`): Gradient Boosted Decision Tree model used as a classical, non-neural baseline

## License

This project is licensed under the MIT License - see the LICENSE file for details.
