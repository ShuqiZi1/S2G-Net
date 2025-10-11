# S²G-Net Training Guide

## Overview

This directory contains the training scripts for the S²G-Net (Mamba-GPS) framework and baseline models. The framework supports multiple model architectures for ICU Length-of-Stay prediction tasks.

## Quick Start

### Basic Training

Train the S²G-Net (Mamba-GPS) model:

```bash
python -m experiments.train_mamba_gps_enhgraph \
    --model mamba-gps \
    --ts_mask \
    --add_flat \
    --class_weights \
    --add_diag \
    --task los \
    --read_best \
    --with_edge_types
```

### Model Evaluation

Evaluate a trained model:

```bash
python -m experiments.train_mamba_gps_enhgraph \
    --mode test \
    --load "results/logs/los/*/checkpoints/best.ckpt" \
    --task los
```

## Supported Models

### 1. S²G-Net (Recommended)

**S²G-Net** - The main S²G-Net architecture with advanced spatio-temporal modeling:

```bash
# Standard training
python -m experiments.train_mamba_gps_enhgraph \
    --model mamba-gps \
    --ts_mask \
    --add_flat \
    --class_weights \
    --add_diag \
    --task los \
    --read_best \
    --with_edge_types

# Custom hyperparameters
python -m experiments.train_mamba_gps_enhgraph \
    --model mamba-gps \
    --ts_mask \
    --add_flat \
    --add_diag \
    --task los \
    --mamba_d_model 256 \
    --gps_layers 3 \
    --epochs 100 \
    --lr 5e-4 \
    --batch_size 32
```

### 2. Component Models

**Mamba Only**:
```bash
python -m experiments.train_mamba_only \
    --model mamba \
    --ts_mask \
    --add_flat \
    --add_diag \
    --task los
```

**GraphGPS Only**:
```bash
python -m experiments.train_graphgps_only \
    --model graphgps \
    --ts_mask \
    --add_flat \
    --class_weights \
    --task los \
    --dynamic_g
```

### 3. Baseline Models


**RNN Baseline**:
```bash
python -m experiments.train_ns_lstm \
    --model rnn \
    --ts_mask \
    --add_flat \
    --class_weights \
    --num_workers 0 \
    --add_diag \
    --task los \
    --read_best
```

**Transformer Baseline**:
```bash
python -m experiments.train_ns_transformer \
    --model transformer \
    --ts_mask \
    --add_flat \
    --class_weights \
    --num_workers 0 \
    --add_diag \
    --task los \
    --read_best
```

**BiLSTM Baseline**:
```bash
python -m experiments.train_ns_lstm \
    --bilstm \
    --ts_mask \
    --add_flat \
    --class_weights \
    --num_workers 0 \
    --add_diag \
    --task los \
    --read_best
```
**GNN Models** (with neighborhood sampling):
```bash
# GNN with GAT
python -m experiments.train_ns_gnn \
    --ts_mask \
    --add_flat \
    --class_weights \
    --gnn_name gat \
    --add_diag \
    --task los \
    --read_best

# GNN with SAGE  
python -m experiments.train_ns_gnn \
    --ts_mask \
    --add_flat \
    --class_weights \
    --gnn_name sage \
    --add_diag \
    --task los \
    --read_best

# GNN with MPNN
python -m experiments.train_ns_gnn \
    --ts_mask \
    --add_flat \
    --class_weights \
    --gnn_name mpnn \
    --add_diag \
    --task los \
    --read_best
```

**LSTM-GNN Models**:
```bash
# LSTM-GNN with GAT
python -m experiments.train_ns_lstmgnn \
    --bilstm \
    --ts_mask \
    --add_flat \
    --class_weights \
    --gnn_name gat \
    --add_diag \
    --task los \
    --read_best

# LSTM-GNN with SAGE
python -m experiments.train_ns_lstmgnn \
    --bilstm \
    --ts_mask \
    --add_flat \
    --class_weights \
    --gnn_name sage \
    --add_diag \
    --task los \
    --read_best

# LSTM-GNN with MPNN
python -m experiments.train_ns_lstmgnn \
    --bilstm \
    --ts_mask \
    --add_flat \
    --class_weights \
    --gnn_name mpnn \
    --add_diag \
    --task los \
    --read_best
```

**Dynamic LSTM-GNN Models**:
```bash
# Dynamic LSTM-GNN with GCN
python -m experiments.train_dynamic \
    --bilstm \
    --random_g \
    --ts_mask \
    --add_flat \
    --class_weights \
    --gnn_name gcn \
    --task los \
    --read_best

# Dynamic LSTM-GNN with GAT
python -m experiments.train_dynamic \
    --bilstm \
    --random_g \
    --ts_mask \
    --add_flat \
    --class_weights \
    --gnn_name gat \
    --task los \
    --read_best

# Dynamic LSTM-GNN with MPNN
python -m experiments.train_dynamic \
    --bilstm \
    --random_g \
    --ts_mask \
    --add_flat \
    --class_weights \
    --gnn_name mpnn \
    --task los \
    --read_best
```

**Traditional Machine Learning Model**:
```
# XGBoost 
python -m experiments.train_xgb \
    --model xgboost \
    --ts_mask \
    --add_flat \
    --add_diag \
    --task los 
```

## Configuration Parameters

### Core Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--model` | Model architecture | `mamba-gps` | `mamba-gps`, `mamba`, `graphgps` |
| `--task` | Prediction task | `los` | `los` (length-of-stay) |
| `--epochs` | Training epochs | 100 | Any positive integer |
| `--lr` | Learning rate | 5e-4 | Any positive float |
| `--batch_size` | Batch size | 32 | Any positive integer |
| `--seed` | Random seed | 42 | Any integer |

### Model Architecture

| Parameter | Description | Default | S²G-Net | Components |
|-----------|-------------|---------|---------|------------|
| `--mamba_d_model` | Mamba hidden dimension | 256 | ✓ | Mamba only |
| `--gps_layers` | GraphGPS layers | 3 | ✓ | GraphGPS |
| `--gnn_name` | GNN architecture | `gat` | ✓ | GNN models |
| `--lg_alpha` | Loss mixing coefficient | 0.3 | ✓ | Hybrid models |

### Data Processing

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `--ts_mask` | Use temporal masks | False | Recommended for S²G-Net |
| `--add_flat` | Include static features | False | Demographic and clinical data |
| `--add_diag` | Include diagnosis codes | False | ICD code features |
| `--class_weights` | Handle class imbalance | False | For classification tasks |
| `--with_edge_types` | Use multi-type edges | False | Multi-view graph support |
| `--read_best` | Use optimal hyperparameters | False | Load best known config |

### Training Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--gpus` | Number of GPUs | 1 |
| `--num_workers` | DataLoader workers | 4 |
| `--use_amp` | Mixed precision training | False |
| `--clip_grad` | Gradient clipping value | 1.0 |
| `--l2` | Weight decay | 1e-4 |
| `--sch` | Learning rate scheduler | `plateau` |

## Hardware Recommendations

For optimal performance, we recommend:
- **GPU**: NVIDIA RTX 3090/4090 or Tesla V100/A100
- **RAM**: 32+ GB
- **Storage**: SSD for datasets (~60GB)
- **VRAM**: 8-12GB for S²G-Net, 4-8GB for baselines

## License


This training framework is released under the same license as the main S²G-Net project.
