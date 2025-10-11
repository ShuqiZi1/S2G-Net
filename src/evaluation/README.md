# S²G-Net Evaluation Framework

## Overview

The evaluation framework provides comprehensive analysis components for S²G-Net (Mamba-GPS) including:

1. **Ablation Studies** - Systematic component analysis
2. **Interpretability Analysis** - Model explainability and feature attribution  
3. **Performance Comparison** - Baseline model comparison
4. **Model Checkpoints** - Trained model analysis and inference

## 🛠 Prerequisites

```bash
# Core requirements from main project
mamba-ssm==2.2.2
torch==2.0.0
pytorch-lightning==1.4.9
torch-geometric==2.6.1

# Interpretability packages
shap>=0.43,<0.45
captum>=0.7
networkx>=2.8
umap-learn>=0.5

# Analysis and visualization
matplotlib>=3.3.2,<4.0
seaborn==0.13.2
pandas==1.3.3
numpy==1.24.4
scikit-learn==1.3.2
```

## Quick Start

### 1. Comprehensive Model Comparison

Run all models for systematic comparison:

```bash
# S²G-Net (Main Model)
python -m experiments.train_mamba_gps_enhgraph \
    --model mamba-gps --ts_mask --add_flat --class_weights \
    --add_diag --task los --read_best --with_edge_types

# Individual Components
python -m experiments.train_mamba_only \
    --model mamba --ts_mask --add_flat --add_diag --task los

python -m experiments.train_graphgps_only \
    --model graphgps --ts_mask --add_flat --class_weights \
    --task los --dynamic_g

# Baseline Models
python -m train_ns_lstm \
    --bilstm --ts_mask --add_flat --class_weights \
    --add_diag --task los --read_best

python -m train_ns_lstmgnn \
    --bilstm --ts_mask --add_flat --class_weights \
    --gnn_name gat --add_diag --task los --read_best
```

### 2. Ablation Studies

Run comprehensive ablation experiments:

```bash
# Component ablation analysis (using trained model checkpoint)
python -m experiments.train_mamba_gps_enhgraph \
    --mode ablate \
    --abl static_only \
    --model mamba-gps \
    --task los \
    --add_flat \
    --read_best \
    --load "results/logs/los/tm_flatd/mamba_gps_dynamic/*/checkpoints/epoch=15-step=1439.ckpt"
```

**Available Ablations:**
- `baseline` - Full model performance
- `static_only` - Only static features (no time series)
- `no_static` - Remove static features
- `last6h`, `last24h`, `full48h` - Temporal window analysis  
- `remove_physio`, `remove_vitals` - Feature importance
- `no_mamba`, `no_gps` - Architecture component analysis
- `drop_edges_30`, `drop_edges_50` - Graph structure impact

### 3. Interpretability Analysis

Run comprehensive explainability analysis:

```bash
# Model explainability
python -m experiments.train_mamba_gps_enhgraph \
    --mode explain \
    --model mamba-gps \
    --ts_mask \
    --add_flat \
    --class_weights \
    --add_diag \
    --task los \
    --read_best \
    --load "results/logs/los/tm_flatd/mamba_gps_dynamic/*/checkpoints/epoch=11-step=1079.ckpt"
```

This generates comprehensive interpretability analysis including:
- **Feature Attribution**: SHAP values for temporal and static features
- **Temporal Attention**: Time-step importance analysis
- **Graph Explanations**: Patient similarity and neighborhood influence
- **Model Predictions**: Per-patient prediction breakdowns

### 4. Performance Analysis

```bash
# Monitor training progress with TensorBoard
tensorboard --logdir results/logs/tensorboard_logs

# View aggregated results
cat results/sum_results/summary.csv

# Compare model performance
ls results/baselines/
ls results/mamba_gps/
```

## Key Metrics

Our evaluation framework employs multiple regression metrics specifically chosen for clinical LOS prediction:

- **Primary Metrics**:
  - **R² Score**: Coefficient of determination (variance explained)
  - **MSE**: Mean Squared Error (prediction accuracy)
  - **MSLE**: Mean Squared Logarithmic Error (handles skewed LOS distributions)

- **Clinical Metrics**:
  - **MAD**: Mean Absolute Deviation (robust to outliers)
  - **log-MAPE**: Mean Absolute Percentage Error computed in log space (relative accuracy)
  - **Kappa**: Inter-rater agreement on discretized LOS bins (clinical relevance)

- **Efficiency Metrics**:
  - **Training Time**: Computational efficiency
  - **Peak VRAM**: Memory requirements
  - **Inference Speed**: Real-time prediction capability


## 🔬 Experimental Details

- **Random Seeds**: Multiple seeds for statistical significance
- **Significance Testing**: Paired t-tests with p < 0.05 threshold
- **Confidence Intervals**: Bootstrap sampling (100 iterations)
- **Cross-validation**: Stratified splitting by patient

## Model Checkpoints

Best performing models are automatically saved:

```bash
# S²G-Net checkpoints
results/logs/los/tm_flatd/mamba_gps_dynamic/*/checkpoints/

# Baseline model checkpoints  
results/baselines/*/checkpoints/

# Load for inference or further analysis
python -m experiments.train_mamba_gps_enhgraph \
    --mode test \
    --load "path/to/checkpoint.ckpt"
```

## Configuration

Key configuration patterns:

```yaml
# Model configuration
model_config = {
    "mamba_d_model": 256,
    "mamba_layers": 3,
    "gps_layers": 3,
    "lg_alpha": 0.7,
    "batch_size": 64,
    "learning_rate": 1e-4
}
```

## Reproducibility

All experiments are fully reproducible:

- **Fixed Seeds**: Consistent random seeds for statistical significance
- **Configuration Logging**: Complete parameter settings saved
- **Version Control**: Git commit hashes in metadata
- **Statistical Testing**: Paired t-tests with confidence intervals


## License


This evaluation framework is released under the same license as the main S²G-Net project.
