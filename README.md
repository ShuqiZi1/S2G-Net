# S²G-Net: Bridging Graph and State-Space Modeling for Intensive Care Unit Length of Stay Prediction

S²G-Net (Mamba-GPS) is a novel hybrid architecture that addresses the challenge of ICU Length-of-Stay (LOS) prediction by effectively modeling multivariate temporal characteristics of ICU patients through the fusion of state-space models and graph neural networks.

## 🎯 Research Motivation

### Core Research Questions

- **How can we jointly capture both the temporal dynamics of individual patients and the inter-patient relationships in the ICU length-of-stay (LOS) prediction task?**
    
    Existing methods often focus only on temporal sequences (e.g., RNN, Transformer) or only on static graph relationships (e.g., conventional GNNs), lacking a unified framework for both.

- **How can we construct effective multi-view patient similarity graphs for heterogeneous, multi-modal, and irregularly sampled ICU clinical data?**

    Most clinical graph models today rely on a single view or static graph, and cannot fully utilize diagnostic, semantic, and administrative information together.

- **How can we improve both interpretability and computational efficiency for LOS prediction models in real clinical deployment without sacrificing prediction accuracy?**

    Traditional graph Transformer architectures are computationally expensive (O(N²) complexity) and often lack sufficient interpretability for clinical use.

- **Do multiple modalities (temporal sequences, graph structure, and static features) provide complementary value in ICU LOS prediction, and how can we efficiently fuse them?**

    The authors test this using a dual-path design (state-space sequence modeling + multi-view graph modeling) plus a static-feature branch.

### Clinical Challenge
ICU Length-of-Stay prediction is crucial for:
- **Resource allocation and bed management**
- **Treatment planning and family communication** 
- **Healthcare cost optimization**
- **Clinical decision support**

Traditional approaches often fail to capture both the complex temporal dynamics within individual patient trajectories and the relational patterns across similar patients.

## 🧠 Model Overview

S²G-Net (Mamba-GPS) solves ICU length-of-stay prediction by combining three parts:

- **Time Series Module (Mamba)**: Processes each patient’s first 48 hours of ICU data with a state-space sequence model, handling irregular measurements and capturing long-range trends for stable temporal features.
- **Graph Module (Optimized GraphGPS)**: Learns from multi-view patient-similarity graphs built from diagnoses, clinical text, and administrative data, combining local graph message passing with global Mamba modeling to capture both close and distant patient relationships.
- **Fusion and Prediction**: Merges temporal, relational, and static patient information using learnable weights, then predicts length-of-stay with an additional auxiliary head to boost robustness.

### Key Innovations

1. **Selective State-Space Modeling**: Mamba's linear-complexity architecture efficiently processes long ICU sequences (48 hours) while maintaining sensitivity to critical temporal patterns
2. **Patient Similarity Graphs**: Multi-view graph construction using diagnosis codes and clinical embeddings to model inter-patient relationships
3. **Hybrid Temporal-Relational Fusion**: Novel integration of patient-specific temporal dynamics with population-level relational structures
4. **Clinical-Aware Architecture**: Specifically designed for medical time-series with attention masks, RMS normalization, and clinically-relevant output constraints

## 🛠 Requirements

### System Requirements
- **Python**: 3.8 (required)
- **CUDA**: 11.8 (module load cuda/11.8)
- **Memory**: ≥32GB RAM for full dataset processing
- **GPU**: NVIDIA RTX 3090/4090 or Tesla V100/A100 (8-12GB VRAM)
- **Storage**: ~60GB for datasets and processed files
- **OS**: Linux/macOS (Windows with WSL2)

### Dependencies

```bash
# Load CUDA module (if using HPC environment)
module load cuda/11.8

# Create Python 3.8 environment
conda create -n s2gnet python=3.8
conda activate s2gnet

# Install requirements
pip install -r requirements.txt
```

**Core Requirements** (`requirements.txt`):
```
# State-space models and deep learning
mamba-ssm==2.2.2
torch==2.0.0
torchvision==0.15.1
torchaudio==2.0.1
pytorch-lightning==1.4.9

# Gradient-boosting / classical models
xgboost>=1.7

# Graph neural networks
torch-geometric==2.6.1
torch-scatter==2.1.2
torch-sparse==0.6.18
torch-cluster==1.6.3
torch-spline-conv==1.2.2

# Data processing and analysis
numpy==1.24.4
pandas==1.3.3
scikit-learn==1.3.2
scipy>=1.5.2,<2.0
pyarrow==17.0.0

# Visualization and analysis
matplotlib>=3.3.2,<4.0
seaborn==0.13.2
tqdm>=4.49.0
tabulate==0.9.0

# Interpretability and optimization
shap>=0.43,<0.45
captum>=0.7
networkx>=2.8
umap-learn>=0.5

# Utilities
PyYAML>=5.3.1
ray==1.13.0
rdflib==7.1.3
python-louvain==0.16
tensorboardX==2.6.2.2
```

### Additional Dependencies (Optional)

```bash
# For BERT embeddings and NLP
pip install transformers>=4.21.0

# For efficient similarity search
pip install faiss-cpu  # or faiss-gpu for acceleration

# For database access (MIMIC-IV)
pip install psycopg2-binary>=2.8.0

# For hyperparameter optimization
pip install optuna
```

## 📊 Dataset Access

### MIMIC-IV Database Setup
1. **Obtain Access**: Register at [PhysioNet](https://physionet.org/content/mimiciv/3.1/)
2. **Dataset Version**: MIMIC-IV v3.1 with 94,440 unique ICU admissions
3. **Database Setup**:
   
   **Option A: BigQuery (Limited Scale)**
   - Follow [BigQuery instructions](https://mimic-iv.mit.edu/docs/access/bigquery/)
   - Note: 1GB free tier limit; full preprocessing requires ~4.5GB
   
   **Option B: Local PostgreSQL (Recommended)**
   - Install PostgreSQL and create MIMIC-IV database
   - Use setup scripts from [MIMIC-IV-Postgres](https://github.com/EmmaRocheteau/MIMIC-IV-Postgres)

### Dataset Characteristics
- **Clinical Scope**: Intensive Care Unit admissions with comprehensive monitoring
- **Temporal Coverage**: Up to 5,432 hours per patient (median 48.3 hours)
- **Data Types**: Demographics, diagnoses (6M+ ICD codes), time-series (136M+ chart events, 21M+ lab results)
- **Missing Data**: Systematic patterns (height 56.9%, verbal GCS 11.7%)

### Configuration

Create `paths.json` in project root:

```json
{
    "MIMIC_path": "/path/to/your/MIMIC_data/",
    "data_dir": "/path/to/processed_data/",
    "graph_dir": "/path/to/graphs/",
    "graph_results": "/path/to/graph_results/",
    "log_path": "/path/to/logs/",
    "ray_dir": "/path/to/ray_results/"
}
```

## 📝 Data Processing Pipeline

### Comprehensive Preprocessing Steps

Based on the MIMIC-IV v3.1 dataset, our preprocessing pipeline transforms raw clinical data into structured formats optimized for temporal modeling:

#### 1. Diagnosis Code Processing
- **Hierarchical Parser**: Extracts semantic levels from ICD-9/10 codes
  - ICD-9: Chapter → 3-digit → 4-digit → 5-digit (e.g., 250.01)
  - ICD-10: Chapter → Category → Subcategory → Modifier (e.g., A41.9)
- **String Representation**: `01-Infectious|001|0010|00100` format
- **Sparse Matrix**: Patient-by-diagnosis binary indicators
- **Prevalence Filtering**: Remove diagnoses <0.1% prevalence and redundant branches

#### 2. Static Feature Processing
- **Categorical Encoding**: One-hot encoding with rare categories (<1000 occurrences) grouped as "misc"
- **Numerical Normalization**:
  - Standardization (z-score): height
  - Min-max scaling [-1,1]: weight, age, GCS scores
- **Outlier Handling**: Clipping to [-4, 4] range
- **Missing Value Imputation**: Zero-fill with missing indicators (e.g., `nullheight`)

#### 3. Time Series Processing
- **Data Restructuring**: (patient_id, time_offset) → feature matrix format
- **Temporal Resampling**: Irregular measurements → hourly resolution (mean aggregation)
- **Missing Value Handling**: Forward-fill with decay masks (time since last measurement)
- **Feature Selection**: Remove non-varying or sparse variables
- **Normalization**: 5th-95th percentile scaling, clipped to [-4, 4]
- **Window Standardization**: Consistent 48-hour observation periods

#### 4. Label and Cohort Alignment
- **Outcome Processing**: Length-of-stay from `labels.csv`
- **Patient Filtering**: Include only patients with complete time series data
- **Stratified Splitting**: 70% train / 15% val / 15% test (patient-level stratification)

### Final Output Structure
```
MIMIC_data/
├── train/val/test/
│   ├── diagnoses.csv      # Hierarchical ICD codes (sparse binary matrix)
│   ├── flat.csv          # 42 static features (normalized, encoded)
│   ├── labels.csv        # LOS outcomes
│   ├── stays.txt         # Patient identifiers
│   └── timeseries.csv    # 48-hour hourly clinical measurements
└── preprocessed_*.csv     # Full processed datasets before splitting
```

### Step 1: Database Extraction

Update paths in `MIMIC_preprocessing/create_all_tables.sql`:
- Replace `/MIMIC_preprocessing/` with your full project path
- Replace `/MIMIC_data/` with your data directory path

```bash
# Connect to MIMIC-IV database
psql 'dbname=mimic user=mimicuser options=--search_path=mimiciv'
```

In psql console:
```sql
\i MIMIC_preprocessing/create_all_tables.sql
\q
```

**Duration**: 1-2 hours

### Step 2: Python Preprocessing

```bash
# Complete preprocessing pipeline
python -m MIMIC_preprocessing.run_all_preprocessing

# Or step-by-step:
python -m MIMIC_preprocessing.timeseries
python -m MIMIC_preprocessing.flat_and_labels  
python -m MIMIC_preprocessing.diagnoses
python -m MIMIC_preprocessing.split_train_test
```

**Duration**: 8-12 hours

### Step 3: Graph Construction

#### Step-by-Step Graph Construction

```bash
# 1. Create diagnosis-based graph
python -m graph.graph_construction.create_graph \
    --freq_adjust --penalise_non_shared --k 3 --mode k_closest

# 2. Extract diagnosis text strings
python -m graph.graph_construction.get_diagnosis_strings

# 3. Generate BERT embeddings
python -m graph.graph_construction.bert

# 4. Create BERT-based graph  
python -m graph.graph_construction.create_bert_graph \
    --k 3 --mode k_closest

# 5. Multi-view graph construction (recommended)
python -m graph.graph_construction.run_graph_construction \
    --k_diag 3 --k_bert 1 --use_gpu
```

#### Advanced Graph Construction

```bash
# With graph rewiring techniques
python -m graph.graph_construction.run_graph_construction \
    --k_diag 5 \
    --k_bert 3 \
    --diag_method faiss \
    --rewiring gdc_light \
    --gdc_alpha 0.05 \
    --use_gpu \
    --max_edges 15

# Batch processing multiple configurations
python -m graph.graph_construction.run_graph_construction \
    --batch \
    --batch_k_diag "3,5,10" \
    --batch_k_bert "1,3,5" \
    --batch_diag_methods "tfidf,faiss" \
    --batch_rewiring "none,gdc_light"
```

#### Graph Construction Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--k_diag` | Neighbors for diagnosis graph | 3 | 3, 5, 10 |
| `--k_bert` | Neighbors for BERT graph | 1 | 1, 3, 5 |
| `--diag_method` | Similarity method | `tfidf` | `tfidf`, `faiss`, `penalize` |
| `--rewiring` | Enhancement technique | `none` | `none`, `mst`, `gdc_light` |
| `--use_gpu` | GPU acceleration | False | Flag |
| `--max_edges` | Maximum edges per node | 15 | int |

**Output**: Multi-view graphs with edge types:
- **Type 0**: Diagnosis-based edges (TF-IDF similarity)
- **Type 1**: BERT embedding edges (semantic similarity)  
- **Type 2**: MST edges (connectivity enhancement)
- **Type 3**: GDC edges (diffusion-based)

### Step 4: Convert to Memory-Mapped Format

```bash
python -m src.dataloader.convert
```

**Output Structure**:
```
MIMIC_data/
├── train/val/test/
│   ├── diagnoses.csv      # ICD diagnosis codes
│   ├── flat.csv          # Static features (42 features)
│   ├── labels.csv        # Mortality, length-of-stay  
│   ├── stays.txt         # Patient IDs
│   └── timeseries.csv    # Hourly measurements (48h windows)
```

**Dataset Statistics**:
- **Total ICU Stays (raw)**: 94 440 admissions  
- **Filtered Cohort (age ≥ 18 y, stay ≥ 5 h, complete data)**: **65 347** stays  
- **Split**: 70% train / 15% val / 15% test (stratified by patient)
- **Time Series**: 48-hour observation windows, hourly sampling
- **Static Features**: 42 demographic, admission, and baseline clinical features
- **Diagnoses**: Hierarchical ICD-9/10 codes with prevalence filtering
- **Storage**: ~60GB total for raw and processed data


## ⏳Model Training

### S²G-Net (Main Model)

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

### Baseline Models

**Mamba Only**:
```bash
python -m experiments.train_mamba_only \
    --model mamba \
    --ts_mask \
    --add_flat \
    --add_diag \
    --task los
```

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

### Key Training Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `--model` | Model architecture | `mamba-gps` | Main S²G-Net model |
| `--task` | Prediction task | `los` | Length-of-stay prediction |
| `--ts_mask` | Use temporal masks | False | Recommended for S²G-Net |
| `--add_flat` | Include static features | False | Demographic and clinical data |
| `--add_diag` | Include diagnosis codes | False | ICD code features |
| `--class_weights` | Handle class imbalance | False | For classification tasks |
| `--with_edge_types` | Use multi-type edges | False | Multi-view graph support |
| `--read_best` | Use optimal hyperparameters | False | Load best known config |


## 📈 Evaluation

### Comprehensive Model Comparison

Run all models for systematic comparison:

```bash
# Run all experiments
python -m src.evaluation.tracking.run_experiments

# Aggregate results
python -m src.evaluation.tracking.aggregate --results_dir results

# Generate visualizations
python -m src.evaluation.tracking.tracking_viz --results_dir results
```

### Results Analysis

After running experiments, results are automatically saved in organized directories:

```bash
# View aggregated results
ls results/sum_results/

# Check individual model performance
ls results/baselines/
ls results/mamba_gps/

# Review training logs
ls results/logs/
```

### Ablation Studies

```bash
# Component ablation analysis (using trained model checkpoint)
python -m experiments.train_mamba_gps_enhgraph \
    --mode ablate \
    --abl static_only \
    --model mamba-gps \
    --task los \
    --add_flat \
    --read_best \
    --load <ckpt>

# Additional ablations (replace --abl parameter)
# Available options: baseline, static_only, no_static, last6h, last24h, full48h
# remove_physio, remove_vitals, no_mamba, no_gps, drop_edges_30, drop_edges_50
```

**Available Ablation Studies**:
- `baseline` - Full model performance
- `static_only` - Only static features (no time series)
- `no_static` - Remove static features
- `last6h`, `last24h`, `full48h` - Temporal window analysis
- `remove_physio`, `remove_vitals` - Feature importance analysis
- `no_mamba`, `no_gps` - Architecture component analysis
- `drop_edges_30`, `drop_edges_50` - Graph structure impact analysis

### Interpretability Analysis

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
    --load <ckpt>

# Individual interpretability components
python -m src.evaluation.explain.global_shap --model_path checkpoints/best_model.ckpt
python -m src.evaluation.explain.temporal_attr --model_path checkpoints/best_model.ckpt
python -m src.evaluation.explain.graph_explain --model_path checkpoints/best_model.ckpt
```

### Performance Analysis

```bash
# Monitor training progress with TensorBoard
tensorboard --logdir results/logs/tensorboard_logs

# View aggregated results
cat results/sum_results/summary.csv

# Compare model performance
ls results/baselines/
ls results/mamba_gps/
```

### Model Checkpoints

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


## 📊  Key Metrics

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


## 🧪 Experimental Reproducibility

All experiments are fully reproducible:

- **Fixed Seeds**: Consistent random seeds (21, 22, 23, 24, 25)
- **Statistical Testing**: Paired t-tests with p < 0.05 threshold
- **Confidence Intervals**: Bootstrap sampling (75 iterations)
- **Configuration Logging**: Complete parameter settings saved
- **Version Control**: Git commit hashes in metadata



## 📄 License

This project is licensed under the MIT License. Users must obtain appropriate data use agreements for MIMIC-IV access.



