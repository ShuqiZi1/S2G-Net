# S²G-Net Graph Construction

## Overview

This module implements the multi-view graph construction pipeline for S²G-Net (Mamba-GPS), supporting diagnosis-based, BERT embedding-based, and hybrid graph generation with advanced rewiring techniques.

The graph construction pipeline creates heterogeneous patient similarity graphs by combining multiple data modalities:

- **Diagnosis-based graphs**: Constructed from ICD diagnosis codes using TF-IDF similarity or FAISS-based approximate nearest neighbors
- **BERT embedding graphs**: Built from clinical text embeddings using efficient KNN search
- **Multi-view fusion**: Intelligent merging of different graph views with type-aware edge selection
- **Graph rewiring**: Optional connectivity enhancement using Minimum Spanning Trees (MST) or Graph Diffusion Convolution (GDC)

## Dependencies

```bash
# Core requirements
numpy==1.24.4
pandas==1.3.3
scipy>=1.5.2,<2.0
torch==2.0.0
transformers>=4.21.0
tqdm>=4.49.0
faiss-cpu  # or faiss-gpu for acceleration
networkx>=2.8
matplotlib>=3.3.2,<4.0
```

## Module Structure

```
graph/
├── graph_construction/
│   ├── bert.py                    # BERT embedding generation
│   ├── create_bert_graph.py       # BERT-based graph construction
│   ├── create_graph.py            # Diagnosis-based graph construction
│   ├── create_graph_gps.py        # Multi-view GraphGPS graph construction
│   ├── knn_utils.py              # Efficient KNN utilities with FAISS
│   ├── get_diagnosis_strings.py   # Diagnosis text preprocessing
│   └── run_graph_construction.py  # Main execution pipeline
└── README.md
```

## Configuration

Create a `paths.json` file in your project root:

```json
{
  "MIMIC_path": "/path/to/your/MIMIC_data/",
  "graph_dir": "/path/to/graphs/",
  "graph_results": "/path/to/graph_results/"
}
```

## Usage

### Step-by-Step Graph Construction

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

### Advanced Graph Construction

The main pipeline supporting multiple data modalities and advanced graph techniques:

```bash
# Basic multi-view construction
python -m graph.graph_construction.run_graph_construction \
    --k_diag 5 \
    --k_bert 3 \
    --diag_method tfidf \
    --max_edges 15

# With graph rewiring
python -m graph.graph_construction.run_graph_construction \
    --k_diag 5 \
    --k_bert 3 \
    --diag_method faiss \
    --rewiring gdc_light \
    --gdc_alpha 0.05 \
    --use_gpu

# Debug mode
python -m graph.graph_construction.run_graph_construction \
    --k_diag 3 \
    --k_bert 1 \
    --debug
```

### Batch Processing

Process multiple graph configurations automatically:

```bash
python -m graph.graph_construction.run_graph_construction \
    --batch \
    --batch_k_diag "3,5,10" \
    --batch_k_bert "1,3,5" \
    --batch_diag_methods "tfidf,faiss" \
    --batch_rewiring "none,gdc_light"
```

## Parameters

### Core Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--k_diag` | Number of neighbors for diagnosis graph | 3 | 3, 5, 10 |
| `--k_bert` | Number of neighbors for BERT graph | 1 | 1, 3, 5 |
| `--diag_method` | Diagnosis similarity method | `tfidf` | `tfidf`, `faiss`, `penalize` |
| `--max_edges` | Maximum edges per node | 15 | int |
| `--batch_size` | Processing batch size | 1000 | int |

### Graph Enhancement

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--rewiring` | Graph rewiring method | `none` | `none`, `mst`, `gdc_light` |
| `--score_transform` | Edge score transformation | `zscore` | `zscore`, `log1p`, `none` |
| `--gdc_alpha` | GDC diffusion parameter | 0.05 | float |

### Computational

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--use_gpu` | Enable GPU acceleration | False |
| `--debug` | Use subset of data | False |

## Output Files

### Graph Files

The pipeline generates multiple output formats in `{graph_dir}/`:

```
graphs/
├── bert_out.npy                          # BERT embeddings
├── gps_k5_3_tf_gdcl_u.npy               # Edge source indices
├── gps_k5_3_tf_gdcl_v.npy               # Edge target indices  
├── gps_k5_3_tf_gdcl_scores.npy          # Edge weights
├── gps_k5_3_tf_gdcl_types.npy           # Edge types
├── gps_k5_3_tf_gdcl_config.json         # Construction configuration
└── gps_k5_3_tf_gdcl_sym_*.npy           # Symmetric versions
```

### Analysis

Results are organized in `{graph_results}/`:

```
results/graphs/
├── graph_analysis/
│   ├── analysis_5_3_tfidf_gdc_light.txt     # Detailed graph analysis
│   ├── graph_stats_5_3_tfidf_gdc_light.json # Summary statistics
│   ├── config_5_3_tfidf_gdc_light.json      # Reproducibility config
│   └── batch_results.json                    # Batch processing results
```

## Edge Types

The multi-view graphs use integer edge types:

- **Type 0**: Diagnosis-based edges (TF-IDF or co-occurrence similarity)
- **Type 1**: BERT embedding edges (semantic similarity)
- **Type 2**: MST edges (connectivity enhancement)
- **Type 3**: GDC edges (diffusion-based)

## Graph Properties

### Typical Statistics (MIMIC-IV)

| Metric | Value Range |
|--------|-------------|
| Nodes | 94,440 patients |
| Edges | 150K-500K (depending on k values) |
| Average degree | 6-20 |
| Largest component | >99% of nodes |
| Edge score range | Z-normalized [-3, 3] |

### Quality Metrics

- **Connectivity**: >99% of nodes in largest connected component
- **Homophily**: Diagnosis-based edges show strong clinical similarity
- **Diversity**: Multi-view edges capture complementary relationships
- **Scalability**: Sub-linear scaling with FAISS approximation

## Performance Optimization

### Memory Management

- **Batch processing**: Configurable batch sizes for large datasets
- **Sparse matrices**: Efficient storage for large graphs
- **GPU acceleration**: FAISS GPU support for embedding similarity

### Computational Efficiency

- **FAISS indexing**: Sub-linear KNN search complexity
- **TF-IDF caching**: Precomputed similarity matrices
- **Parallel processing**: Multi-threaded BERT inference

## License

This module is released under the MIT License. See LICENSE file for details.