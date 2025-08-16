"""
Efficient approximate KNN implementations for graph construction
"""
import numpy as np
import torch
import faiss
import os
from scipy import sparse
from typing import Tuple, List, Optional, Union, Dict
import time
from pathlib import Path
import faiss
faiss.omp_set_num_threads(4)
print("FAISS using", faiss.omp_get_max_threads(), "threads")


def save_edge_list(u: np.ndarray, v: np.ndarray, 
                   scores: Optional[np.ndarray] = None,
                   edge_types: Optional[np.ndarray] = None,
                   prefix: str = "", graph_dir: str = None) -> None:
    """Save edge list in COO format with optional scores and types"""
    if graph_dir is None:
        print("No graph_dir specified, not saving")
        return
    
    os.makedirs(graph_dir, exist_ok=True)
    
    # Save edges
    np.save(f"{graph_dir}/{prefix}u.npy", u.astype(np.int32))
    np.save(f"{graph_dir}/{prefix}v.npy", v.astype(np.int32))
    
    # Save scores if provided
    if scores is not None:
        np.save(f"{graph_dir}/{prefix}scores.npy", scores.astype(np.float32))
    
    # Save edge types if provided
    if edge_types is not None:
        np.save(f"{graph_dir}/{prefix}types.npy", edge_types.astype(np.int32))
    
    # Also save in TXT format for backward compatibility
    np.savetxt(f"{graph_dir}/{prefix}u.txt", u.astype(np.int32), fmt='%i')
    np.savetxt(f"{graph_dir}/{prefix}v.txt", v.astype(np.int32), fmt='%i')
    if scores is not None:
        np.savetxt(f"{graph_dir}/{prefix}scores.txt", scores.astype(np.float32), fmt='%.6f')
    if edge_types is not None:
        np.savetxt(f"{graph_dir}/{prefix}types.txt", edge_types.astype(np.int32), fmt='%i')


def build_faiss_index(features: np.ndarray, 
                      metric: str = 'l2',
                      index_type: str = 'hnsw',
                      use_gpu: bool = True) -> faiss.Index:
    """Build FAISS index for fast nearest neighbor search
    
    Args:
        features: Feature vectors to index (N, dim)
        metric: Distance metric ('l2' or 'ip' for inner product)
        index_type: Type of index ('flat', 'hnsw', 'ivf')
        use_gpu: Whether to use GPU for indexing
        
    Returns:
        FAISS index
    """
    features = np.ascontiguousarray(features, dtype=np.float32)
    d = features.shape[1]
    
    # Build appropriate index type
    if index_type == 'flat':
        if metric == 'l2':
            index = faiss.IndexFlatL2(d)
        else:  # metric == 'ip'
            index = faiss.IndexFlatIP(d)
    elif index_type == 'hnsw':
        index = faiss.IndexFlatIP(d)
        if metric == 'ip':
            index.metric_type = faiss.METRIC_INNER_PRODUCT
    elif index_type == 'ivf':
        nlist = min(4096, features.shape[0] // 30)  # num of voronoi cells
        if metric == 'l2':
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        else:  # metric == 'ip'
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(features)
    else:
        raise ValueError(f"Unsupported index type: {index_type}")
    
    # GPU conversion for supported index types
    if use_gpu and faiss.get_num_gpus() > 0:
        if isinstance(index, faiss.IndexHNSWFlat):
            print("Warning: HNSW indexes cannot be moved to GPU. Using CPU version.")
        else:
            print(f"Using GPU for FAISS indexing ({faiss.get_num_gpus()} GPUs available)")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
    
    # Add vectors to index
    index.add(features)

    return index


def compute_knn_graph(features: np.ndarray,
                      k: int = 10,
                      metric: str = 'l2',
                      batch_size: int = 10000,
                      return_distances: bool = True,
                      index_type: str = 'hnsw',
                      self_loops: bool = False,
                      use_gpu: bool = True) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Compute approximate k-nearest neighbors graph using FAISS
    
    Args:
        features: Feature vectors (N, dim)
        k: Number of neighbors per node
        metric: Distance metric ('l2' or 'ip')
        batch_size: Batch size for queries
        return_distances: Whether to return distances
        index_type: Type of FAISS index
        self_loops: Whether to include self loops
        use_gpu: Whether to use GPU
        
    Returns:
        u, v: Source and target node indices
        scores: Edge scores (optional)
    """
    features = np.ascontiguousarray(features, dtype=np.float32)
    N = features.shape[0]
    start_time = time.time()
    
    # Adjust k if including self loops
    k_search = k if self_loops else k + 1
    
    # Build index
    print(f"Building {index_type} index with {metric} metric for {N} vectors...")
    index = build_faiss_index(features, metric, index_type, use_gpu)
    print(f"Index built in {time.time() - start_time:.2f}s")
    
    # Process rows and distances consistently
    all_rows = []
    all_cols = []
    all_scores = []
    
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        batch = features[i:end_i]
        
        # Search kNN
        D, I = index.search(batch, k_search)
        
        # Process each node in batch
        for j in range(end_i - i):
            node_idx = i + j
            neighbors = I[j]
            distances = D[j]
            
            # Filter self-loops and keep only k neighbors
            if not self_loops:
                mask = neighbors != node_idx
                neighbors = neighbors[mask][:k]
                distances = distances[mask][:k]
            
            if neighbors.size == 0:
                neighbors  = np.array([node_idx])
                distances  = np.array([0.0])

            # Add to results
            all_rows.extend([node_idx] * len(neighbors))
            all_cols.extend(neighbors)
            
            if return_distances:
                all_scores.extend(distances)
        
        if (i // batch_size) % 10 == 0:
            print(f"Processed {end_i}/{N} vectors...")
    
    # Create edge list
    rows = np.array(all_rows)
    cols = np.array(all_cols)
    
    # Process distances if requested
    if return_distances:
        scores = np.array(all_scores)
        
        # Improved similarity transformation
        if metric == 'l2':
            # Use Gaussian kernel with adaptive bandwidth
            # Scale by median to ensure good distribution
            median_dist = np.median(scores[scores > 0])  # Exclude self-loops
            sigma = median_dist / np.sqrt(2.0)
            scores = np.exp(-(scores**2) / (2 * sigma**2))
            
            # Apply z-score normalization to preserve distribution
            scores_mean = scores.mean()
            scores_std = scores.std() + 1e-6
            scores = (scores - scores_mean) / scores_std
            
        elif metric == 'ip':
            # For inner product, preserve negative values
            # Don't normalize to [0,1], use z-score instead
            scores_mean = scores.mean()
            scores_std = scores.std() + 1e-6
            scores = (scores - scores_mean) / scores_std
    else:
        scores = None
    
    print(f"KNN graph computed in {time.time() - start_time:.2f}s")
    return rows, cols, scores


def tf_idf_similarity(diagnoses: np.ndarray, 
                      k: int = 10,
                      batch_size: int = 2000) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute TF-IDF weighted similarity matrix for diagnoses
    
    Args:
        diagnoses: Binary diagnosis matrix (N, D)
        k: Number of neighbors per node
        batch_size: Batch size for processing
        
    Returns:
        similarity_sparse: Sparse similarity matrix
        rows, cols: Edge indices
        scores: Similarity scores
    """
    N, D = diagnoses.shape
    
    # Compute IDF weights (log of inverse document frequency)
    doc_freq = np.sum(diagnoses > 0, axis=0)  # How many patients have each diagnosis
    idf = np.log(N / (doc_freq + 1))  # Add 1 to avoid division by zero
    
    # Create TF-IDF matrix (TF is binary in this case)
    tfidf = diagnoses * idf
    
    print(f"Computing TF-IDF similarity for {N} patients...")
    start_time = time.time()
    
    # For large datasets, use FAISS for efficiency
    if N > 20000 and k < 50:
        print("Using FAISS for efficient TF-IDF similarity search")
        # Normalize rows for cosine similarity
        norms = np.sqrt((tfidf * tfidf).sum(axis=1))
        norms[norms == 0] = 1.0
        tfidf_norm = tfidf / norms[:, np.newaxis]
        
        # Use FAISS inner product
        rows, cols, scores = compute_knn_graph(
            tfidf_norm, k=k, metric='ip', 
            batch_size=batch_size, index_type='flat',
            use_gpu=True
        )
        
        # Don't need to return sparse matrix for large datasets
        return None, rows, cols, scores
    
    # For smaller datasets, use sparse matrix multiplication
    # Convert to sparse for efficient multiplication
    tfidf_sparse = sparse.csr_matrix(tfidf)
    
    # Normalize the entire matrix first for proper cosine similarity
    norms = np.array(np.sqrt(tfidf_sparse.multiply(tfidf_sparse).sum(axis=1))).flatten()
    norms[norms == 0] = 1.0  # Avoid division by zero
    row_diag = sparse.diags(1.0 / norms)
    tfidf_norm = row_diag @ tfidf_sparse
    
    # Compute similarity in batches
    rows = []
    cols = []
    scores = []
    
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        batch = tfidf_norm[i:end_i]
        
        # Compute cosine similarity with all other patients
        sim = batch @ tfidf_norm.T
        
        # Find top k similar patients per row
        for j, row in enumerate(range(i, end_i)):
            # Get similarity scores and indices
            row_sim = sim[j].toarray().flatten()
            
            # Sort indices by similarity (descending)
            sorted_indices = np.argsort(-row_sim)
            
            # Take top k (excluding self) using parameter k
            top_indices = sorted_indices[sorted_indices != row][:k]
            top_scores = row_sim[top_indices]

            if len(top_indices) == 0:
                top_indices = np.array([row])
                top_scores  = np.array([0.0])
            
            # Apply z-score normalization to preserve negative correlations
            if len(top_scores) > 1:
                scores_mean = top_scores.mean()
                scores_std = top_scores.std() + 1e-6
                top_scores = (top_scores - scores_mean) / scores_std
            
            # Add to results
            rows.extend([row] * len(top_indices))
            cols.extend(top_indices)
            scores.extend(top_scores)
        
        if (i // batch_size) % 10 == 0:
            print(f"Processed {end_i}/{N} patients...")
    
    # Create sparse matrix from COO
    similarity_sparse = sparse.coo_matrix((scores, (rows, cols)), shape=(N, N))
    
    print(f"TF-IDF similarity computed in {time.time() - start_time:.2f}s")
    return similarity_sparse, np.array(rows), np.array(cols), np.array(scores)


def read_txt(path, node=True):
    """
    Read raw txt file into lists
    """
    with open(path, "r") as f:
        content = f.read()
    if node:
        return [int(n) for n in content.split('\n') if n != '']
    else:
        return [float(n) for n in content.split('\n') if n != '']


def load_edge_list(graph_dir: str, 
                  prefix: str,
                  load_binary: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load edge list from files
    
    Args:
        graph_dir: Directory containing graph files
        prefix: Prefix for filenames
        load_binary: Whether to load from .npy (faster) or .txt
        
    Returns:
        u, v: Source and target node indices
        scores: Edge scores
        types: Edge types (if exists)
    """
    # First try binary format for speed
    if load_binary and os.path.exists(f"{graph_dir}/{prefix}u.npy"):
        u = np.load(f"{graph_dir}/{prefix}u.npy")
        v = np.load(f"{graph_dir}/{prefix}v.npy")
        
        scores = None
        if os.path.exists(f"{graph_dir}/{prefix}scores.npy"):
            scores = np.load(f"{graph_dir}/{prefix}scores.npy")
            
        types = None
        if os.path.exists(f"{graph_dir}/{prefix}types.npy"):
            types = np.load(f"{graph_dir}/{prefix}types.npy")
    else:
        # Fall back to text format
        u_path = Path(graph_dir) / f"{prefix}u.txt"
        v_path = Path(graph_dir) / f"{prefix}v.txt"
        scores_path = Path(graph_dir) / f"{prefix}scores.txt"
        types_path = Path(graph_dir) / f"{prefix}types.txt"
        
        try:
            u = np.loadtxt(u_path, dtype=np.int32)
            v = np.loadtxt(v_path, dtype=np.int32)
            
            scores = None
            if os.path.exists(scores_path):
                scores = np.loadtxt(scores_path, dtype=np.float32)
                
            types = None
            if os.path.exists(types_path):
                types = np.loadtxt(types_path, dtype=np.int32)
        except:
            print(f"Error loading edge list from {graph_dir}/{prefix}*.txt")
            raise
            
    return u, v, scores, types