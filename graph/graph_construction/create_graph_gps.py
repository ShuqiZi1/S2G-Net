"""
Enhanced graph construction for GraphGPS-Mamba with performance optimizations
"""
import os
import numpy as np
import pandas as pd
import torch
import argparse
import json
import time
from pathlib import Path
from scipy import sparse
from typing import Tuple, List, Optional, Dict, Union

# Import our custom KNN utils
from graph.graph_construction.knn_utils import (
    compute_knn_graph, build_faiss_index, save_edge_list, 
    tf_idf_similarity, load_edge_list, read_txt
)


def convert_numpy_to_python(obj):
    """
    Convert numpy data types to Python native types for JSON serialization
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(v) for v in obj]
    else:
        return obj


def generate_graph_prefix(config: Dict) -> str:
    """
    Generate a descriptive prefix for graph files based on construction parameters
    
    Args:
        config: Configuration dictionary
        
    Returns:
        prefix: Descriptive prefix string
    """
    k_diag = config.get("k_diag", 5)
    k_bert = config.get("k_bert", 3)
    method = config.get("diag_method", "tfidf")
    rewiring = config.get("rewiring", "none")
    score_transform = config.get("score_transform", "zscore")
    
    # Build prefix components
    prefix_parts = [
        f"gps_k{k_diag}_{k_bert}"
    ]
    
    # Add method abbreviation
    method_abbr = {
        "tfidf": "tf",
        "faiss": "fa", 
        "penalize": "pn"
    }.get(method, method[:2])
    prefix_parts.append(method_abbr)
    
    # Add rewiring if not none
    if rewiring != "none":
        rewiring_abbr = {
            "mst": "mst",
            "gdc": "gdc",
            "gdc_light": "gdcl"
        }.get(rewiring, rewiring[:3])
        prefix_parts.append(rewiring_abbr)
    
    # Add score transform if not default
    if score_transform != "zscore":
        transform_abbr = {
            "log1p": "log",
            "none": "raw"
        }.get(score_transform, score_transform[:3])
        prefix_parts.append(transform_abbr)
    
    # Join with underscores and add trailing underscore
    prefix = "_".join(prefix_parts) + "_"
    
    return prefix


def create_diagnosis_graph(diagnoses: pd.DataFrame, 
                          k: int = 5,
                          method: str = 'tfidf',
                          batch_size: int = 2000,
                          score_transform: str = 'log1p') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create graph from diagnosis co-occurrence with improved edge weighting
    
    Args:
        diagnoses: Binary diagnosis matrix
        k: Number of neighbors per node
        method: Method to compute similarity ('tfidf' or 'penalize')
        batch_size: Batch size for processing
        score_transform: Transform for edge scores ('log1p', 'zscore', 'none')
    
    Returns:
        row_indices, col_indices: Edge indices
        scores: Similarity scores (transformed)
        edge_type: Edge types
    """
    N = len(diagnoses)
    diagnoses_array = np.array(diagnoses).astype(np.float32)
    
    if method == 'tfidf':
        # Use TF-IDF weighted similarity
        _, rows, cols, values = tf_idf_similarity(diagnoses_array, k=k, batch_size=batch_size)
        edge_type = np.zeros(len(rows), dtype=np.int32)  # Type 0 for diagnosis edges
        
    elif method == 'faiss':
        # Use FAISS for approximate inner product search
        rows, cols, values = compute_knn_graph(
            diagnoses_array, k=k, metric='ip', 
            batch_size=batch_size, self_loops=False,
            return_distances=True, use_gpu=True
        )
        edge_type = np.zeros(len(rows), dtype=np.int32)  # Type 0 for diagnosis edges
        
    else:  # Legacy method with penalization
        # Compute scores using original penalization method
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        diagnoses_tensor = torch.tensor(diagnoses_array, device=device)
        
        scores = torch.zeros((N, N), device=device)
        
        # Process in batches to avoid OOM
        for i in range(0, N, batch_size):
            end_i = min(i + batch_size, N)
            print(f"Processing batch {i} to {end_i}...")
            
            # Compute raw co-occurrence scores
            batch_scores = torch.matmul(diagnoses_tensor[i:end_i], diagnoses_tensor.T)
            
            # Penalize by combined diagnosis count
            diags_per_pt = diagnoses_tensor.sum(dim=1)
            total_diags = diags_per_pt.repeat(end_i-i, 1) + diags_per_pt[i:end_i].unsqueeze(1)
            
            # Apply penalization formula
            batch_scores = 5 * batch_scores - total_diags
            
            # Remove self connections
            for j in range(i, end_i):
                batch_scores[j-i, j] = -1000  # Large negative to avoid self-connection
                
            # Store in full score matrix
            scores[i:end_i] = batch_scores
        
        # Convert to CPU and numpy for processing
        scores = scores.cpu().numpy()
        
        # Extract top-k connections for each patient
        rows = []
        cols = []
        values = []
        
        for i in range(N):
            # Get top-k indices by score
            top_k_idx = np.argsort(scores[i])[-k:]
            
            # Add to edge lists
            rows.extend([i] * len(top_k_idx))
            cols.extend(top_k_idx)
            values.extend(scores[i, top_k_idx])
        
        rows = np.array(rows)
        cols = np.array(cols)
        values = np.array(values)
        edge_type = np.zeros(len(rows), dtype=np.int32)  # Type 0 for diagnosis edges
    
    # Apply score transformation for better distribution
    if score_transform == 'log1p':
        # Log transform for positive values
        values = values - np.min(values) + 1e-6  # Make positive
        values = np.log1p(values)
    elif score_transform == 'zscore':
        # Z-score normalization to preserve negative correlations
        values = (values - np.mean(values)) / (np.std(values) + 1e-6)
    else:  # 'none' or normalize to [0,1]
        # Normalize scores to [0,1]
        min_val = np.min(values)
        max_val = np.max(values)
        if max_val > min_val:
            values = (values - min_val) / (max_val - min_val)
        else:
            values = np.ones_like(values) * 0.5
    
    # Prune weak edges (bottom 30%)
    threshold = np.percentile(np.abs(values), 30)
    mask = np.abs(values) >= threshold
    rows = rows[mask]
    cols = cols[mask]
    values = values[mask]
    edge_type = edge_type[mask]
    
    return rows, cols, values, edge_type


def create_bert_graph(bert_embeddings: np.ndarray,
                     k: int = 5,
                     batch_size: int = 10000,
                     use_gpu: bool = True,
                     score_transform: str = 'zscore') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create graph from BERT embeddings using approximate KNN with improved scoring
    
    Args:
        bert_embeddings: BERT embeddings
        k: Number of neighbors per node
        batch_size: Batch size for processing
        use_gpu: Whether to use GPU for FAISS
        score_transform: Transform for edge scores
        
    Returns:
        rows, cols: Edge indices
        scores: Similarity scores
        edge_type: Edge types
    """
    # Use Approximate KNN with FAISS
    rows, cols, values = compute_knn_graph(
        bert_embeddings, k=k, metric='l2',
        batch_size=batch_size, return_distances=True,
        index_type='hnsw', self_loops=False,
        use_gpu=use_gpu
    )
    
    # Edge type 1 for BERT edges
    edge_type = np.ones(len(rows), dtype=np.int32)
    
    # Apply score transformation
    if score_transform == 'zscore':
        values = (values - np.mean(values)) / (np.std(values) + 1e-6)
    
    return rows, cols, values, edge_type


def merge_graph_views(edge_lists: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                      max_edges_per_node: int = 15,
                      undirected: bool = True,
                      edge_weight_balance: List[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge multiple graph views with intelligent edge selection and type preservation
    
    Args:
        edge_lists: List of (rows, cols, values, types) for each view
        max_edges_per_node: Maximum edges to keep per node
        undirected: Whether to remove duplicates in undirected sense
        edge_weight_balance: Weight multipliers for each view
        
    Returns:
        u, v: Final edge indices
        scores: Edge scores
        types: Edge types (preserved)
    """
    all_edges = []
    
    # Apply view-specific weights if provided
    if edge_weight_balance is None:
        edge_weight_balance = [1.0] * len(edge_lists)
    
    # Collect all edges from different views
    for i, (rows, cols, values, types) in enumerate(edge_lists):
        # Apply view weight
        weighted_values = values * edge_weight_balance[i]
        
        # Create DataFrame for easier processing
        df = pd.DataFrame({
            'u': rows,
            'v': cols,
            'score': weighted_values,
            'type': types,
            'view': i  # Track which view the edge came from
        })
        all_edges.append(df)
    
    # Concatenate all edges
    combined = pd.concat(all_edges, ignore_index=True)
    
    # Create edge key for undirected graph deduplication
    if undirected:
        # Create a canonical edge ID (min node, max node)
        combined['key'] = combined.apply(
            lambda r: (min(r.u, r.v), max(r.u, r.v)), axis=1
        )
        # For duplicate edges, keep the one with highest score but preserve its type
        combined = combined.sort_values('score', ascending=False)
        combined = combined.drop_duplicates('key', keep='first')
    else:
        combined = combined.sort_values('score', ascending=False)
        combined = combined.drop_duplicates(['u', 'v'], keep='first')
    
    # Compute out-degree for each node
    out_degrees = combined['u'].value_counts()
    
    # Identify nodes with too many edges
    high_degree_nodes = out_degrees[out_degrees > max_edges_per_node].index
    
    # Filter edges for high-degree nodes while preserving type diversity
    filtered_edges = []
    
    for node in high_degree_nodes:
        # Get all edges from this node
        node_edges = combined[combined['u'] == node]
        
        # Keep top edges per type to maintain diversity
        types_in_edges = node_edges['type'].unique()
        edges_per_type = max(1, max_edges_per_node // len(types_in_edges))
        
        selected_edges = []
        for edge_type in types_in_edges:
            type_edges = node_edges[node_edges['type'] == edge_type]
            top_type_edges = type_edges.nlargest(edges_per_type, 'score')
            selected_edges.append(top_type_edges)
        
        # Combine and take top k
        all_selected = pd.concat(selected_edges)
        top_edges = all_selected.nlargest(max_edges_per_node, 'score')
        filtered_edges.append(top_edges)
    
    # Get edges for normal-degree nodes
    normal_edges = combined[~combined['u'].isin(high_degree_nodes)]
    
    # Combine filtered and normal edges
    if filtered_edges:
        filtered_df = pd.concat(filtered_edges)
        final_edges = pd.concat([normal_edges, filtered_df])
    else:
        final_edges = normal_edges
    
    # Drop the key column if it exists
    if 'key' in final_edges.columns:
        final_edges = final_edges.drop(columns=['key'])
    
    # Convert back to arrays
    u = final_edges['u'].values
    v = final_edges['v'].values
    scores = final_edges['score'].values
    types = final_edges['type'].values
    
    # Final score normalization to prevent gradient issues
    scores = (scores - np.mean(scores)) / (np.std(scores) + 1e-6)
    
    return u, v, scores, types


def sigmoid_fast(x):
    """Fast sigmoid that avoids overflow"""
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def inverse_sigmoid_fast(y):
    """Fast inverse sigmoid (logit)"""
    # Clip to avoid log(0) or log(1)
    y = np.clip(y, 1e-7, 1 - 1e-7)
    return np.log(y / (1 - y))


def apply_graph_rewiring(u: np.ndarray, v: np.ndarray, scores: np.ndarray, types: np.ndarray,
                         method: str = 'none',
                         N: int = None,
                         alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply graph rewiring techniques to improve connectivity
    
    Args:
        u, v: Edge indices
        scores: Edge scores
        types: Edge types
        method: Rewiring method ('none', 'mst', 'gdc_light')
        N: Number of nodes
        alpha: GDC alpha parameter
        
    Returns:
        u, v: Rewired edge indices
        scores: Updated edge scores
        types: Edge types
    """
    if method == 'none':
        return u, v, scores, types
    
    if N is None:
        N = max(np.max(u), np.max(v)) + 1
    
    if method == 'mst':
        # Add minimum spanning tree edges to ensure connectivity
        try:
            import networkx as nx
            
            # Create weighted graph
            G = nx.Graph()
            G.add_nodes_from(range(N))
            
            # Add weighted edges
            for i, (src, dst, score) in enumerate(zip(u, v, scores)):
                G.add_edge(src, dst, weight=1.0 - score)  # Lower score = higher weight for MST
            
            # Find disconnected components
            components = list(nx.connected_components(G))
            
            if len(components) > 1:
                print(f"Graph has {len(components)} connected components. Adding MST edges...")
                
                # Connect components with minimal edges
                min_deg_nodes = []
                for comp in components:
                    comp_nodes = list(comp)
                    if len(comp_nodes) == 1:
                        min_deg_nodes.append(comp_nodes[0])
                    else:
                        degrees = {n: G.degree(n) for n in comp_nodes}
                        min_deg_nodes.append(min(degrees, key=degrees.get))
                
                # Connect minimum degree nodes in a ring
                new_edges_u = []
                new_edges_v = []
                
                for i in range(len(min_deg_nodes)):
                    j = (i + 1) % len(min_deg_nodes)
                    new_edges_u.extend([min_deg_nodes[i], min_deg_nodes[j]])
                    new_edges_v.extend([min_deg_nodes[j], min_deg_nodes[i]])
                
                # Add new edges with default score and type
                u = np.append(u, new_edges_u)
                v = np.append(v, new_edges_v)
                scores = np.append(scores, np.full(len(new_edges_u), 0.0))  # Neutral score
                types = np.append(types, np.full(len(new_edges_u), 2, dtype=np.int32))  # Type 2 for MST
                
                print(f"Added {len(new_edges_u)} MST edges")
            
        except ImportError:
            print("NetworkX not found, skipping MST rewiring")
    
    elif method == 'gdc_light':
        # Simple and fast GDC implementation
        print(f"Applying optimized GDC with alpha={alpha}")
        
        try:
            # For large graphs, use simple connectivity enhancement
            if N > 10000:
                # Simple GDC: connect high-degree nodes
                degrees = np.bincount(np.concatenate([u, v]), minlength=N)
                
                # Get top 100 highest degree nodes
                top_k = min(100, N // 100)
                top_nodes = np.argsort(-degrees)[:top_k]
                
                # Connect them in a cycle
                new_u = []
                new_v = []
                
                for i in range(len(top_nodes)):
                    j = (i + 1) % len(top_nodes)
                    new_u.append(top_nodes[i])
                    new_v.append(top_nodes[j])
                    # Also add reverse edge
                    new_u.append(top_nodes[j])
                    new_v.append(top_nodes[i])
                
                # Add to existing edges
                u = np.append(u, new_u)
                v = np.append(v, new_v)
                scores = np.append(scores, np.zeros(len(new_u)))
                types = np.append(types, np.full(len(new_u), 3, dtype=np.int32))
                
                print(f"Added {len(new_u)} GDC edges (simple method)")
                
            else:
                # For smaller graphs, can do more sophisticated GDC
                from scipy import sparse
                
                # Convert to adjacency matrix
                edge_weights = sigmoid_fast(scores)
                A = sparse.coo_matrix((edge_weights, (u, v)), shape=(N, N))
                A = A.tocsr()
                
                # Make symmetric and add self-loops
                A = A + A.T
                A = A.multiply(0.5)
                I = sparse.eye(N, format='csr') * 0.1
                A = A + I
                
                # Simple diffusion: A' = alpha*I + (1-alpha)*A
                A_diffused = alpha * sparse.eye(N, format='csr') + (1 - alpha) * A
                
                # Extract edges above threshold
                A_coo = A_diffused.tocoo()
                mask = (A_coo.row != A_coo.col) & (A_coo.data > 0.01)
                
                new_u = A_coo.row[mask]
                new_v = A_coo.col[mask]
                new_weights = A_coo.data[mask]
                
                # Keep only strongest edges
                if len(new_u) > len(u) * 2:
                    sorted_idx = np.argsort(-new_weights)[:len(u) * 2]
                    new_u = new_u[sorted_idx]
                    new_v = new_v[sorted_idx]
                    new_weights = new_weights[sorted_idx]
                
                # Convert back to z-scores
                new_scores = inverse_sigmoid_fast(new_weights)
                new_scores = (new_scores - np.mean(new_scores)) / (np.std(new_scores) + 1e-6)
                
                # Replace edges
                u = new_u
                v = new_v
                scores = new_scores
                types = np.full(len(u), 3, dtype=np.int32)
                
                print(f"GDC complete: {len(u)} edges")
            
        except Exception as e:
            print(f"Error in GDC: {e}")
            print("Skipping GDC rewiring")
    
    return u, v, scores, types


def main_graph_construction(
    config: Dict,
    diagnoses_df: pd.DataFrame = None,
    bert_embeddings: np.ndarray = None,
    time_series: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Main function for multi-view graph construction with enhanced naming
    
    Args:
        config: Configuration dictionary
        diagnoses_df: Diagnosis data
        bert_embeddings: BERT embeddings
        time_series: Time series data (optional)
        
    Returns:
        u, v: Final edge indices
        scores: Edge scores (z-normalized)
        types: Edge types (preserved)
    """
    # Extract configuration with optimized defaults
    graph_dir = config.get("graph_dir")
    k_diag = config.get("k_diag", 5)
    k_bert = config.get("k_bert", 3)
    k_ts = config.get("k_ts", 0)  # 0 means don't use TS edges
    max_edges_per_node = config.get("max_edges_per_node", 15)
    batch_size = config.get("batch_size", 1000)
    use_gpu = config.get("use_gpu", True)
    rewiring = config.get("rewiring", "none")
    diag_method = config.get("diag_method", "tfidf")
    score_transform = config.get("score_transform", "zscore")
    edge_weight_balance = config.get("edge_weight_balance", [1.0, 0.8])
    gdc_alpha = config.get("gdc_alpha", 0.05)
    
    # Generate descriptive prefix
    prefix = generate_graph_prefix(config)
    config['graph_prefix'] = prefix  # Store for later use
    
    print(f"\nUsing graph prefix: {prefix}")
    
    # Create edge lists for each view
    edge_lists = []
    
    # 1. Diagnosis graph
    if diagnoses_df is not None:
        print("\n=== Creating diagnosis graph ===")
        u_diag, v_diag, scores_diag, types_diag = create_diagnosis_graph(
            diagnoses_df, k=k_diag, method=diag_method, 
            batch_size=batch_size, score_transform=score_transform
        )
        
        print(f"Diagnosis graph: {len(u_diag)} edges")
        print(f"Score stats: mean={np.mean(scores_diag):.3f}, std={np.std(scores_diag):.3f}")
        edge_lists.append((u_diag, v_diag, scores_diag, types_diag))
        
        # Save individual view
        if graph_dir:
            save_edge_list(
                u_diag, v_diag, scores_diag, types_diag,
                prefix="diag_", graph_dir=graph_dir
            )
    
    # 2. BERT embedding graph
    if bert_embeddings is not None:
        print("\n=== Creating BERT embedding graph ===")
        u_bert, v_bert, scores_bert, types_bert = create_bert_graph(
            bert_embeddings, k=k_bert, batch_size=batch_size, 
            use_gpu=use_gpu, score_transform=score_transform
        )
        
        print(f"BERT graph: {len(u_bert)} edges")
        print(f"Score stats: mean={np.mean(scores_bert):.3f}, std={np.std(scores_bert):.3f}")
        edge_lists.append((u_bert, v_bert, scores_bert, types_bert))
        
        # Save individual view
        if graph_dir:
            save_edge_list(
                u_bert, v_bert, scores_bert, types_bert,
                prefix="bert_", graph_dir=graph_dir
            )
    
    # 3. Time series graph (optional)
    if time_series is not None and k_ts > 0:
        # Placeholder for TS graph construction
        # This would be implemented with DTW or other time series similarity
        pass
    
    # Merge views and apply rewiring
    if len(edge_lists) > 0:
        print("\n=== Merging graph views ===")
        u, v, scores, types = merge_graph_views(
            edge_lists, max_edges_per_node=max_edges_per_node,
            undirected=True, edge_weight_balance=edge_weight_balance
        )
        
        print(f"Merged graph: {len(u)} edges")
        print(f"Score stats: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")
        print(f"Edge type distribution: {np.bincount(types)}")
        
        # Determine number of nodes
        N = max(np.max(u), np.max(v)) + 1
        
        # Apply rewiring if needed
        if rewiring != "none":
            print(f"\n=== Applying {rewiring} rewiring ===")
            u, v, scores, types = apply_graph_rewiring(
                u, v, scores, types, method=rewiring, N=N, alpha=gdc_alpha
            )
            print(f"After rewiring: {len(u)} edges")
            print(f"Score stats: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")
        
        # Save final graph with descriptive prefix
        if graph_dir:
            save_edge_list(
                u, v, scores, types,
                prefix=prefix, graph_dir=graph_dir
            )
            
            # Also save configuration used for this graph
            # Convert numpy types to Python native types for JSON serialization
            edge_type_dist = np.bincount(types)
            edge_type_dist_dict = {int(i): int(count) for i, count in enumerate(edge_type_dist) if count > 0}
            
            config_save = {
                'k_diag': int(k_diag),
                'k_bert': int(k_bert),
                'diag_method': str(diag_method),
                'rewiring': str(rewiring),
                'score_transform': str(score_transform),
                'max_edges_per_node': int(max_edges_per_node),
                'edge_weight_balance': [float(x) for x in edge_weight_balance],
                'gdc_alpha': float(gdc_alpha) if rewiring == 'gdc_light' else None,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'n_nodes': int(N),
                'n_edges': int(len(u)),
                'edge_type_distribution': edge_type_dist_dict
            }
            
            config_path = os.path.join(graph_dir, f"{prefix}config.json")
            with open(config_path, 'w') as f:
                json.dump(config_save, f, indent=2)
            print(f"Saved graph configuration to {config_path}")
        
        # Create symmetric versions for traditional GNN libraries
        u_sym = np.concatenate([u, v])
        v_sym = np.concatenate([v, u])
        scores_sym = np.concatenate([scores, scores])
        types_sym = np.concatenate([types, types])
        
        if graph_dir:
            save_edge_list(
                u_sym, v_sym, scores_sym, types_sym,
                prefix=f"{prefix}sym_", graph_dir=graph_dir
            )
        
        return u, v, scores, types
    
    return None, None, None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_diag', type=int, default=5)
    parser.add_argument('--k_bert', type=int, default=3)
    parser.add_argument('--diag_method', type=str, default='tfidf', choices=['tfidf', 'faiss', 'penalize'])
    parser.add_argument('--max_edges', type=int, default=15)
    parser.add_argument('--rewiring', type=str, default='none', choices=['none', 'mst', 'gdc_light'])
    parser.add_argument('--score_transform', type=str, default='zscore', choices=['zscore', 'log1p', 'none'])
    parser.add_argument('--gdc_alpha', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--debug', action='store_true', help='Run with small subset of data')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for FAISS')
    args = parser.parse_args()

    print(args)

    # Load configuration
    try:
        with open('paths.json', 'r') as f:
            data = json.load(f)
            MIMIC_path = data["MIMIC_path"]
            graph_dir = data["graph_dir"]
            print(f"MIMIC path: {MIMIC_path}")
            print(f"Graph directory: {graph_dir}")
    except FileNotFoundError:
        print("paths.json not found, using default paths")
        MIMIC_path = "./data/"
        graph_dir = "./graphs/"
    
    # Ensure graph directory exists
    os.makedirs(graph_dir, exist_ok=True)
    
    # Read diagnosis data
    try:
        print("Reading diagnoses data...")
        train_diagnoses = pd.read_csv(f'{MIMIC_path}train/diagnoses.csv', index_col='patient')
        val_diagnoses = pd.read_csv(f'{MIMIC_path}val/diagnoses.csv', index_col='patient')
        test_diagnoses = pd.read_csv(f'{MIMIC_path}test/diagnoses.csv', index_col='patient')
        all_diagnoses = pd.concat([train_diagnoses, val_diagnoses, test_diagnoses], sort=False)
        print(f"Shape of all_diagnoses: {all_diagnoses.shape}")
        
        if args.debug:
            all_diagnoses = all_diagnoses.iloc[:1000]
            print(f"Debug mode: using {len(all_diagnoses)} samples")
    except FileNotFoundError:
        print("Diagnosis files not found")
        all_diagnoses = None
    
    # Read BERT embeddings
    try:
        print("Reading BERT embeddings...")
        bert_path = f"{graph_dir}bert_out.npy"
        bert_embeddings = np.load(bert_path)
        print(f"BERT embeddings shape: {bert_embeddings.shape}")
        
        if args.debug:
            bert_embeddings = bert_embeddings[:1000]
    except FileNotFoundError:
        print(f"BERT embeddings not found at {bert_path}")
        bert_embeddings = None
    
    # Configure and run graph construction
    config = {
        "graph_dir": graph_dir,
        "k_diag": args.k_diag,
        "k_bert": args.k_bert,
        "diag_method": args.diag_method,
        "max_edges_per_node": args.max_edges,
        "batch_size": args.batch_size,
        "use_gpu": args.use_gpu,
        "rewiring": args.rewiring,
        "score_transform": args.score_transform,
        "gdc_alpha": args.gdc_alpha,
        "edge_weight_balance": [1.0, 0.8],
    }
    
    # Run graph construction
    u, v, scores, types = main_graph_construction(
        config, 
        diagnoses_df=all_diagnoses,
        bert_embeddings=bert_embeddings
    )
    
    if u is not None and v is not None:
        # Print summary statistics
        print("\n=== Graph Construction Summary ===")
        print(f"Graph prefix: {config.get('graph_prefix', 'N/A')}")
        print(f"Total nodes: {max(np.max(u), np.max(v)) + 1}")
        print(f"Total edges: {len(u)}")
        print(f"Average degree: {2 * len(u) / (max(np.max(u), np.max(v)) + 1):.2f}")
        
        # Edge type breakdown
        edge_type_names = {0: "Diagnosis", 1: "BERT", 2: "MST", 3: "GDC"}
        print("\nEdge type breakdown:")
        edge_counts = np.bincount(types)
        for edge_type, count in enumerate(edge_counts):
            if count > 0:
                type_name = edge_type_names.get(edge_type, f"Type {edge_type}")
                print(f"  {type_name}: {count} edges ({100 * count / len(types):.1f}%)")
    
    print("\nGraph construction complete!")