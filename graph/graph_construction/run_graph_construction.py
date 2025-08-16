"""
Run script for GraphGPS graph construction with proper data alignment and academic visualization
"""
import os
import numpy as np
import pandas as pd
import json
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from graph.graph_construction.create_graph_gps import main_graph_construction
from pathlib import Path
import time

# Import visualization functions
try:
    from src.visualization.graph_viz.graph_viz import (
        visualize_graph_as_parameter,
        compare_multiple_graph_versions,
        create_graph_comparison_figure,
        create_graph_statistics_table,
        create_connectivity_analysis
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("Warning: academic_graph_viz module not found. Visualization features will be disabled.")
    VISUALIZATION_AVAILABLE = False

# Set environment variables for performance
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def convert_numpy_types(obj):
    """
    Convert numpy types to Python native types for JSON serialization
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    else:
        return obj


def align_data_sources(all_diagnoses, bert_embeddings, MIMIC_path):
    """
    Align diagnosis and BERT data to ensure they have the same patients
    
    Args:
        all_diagnoses: DataFrame with diagnosis data
        bert_embeddings: numpy array with BERT embeddings
        MIMIC_path: Path to MIMIC data directory
        
    Returns:
        aligned_diagnoses: DataFrame aligned with BERT
        aligned_bert: numpy array aligned with diagnoses
    """
    n_diag = len(all_diagnoses)
    n_bert = len(bert_embeddings)
    
    print(f"\nData alignment check:")
    print(f"  Initial diagnosis patients: {n_diag}")
    print(f"  BERT embeddings: {n_bert}")
    
    if n_diag == n_bert:
        print("  ✓ Data already aligned")
        return all_diagnoses, bert_embeddings
    
    # Load patient order from train/val/test splits to match BERT order
    try:
        # BERT embeddings are created in order: train -> val -> test
        train_diag = pd.read_csv(f'{MIMIC_path}train/diagnoses.csv', index_col='patient')
        val_diag = pd.read_csv(f'{MIMIC_path}val/diagnoses.csv', index_col='patient')
        test_diag = pd.read_csv(f'{MIMIC_path}test/diagnoses.csv', index_col='patient')
        
        # Get the ordered patient list as BERT would have seen it
        bert_patient_order = list(train_diag.index) + list(val_diag.index) + list(test_diag.index)
        
        # Get patients currently in diagnosis data
        diag_patients = set(all_diagnoses.index)
        
        # Find patients in BERT but not in current diagnoses (these would have 0 diagnoses)
        missing_in_diag = []
        for patient_id in bert_patient_order:
            if patient_id not in diag_patients:
                missing_in_diag.append(patient_id)
        
        if missing_in_diag:
            print(f"  Found {len(missing_in_diag)} patients in BERT but not in diagnoses")
            print("  These are patients with no diagnosis codes")
            
            # Create zero rows for missing patients
            zero_data = pd.DataFrame(
                0, 
                index=missing_in_diag,
                columns=all_diagnoses.columns,
                dtype=all_diagnoses.dtypes.to_dict()
            )
            
            # Concatenate and reorder to match BERT
            all_diagnoses_extended = pd.concat([all_diagnoses, zero_data])
            all_diagnoses_aligned = all_diagnoses_extended.loc[bert_patient_order]
            
            print(f"  ✓ Added zero rows for missing patients")
            print(f"  Final diagnosis shape: {all_diagnoses_aligned.shape}")
            
            return all_diagnoses_aligned, bert_embeddings
            
    except Exception as e:
        print(f"  Error during alignment: {e}")
        raise
    
    return all_diagnoses, bert_embeddings


def analyze_graph(u, v, scores, types, N=None, save_path=None):
    """
    Analyze the graph structure and properties
    
    Args:
        u, v: Edge indices
        scores: Edge weights
        types: Edge types
        N: Number of nodes (optional)
        save_path: Path to save analysis results (optional)
    """
    if N is None:
        N = max(np.max(u), np.max(v)) + 1
    
    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(N))
    
    for i, (src, dst, score, edge_type) in enumerate(zip(u, v, scores, types)):
        G.add_edge(src, dst, weight=score, type=edge_type)
    
    # Basic statistics
    n_edges = G.number_of_edges()
    n_nodes = G.number_of_nodes()
    avg_degree = 2 * n_edges / n_nodes
    density = nx.density(G)
    
    analysis_text = []
    analysis_text.append("\n=== Graph Analysis ===")
    analysis_text.append(f"Nodes: {n_nodes}")
    analysis_text.append(f"Edges: {n_edges}")
    analysis_text.append(f"Average degree: {avg_degree:.2f}")
    analysis_text.append(f"Graph density: {density:.6f}")
    
    # Connectivity
    connected_components = list(nx.connected_components(G))
    n_components = len(connected_components)
    component_sizes = [len(comp) for comp in connected_components]
    largest_component = max(connected_components, key=len)
    largest_component_size = len(largest_component)
    largest_component_pct = 100 * largest_component_size / n_nodes
    
    analysis_text.append(f"\nConnected components: {n_components}")
    analysis_text.append(f"Largest component: {largest_component_size} nodes ({largest_component_pct:.1f}%)")
    
    if n_components > 1:
        analysis_text.append("Component size distribution:")
        analysis_text.append(f"  Min: {min(component_sizes)}, Max: {max(component_sizes)}")
        analysis_text.append(f"  Mean: {np.mean(component_sizes):.1f}, Median: {np.median(component_sizes):.1f}")
    
    # Edge type distribution
    unique_types = np.unique(types)
    analysis_text.append("\nEdge type distribution:")
    type_names = {0: "Diagnosis", 1: "BERT", 2: "MST", 3: "GDC"}
    for t in unique_types:
        type_count = np.sum(types == t)
        type_name = type_names.get(t, f"Type{t}")
        analysis_text.append(f"  {type_name}: {type_count} edges ({100 * type_count / len(types):.1f}%)")
    
    # Score distribution by type
    analysis_text.append("\nEdge score statistics by type:")
    for t in unique_types:
        type_mask = types == t
        type_scores = scores[type_mask]
        type_name = type_names.get(t, f"Type{t}")
        analysis_text.append(f"  {type_name}: mean={np.mean(type_scores):.4f}, std={np.std(type_scores):.4f}, "
                           f"range=[{np.min(type_scores):.4f}, {np.max(type_scores):.4f}]")
    
    # Score distribution
    analysis_text.append("\nOverall edge score distribution:")
    percentiles = [0, 10, 25, 50, 75, 90, 100]
    for p in percentiles:
        analysis_text.append(f"  {p:3d}th percentile: {np.percentile(scores, p):.4f}")
    
    # Degree distribution
    degrees = np.array([G.degree(n) for n in range(N)])
    analysis_text.append("\nDegree distribution:")
    for p in percentiles:
        analysis_text.append(f"  {p:3d}th percentile: {np.percentile(degrees, p):.1f}")
    
    # High degree nodes
    high_deg_threshold = np.percentile(degrees, 95)
    high_deg_count = np.sum(degrees > high_deg_threshold)
    analysis_text.append(f"\nNodes with degree > {high_deg_threshold}: {high_deg_count} ({100 * high_deg_count / N:.1f}%)")
    
    # Clustering coefficient (sample for large graphs)
    if N < 10000:
        avg_clustering = nx.average_clustering(G)
        analysis_text.append(f"Average clustering coefficient: {avg_clustering:.4f}")
    else:
        # Sample nodes for clustering calculation
        sample_size = min(1000, N)
        sample_nodes = np.random.choice(N, sample_size, replace=False)
        subgraph = G.subgraph(sample_nodes)
        avg_clustering = nx.average_clustering(subgraph)
        analysis_text.append(f"Average clustering coefficient (sampled): {avg_clustering:.4f}")
    
    # Print and optionally save analysis
    for line in analysis_text:
        print(line)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write('\n'.join(analysis_text))
        print(f"\nAnalysis saved to {save_path}")
    
    # Return graph for further analysis
    return G


def create_simple_visualization(G, u, v, scores, types, graph_dir, prefix, sample_size=100):
    """
    Create a simple graph visualization when academic_graph_viz is not available
    """
    N = G.number_of_nodes()
    
    # Sample nodes for visualization
    if N > sample_size:
        sample_nodes = np.random.choice(N, sample_size, replace=False)
        subgraph = G.subgraph(sample_nodes)
    else:
        subgraph = G
    
    pos = nx.spring_layout(subgraph, k=1/np.sqrt(len(subgraph)), iterations=50)
    plt.figure(figsize=(10, 8))
    
    # Color edges by type
    edge_colors = []
    type_colors = {0: 'blue', 1: 'red', 2: 'green', 3: 'orange'}
    for _, _, data in subgraph.edges(data=True):
        edge_colors.append(type_colors.get(data.get('type', 0), 'gray'))
    
    nx.draw_networkx(
        subgraph, pos=pos, 
        with_labels=False, 
        node_size=50, 
        edge_color=edge_colors,
        alpha=0.7
    )
    
    # Add legend
    from matplotlib.patches import Patch
    type_names = {0: "Diagnosis", 1: "BERT", 2: "MST", 3: "GDC"}
    legend_elements = []
    for t in np.unique(types):
        legend_elements.append(Patch(facecolor=type_colors[t], label=type_names.get(t, f"Type{t}")))
    plt.legend(handles=legend_elements, loc='upper left')
    
    plt.title(f"Graph Sample ({sample_size} nodes from {N} total)")
    plt.axis('off')
    
    output_path = f"{graph_dir}/visualization_{prefix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved simple visualization to {output_path}")


def process_pipeline(args):
    """Run the complete graph construction pipeline with proper alignment and visualization"""
    start_time = time.time()
    
    # Load configuration
    try:
        with open('paths.json', 'r') as f:
            paths = json.load(f)
            MIMIC_path = paths["MIMIC_path"]
            graph_dir = paths["graph_dir"]
            # Get graph_results directory, default to graph_dir/graph_results if not specified
            graph_results_dir = paths.get("graph_results", os.path.join(graph_dir, "graph_results"))
    except FileNotFoundError:
        print("paths.json not found, using default paths")
        MIMIC_path = "./data/"
        graph_dir = "./graphs/"
        graph_results_dir = "./graphs/graph_results/"
    
    # Ensure directories exist
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(graph_results_dir, exist_ok=True)
    
    # Create subdirectories for organization
    graph_viz_dir = os.path.join(graph_results_dir, "graph_viz")
    graph_analysis_dir = os.path.join(graph_results_dir, "graph_analysis")
    os.makedirs(graph_viz_dir, exist_ok=True)
    os.makedirs(graph_analysis_dir, exist_ok=True)
    
    print("\n=== Graph Construction Configuration ===")
    for key, value in vars(args).items():
        print(f"{key:>20}: {value}")
        
    # 1. Read data
    print("\n=== Reading input data ===")
    
    # Read diagnoses
    try:
        train_diagnoses = pd.read_csv(f'{MIMIC_path}train/diagnoses.csv', index_col='patient')
        val_diagnoses = pd.read_csv(f'{MIMIC_path}val/diagnoses.csv', index_col='patient')
        test_diagnoses = pd.read_csv(f'{MIMIC_path}test/diagnoses.csv', index_col='patient')
        
        # Get dimensions for verification
        n_train = len(train_diagnoses)
        n_val = len(val_diagnoses)
        n_test = len(test_diagnoses)
        n_total_expected = n_train + n_val + n_test
        
        print(f"Train patients: {n_train}")
        print(f"Val patients: {n_val}")
        print(f"Test patients: {n_test}")
        print(f"Total expected: {n_total_expected}")
        
        # Concatenate all diagnoses - DO NOT remove zero diagnosis patients
        all_diagnoses = pd.concat([train_diagnoses, val_diagnoses, test_diagnoses], sort=False)
        
        # Check how many patients have zero diagnoses
        zero_diag_mask = (all_diagnoses.sum(axis=1) == 0)
        n_zero_diag = zero_diag_mask.sum()
        
        if n_zero_diag > 0:
            print(f"\nFound {n_zero_diag} patients with zero diagnoses")
            print("Keeping them to maintain alignment with BERT embeddings")
        
        print(f"Shape of all_diagnoses: {all_diagnoses.shape}")
        
        if args.debug:
            # For debug mode, keep proportions
            sample_n = 1000
            prop_train = n_train / n_total_expected
            prop_val = n_val / n_total_expected
            
            n_train_debug = int(sample_n * prop_train)
            n_val_debug = int(sample_n * prop_val)
            n_test_debug = sample_n - n_train_debug - n_val_debug
            
            # Get patient indices for each split
            train_patients = list(train_diagnoses.index)[:n_train_debug]
            val_patients = list(val_diagnoses.index)[:n_val_debug]
            test_patients = list(test_diagnoses.index)[:n_test_debug]
            
            # Combine and get subset
            debug_patients = train_patients + val_patients + test_patients
            all_diagnoses = all_diagnoses.loc[debug_patients]
            
            print(f"Debug mode: using {len(all_diagnoses)} samples")
            
    except FileNotFoundError as e:
        print(f"Diagnosis files not found: {e}")
        all_diagnoses = None
        return None, None, None, None
    
    # Read BERT embeddings
    try:
        print("\nReading BERT embeddings...")
        bert_path = f"{graph_dir}bert_out.npy"
        bert_embeddings = np.load(bert_path)
        print(f"BERT embeddings shape: {bert_embeddings.shape}")
        
        if args.debug:
            # In debug mode, take same subset as diagnoses
            bert_embeddings = bert_embeddings[:len(all_diagnoses)]
            print(f"Debug mode: using {len(bert_embeddings)} BERT embeddings")
            
    except FileNotFoundError:
        print(f"BERT embeddings not found at {bert_path}")
        bert_embeddings = None
        return None, None, None, None
    
    # 2. Verify and align data
    if all_diagnoses is not None and bert_embeddings is not None:
        # Check alignment
        if len(all_diagnoses) != len(bert_embeddings):
            print(f"\nData size mismatch detected!")
            print(f"   Diagnoses: {len(all_diagnoses)}")
            print(f"   BERT: {len(bert_embeddings)}")
            
            # Try to align
            all_diagnoses, bert_embeddings = align_data_sources(
                all_diagnoses, bert_embeddings, MIMIC_path
            )
        else:
            print(f"\nData already aligned: {len(all_diagnoses)} patients")
    
    # 3. Configure graph construction
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
        "graph_prefix": f"gps_k{args.k_diag}_{args.k_bert}_"
    }
    
    # 4. Run graph construction
    print("\n=== Running graph construction ===")
    u, v, scores, types = main_graph_construction(
        config, 
        diagnoses_df=all_diagnoses,
        bert_embeddings=bert_embeddings
    )
    
    # 5. Analyze results
    if u is not None and v is not None:
        N = max(np.max(u), np.max(v)) + 1
        
        # Verify node count matches our data
        if N != len(all_diagnoses):
            print(f"\nWarning: Graph has {N} nodes but data has {len(all_diagnoses)} patients")
        
        # Analyze graph and save to graph_analysis directory
        analysis_filename = f"analysis_{args.k_diag}_{args.k_bert}_{args.diag_method}_{args.rewiring}.txt"
        analysis_path = os.path.join(graph_analysis_dir, analysis_filename)
        G = analyze_graph(u, v, scores, types, N, save_path=analysis_path)
        
        # 6. Print data split info
        print("\n=== Data Split ===")
        print(f"Train nodes: {n_train}")
        print(f"Validation nodes: {n_val}")
        print(f"Test nodes: {n_test}")
        
        # 7. Save graph statistics to graph_analysis directory
        stats = {
            "n_nodes": int(N),
            "n_edges": int(len(u)),
            "avg_degree": float(len(u) * 2 / N),
            "n_train": int(n_train),
            "n_val": int(n_val),
            "n_test": int(n_test),
            "k_diag": int(args.k_diag),
            "k_bert": int(args.k_bert),
            "diag_method": args.diag_method,
            "rewiring": args.rewiring,
            "score_transform": args.score_transform,
            "edge_types": {
                "diagnosis": int(np.sum(types == 0)),
                "bert": int(np.sum(types == 1)),
                "mst": int(np.sum(types == 2)) if 2 in types else 0,
                "gdc": int(np.sum(types == 3)) if 3 in types else 0
            },
            "edge_score_stats": {
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores))
            },
            "construction_time": float(time.time() - start_time)
        }
        
        stats_filename = f"graph_stats_{args.k_diag}_{args.k_bert}_{args.diag_method}_{args.rewiring}.json"
        stats_path = os.path.join(graph_analysis_dir, stats_filename)
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved graph statistics to {stats_path}")
        
        # 8. Visualization
        if args.visualize:
            print("\n=== Creating visualizations ===")
            
            if VISUALIZATION_AVAILABLE:
                # Use academic visualization with graph_viz directory
                visualize_graph_as_parameter(
                    config, u, v, scores, types,
                    output_dir=graph_viz_dir
                )
                
                # If comparing multiple versions
                if args.compare_versions:
                    version_prefixes = args.compare_versions.split(',')
                    compare_output_dir = os.path.join(graph_viz_dir, 'comparisons')
                    os.makedirs(compare_output_dir, exist_ok=True)
                    
                    compare_multiple_graph_versions(
                        graph_dir,
                        version_prefixes,
                        compare_output_dir
                    )
            else:
                # Fallback to simple visualization in graph_viz directory
                print("Using simple visualization (academic_graph_viz not available)")
                create_simple_visualization(
                    G, u, v, scores, types, 
                    graph_viz_dir, 
                    f"{args.k_diag}_{args.k_bert}_{args.diag_method}_{args.rewiring}",
                    sample_size=min(500, N)
                )
        
        # 9. Save configuration for reproducibility in graph_analysis directory
        config_filename = f"config_{args.k_diag}_{args.k_bert}_{args.diag_method}_{args.rewiring}.json"
        config_path = os.path.join(graph_analysis_dir, config_filename)
        with open(config_path, 'w') as f:
            json.dump(convert_numpy_types(config), f, indent=2)
        print(f"Saved configuration to {config_path}")
    
    total_time = time.time() - start_time
    print(f"\n=== Graph construction completed in {total_time:.2f} seconds ===")
    
    return u, v, scores, types


def batch_process(args):
    """
    Process multiple graph configurations in batch
    """
    # Load paths
    try:
        with open('paths.json', 'r') as f:
            paths = json.load(f)
            graph_dir = paths["graph_dir"]
            graph_results_dir = paths.get("graph_results", os.path.join(graph_dir, "graph_results"))
    except:
        graph_dir = "./graphs/"
        graph_results_dir = "./graphs/graph_results/"
    
    # Create graph_analysis directory for batch results
    graph_analysis_dir = os.path.join(graph_results_dir, "graph_analysis")
    os.makedirs(graph_analysis_dir, exist_ok=True)
    # Parse batch parameters
    k_diag_values = [int(k) for k in args.batch_k_diag.split(',')]
    k_bert_values = [int(k) for k in args.batch_k_bert.split(',')]
    diag_methods = args.batch_diag_methods.split(',')
    rewiring_methods = args.batch_rewiring.split(',')
    
    print(f"\n=== Batch Processing Configuration ===")
    print(f"k_diag values: {k_diag_values}")
    print(f"k_bert values: {k_bert_values}")
    print(f"Diagnosis methods: {diag_methods}")
    print(f"Rewiring methods: {rewiring_methods}")
    
    results = []
    
    for k_diag in k_diag_values:
        for k_bert in k_bert_values:
            for diag_method in diag_methods:
                for rewiring in rewiring_methods:
                    print(f"\n{'='*60}")
                    print(f"Processing: k_diag={k_diag}, k_bert={k_bert}, "
                          f"method={diag_method}, rewiring={rewiring}")
                    print(f"{'='*60}")
                    
                    # Update args
                    args.k_diag = k_diag
                    args.k_bert = k_bert
                    args.diag_method = diag_method
                    args.rewiring = rewiring
                    
                    # Process
                    try:
                        u, v, scores, types = process_pipeline(args)
                        if u is not None:
                            results.append({
                                'k_diag': k_diag,
                                'k_bert': k_bert,
                                'diag_method': diag_method,
                                'rewiring': rewiring,
                                'success': True,
                                'n_edges': len(u)
                            })
                    except Exception as e:
                        print(f"Error: {e}")
                        results.append({
                            'k_diag': k_diag,
                            'k_bert': k_bert,
                            'diag_method': diag_method,
                            'rewiring': rewiring,
                            'success': False,
                            'error': str(e)
                        })
    
    # Save batch results in graph_analysis directory
    batch_results_path = os.path.join(graph_analysis_dir, 'batch_results.json')
    with open(batch_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Batch processing complete ===")
    print(f"Results saved to {batch_results_path}")
    
    # Print summary
    successful = sum(1 for r in results if r['success'])
    print(f"Successfully processed: {successful}/{len(results)} configurations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphGPS Graph Construction")
    
    # Basic parameters
    parser.add_argument('--k_diag', type=int, default=3, help='K for diagnosis graph')
    parser.add_argument('--k_bert', type=int, default=1, help='K for BERT graph')
    parser.add_argument('--diag_method', type=str, default='faiss', 
                        choices=['tfidf', 'faiss', 'penalize'], 
                        help='Method for diagnosis similarity')
    parser.add_argument('--max_edges', type=int, default=10, help='Maximum edges per node')
    parser.add_argument('--rewiring', type=str, default='gdc_light', 
                        choices=['none', 'mst', 'gdc_light'], 
                        help='Graph rewiring method')
    parser.add_argument('--score_transform', type=str, default='zscore', 
                        choices=['zscore', 'log1p', 'none'],
                        help='Score transformation method')
    parser.add_argument('--gdc_alpha', type=float, default=0.05, 
                        help='GDC alpha parameter')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for processing')
    parser.add_argument('--debug', action='store_true', help='Run with small subset of data')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for FAISS')
    
    # Visualization parameters
    parser.add_argument('--visualize', action='store_true', 
                        help='Create academic visualizations')
    parser.add_argument('--compare_versions', type=str, default=None,
                        help='Comma-separated list of version prefixes to compare')
    
    # Batch processing parameters
    parser.add_argument('--batch', action='store_true', 
                        help='Enable batch processing mode')
    parser.add_argument('--batch_k_diag', type=str, default='3,5,10',
                        help='Comma-separated k_diag values for batch processing')
    parser.add_argument('--batch_k_bert', type=str, default='1,3,5',
                        help='Comma-separated k_bert values for batch processing')
    parser.add_argument('--batch_diag_methods', type=str, default='tfidf,faiss',
                        help='Comma-separated diagnosis methods for batch processing')
    parser.add_argument('--batch_rewiring', type=str, default='none,gdc_light',
                        help='Comma-separated rewiring methods for batch processing')
    
    args = parser.parse_args()
    
    if args.batch:
        batch_process(args)
    else:
        process_pipeline(args)