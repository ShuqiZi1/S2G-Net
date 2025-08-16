"""
Comprehensive graph construction study for Mamba-GPS ICU LOS prediction
Systematically evaluates k_diag × k_bert combinations across different methods and rewiring strategies
"""
import os
import sys
import json
import subprocess
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from graph.graph_construction.create_graph_gps import main_graph_construction
from src.evaluation.graph_study.utils import seed_everything

import re
import csv

def _extract_log_path(stdout: str):
    """Parse log path from stdout"""
    m = re.search(r"📝 Saved runtime:\s*(.*?/runtime\.json)", stdout)
    if m:
        return Path(m.group(1)).parent
    return None

def _read_r2_from_logs(log_path: Path, version: str):
    """Try to read test_r2 from test_results.json or all_test_results.csv"""
    json_path = log_path / 'default' / version / 'test_results.json'
    if json_path.exists():
        with open(json_path, 'r') as f:
            return json.load(f).get('test_r2', float('nan'))

    csv_path = log_path / 'all_test_results.csv'
    if csv_path.exists():
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('version') == version:
                    return float(row.get('test_r2', float('nan')))
    return float('nan')
    

class GraphStudyConfig:
    """Configuration class for graph construction study"""
    
    # Default parameter grids
    K_DIAG_DEFAULT = [1, 3, 5]
    K_BERT_DEFAULT = [1, 3] 
    METHODS_DEFAULT = ['faiss', 'tfidf', 'penalize']
    REWIRING_DEFAULT = ['none', 'gdc_light', 'mst']
    SEEDS_DEFAULT = [42]
    
    def __init__(self, args):
        """Initialize config from command line arguments"""
        self.k_diag = self._parse_list(args.k_diag, self.K_DIAG_DEFAULT, int)
        self.k_bert = self._parse_list(args.k_bert, self.K_BERT_DEFAULT, int)
        self.methods = self._parse_list(args.methods, self.METHODS_DEFAULT, str)
        self.rewiring = self._parse_list(args.rewiring, self.REWIRING_DEFAULT, str)
        self.seeds = self._parse_list(args.seeds, self.SEEDS_DEFAULT, int)
        
        self.num_workers = args.num_workers
        self.results_dir = Path(args.results_dir)
        self.dry_run = args.dry_run
        self.overwrite = args.overwrite
        self.debug = args.debug
        
        # Training configuration
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'raw_graphs').mkdir(exist_ok=True)
        
    def _parse_list(self, value: str, default: List, dtype):
        """Parse comma-separated string into typed list"""
        if value is None:
            return default
        return [dtype(x.strip()) for x in value.split(',')]
    
    def total_experiments(self) -> int:
        """Calculate total number of experiments"""
        return len(self.k_diag) * len(self.k_bert) * len(self.methods) * len(self.rewiring)
    
    def __str__(self):
        """String representation for logging"""
        return f"""Graph Study Configuration:
  k_diag: {self.k_diag}
  k_bert: {self.k_bert}
  methods: {self.methods}
  rewiring: {self.rewiring}
  seeds: {self.seeds}
  total_experiments: {self.total_experiments()}
  num_workers: {self.num_workers}
  results_dir: {self.results_dir}"""


def load_data_sources():
    """Load diagnosis and BERT data for graph construction"""
    try:
        # Load paths configuration
        with open('paths.json', 'r') as f:
            paths = json.load(f)
        
        MIMIC_path = paths["MIMIC_path"]
        graph_dir = paths["graph_dir"]
        
        print("Loading diagnosis data...")
        # Load diagnosis data
        train_diag = pd.read_csv(f'{MIMIC_path}train/diagnoses.csv', index_col='patient')
        val_diag = pd.read_csv(f'{MIMIC_path}val/diagnoses.csv', index_col='patient')
        test_diag = pd.read_csv(f'{MIMIC_path}test/diagnoses.csv', index_col='patient')
        all_diagnoses = pd.concat([train_diag, val_diag, test_diag], sort=False)
        
        print("Loading BERT embeddings...")
        # Load BERT embeddings
        bert_path = f"{graph_dir}bert_out.npy"
        bert_embeddings = np.load(bert_path)
        
        print(f"Loaded {len(all_diagnoses)} diagnosis records and {len(bert_embeddings)} BERT embeddings")
        
        # Verify alignment
        if len(all_diagnoses) != len(bert_embeddings):
            print(f"Warning: Size mismatch - diagnoses: {len(all_diagnoses)}, BERT: {len(bert_embeddings)}")
            min_size = min(len(all_diagnoses), len(bert_embeddings))
            all_diagnoses = all_diagnoses.iloc[:min_size]
            bert_embeddings = bert_embeddings[:min_size]
            print(f"Using aligned size: {min_size}")
        
        return all_diagnoses, bert_embeddings, paths
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def run_one_graph_experiment(params: Dict) -> Dict:
    """
    Run single graph construction + training experiment
    
    Args:
        params: Dictionary with k_diag, k_bert, method, rewiring, seeds, etc.
        
    Returns:
        Dictionary with experiment results
    """
    k_diag = params['k_diag']
    k_bert = params['k_bert']
    method = params['method']
    rewiring = params['rewiring']
    seeds = params['seeds']
    config_data = params['config_data']
    
    experiment_id = f"k{k_diag}_{k_bert}_{method}_{rewiring}"
    print(f"Running experiment: {experiment_id}")
    
    try:
        # 1. Graph construction
        all_diagnoses, bert_embeddings, paths = config_data
        
        graph_config = {
            "graph_dir": paths["graph_dir"],
            "k_diag": k_diag,
            "k_bert": k_bert,
            "diag_method": method,
            "rewiring": rewiring,
            "max_edges_per_node": 15,
            "batch_size": 1000,
            "use_gpu": True,
            "score_transform": "zscore",
            "edge_weight_balance": [1.0, 0.8]
        }
        
        # Run graph construction
        u, v, scores, types = main_graph_construction(
            graph_config,
            diagnoses_df=all_diagnoses,
            bert_embeddings=bert_embeddings
        )
        
        if u is None or v is None:
            return {
                'k_diag': k_diag, 'k_bert': k_bert,
                'method': method, 'rewiring': rewiring,
                'status': 'graph_construction_failed',
                'r2_mean': np.nan, 'edges': 0, 'density': 0.0
            }
        
        # Calculate graph statistics
        n_edges = len(u)
        n_nodes = max(np.max(u), np.max(v)) + 1
        density = n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0.0
        
        # Get graph prefix for training
        graph_prefix = graph_config.get('graph_prefix', f"gps_k{k_diag}_{k_bert}_{method[:2]}_{rewiring[:3]}_")
        
        print(f"Graph constructed: {n_edges} edges, {n_nodes} nodes, density={density:.6f}")
        
        # 2. Training with multiple seeds
        r2_results = []
        training_times = []
        
        for seed in seeds:
            print(f"Training with seed {seed}...")
            
            # Prepare training command
            train_cmd = [
                'python', '-m', 'experiments.train_mamba_gps_enhgraph',
                '--model', 'mamba-gps',
                '--log_path', str(Path(params['results_dir']) / 'logs'),
                '--ts_mask', '--add_flat', '--class_weights',
                '--add_diag', '--with_edge_types',
                '--seed', str(seed),
                '--g_version', graph_prefix.rstrip('_'),
                '--epochs', str(params.get('epochs', 15)), 
                '--batch_size', str(params.get('batch_size', 256)),
                '--lr', str(params.get('learning_rate', 1e-4)),
                '--task', 'los',
                '--version', f"{experiment_id}_seed{seed}",
                '--use_amp'
            ]
            
            # Run training
            start_time = time.time()
            try:
                result = subprocess.run(
                    train_cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )
                training_time = time.time() - start_time
                training_times.append(training_time)
                
                if result.returncode != 0:
                    print(f"Training failed for seed {seed}: {result.stderr}")
                    r2_results.append(np.nan)
                    continue
                
                # Read results
                log_path = _extract_log_path(result.stdout)
                if log_path is None:
                    print(f"Could not determine log_path for seed {seed}")
                    r2_results.append(np.nan)
                else:
                    version = f"{experiment_id}_seed{seed}"
                    r2_val = _read_r2_from_logs(log_path, version)
                    if np.isnan(r2_val):
                        print(f"Could not find R² in logs for version {version}")
                    r2_results.append(r2_val)
                    
            except subprocess.TimeoutExpired:
                print(f"Training timeout for seed {seed}")
                r2_results.append(np.nan)
                training_times.append(3600)
            except Exception as e:
                print(f"Training error for seed {seed}: {e}")
                r2_results.append(np.nan)
                training_times.append(np.nan)
        
        # 3. Aggregate results
        valid_r2 = [r for r in r2_results if not np.isnan(r)]
        
        result = {
            'k_diag': k_diag,
            'k_bert': k_bert, 
            'method': method,
            'rewiring': rewiring,
            'experiment_id': experiment_id,
            'n_edges': n_edges,
            'n_nodes': n_nodes,
            'density': density,
            'graph_prefix': graph_prefix,
            'status': 'completed' if len(valid_r2) > 0 else 'training_failed'
        }
        
        # Add per-seed results
        for i, (seed, r2) in enumerate(zip(seeds, r2_results)):
            result[f'r2_seed_{seed}'] = r2
            result[f'time_seed_{seed}'] = training_times[i] if i < len(training_times) else np.nan
        
        # Add aggregate statistics
        if len(valid_r2) > 0:
            result.update({
                'r2_mean': np.mean(valid_r2),
                'r2_std': np.std(valid_r2),
                'r2_min': np.min(valid_r2),
                'r2_max': np.max(valid_r2),
                'successful_seeds': len(valid_r2)
            })
        else:
            result.update({
                'r2_mean': np.nan,
                'r2_std': np.nan,
                'r2_min': np.nan,
                'r2_max': np.nan,
                'successful_seeds': 0
            })
        
        if len(training_times) > 0:
            valid_times = [t for t in training_times if not np.isnan(t)]
            if len(valid_times) > 0:
                result['avg_training_time'] = np.mean(valid_times)
        
        print(f"Experiment {experiment_id} completed: R²={result['r2_mean']:.4f}±{result['r2_std']:.4f}")
        return result
        
    except Exception as e:
        print(f"Experiment {experiment_id} failed: {e}")
        return {
            'k_diag': k_diag, 'k_bert': k_bert,
            'method': method, 'rewiring': rewiring,
            'experiment_id': experiment_id,
            'status': 'failed',
            'error': str(e),
            'r2_mean': np.nan,
            'n_edges': 0,
            'density': 0.0
        }


def generate_parameter_grid(config: GraphStudyConfig) -> List[Dict]:
    """Generate all parameter combinations for the study"""
    experiments = []
    
    for k_diag in config.k_diag:
        for k_bert in config.k_bert:
            for method in config.methods:
                for rewiring in config.rewiring:
                    experiment = {
                        'k_diag': k_diag,
                        'k_bert': k_bert,
                        'method': method,
                        'rewiring': rewiring,
                        'seeds': config.seeds,
                        'results_dir': config.results_dir,
                        'epochs': config.epochs,
                        'batch_size': config.batch_size,
                        'learning_rate': config.learning_rate
                    }
                    experiments.append(experiment)
    
    return experiments


def save_results_matrices(df: pd.DataFrame, results_dir: Path):
    """Save results as pivot tables for easy visualization"""
    
    # Group by method and rewiring, then create pivot tables
    for (method, rewiring), group in df.groupby(['method', 'rewiring']):
        if len(group) == 0:
            continue
            
        try:
            # Create pivot tables
            r2_matrix = group.pivot(index='k_diag', columns='k_bert', values='r2_mean')
            edges_matrix = group.pivot(index='k_diag', columns='k_bert', values='n_edges')
            density_matrix = group.pivot(index='k_diag', columns='k_bert', values='density')
            
            # Save matrices
            prefix = f"{method}_{rewiring}"
            r2_matrix.to_csv(results_dir / f"r2_{prefix}.csv")
            edges_matrix.to_csv(results_dir / f"edges_{prefix}.csv")
            density_matrix.to_csv(results_dir / f"density_{prefix}.csv")
            
            print(f"Saved matrices for {method}-{rewiring}")
            
        except Exception as e:
            print(f"Error creating matrices for {method}-{rewiring}: {e}")


def run_graph_study(config: GraphStudyConfig):
    """Main function to run the complete graph construction study"""
    
    print("="*80)
    print("GRAPH CONSTRUCTION STUDY FOR MAMBA-GPS ICU LOS PREDICTION")
    print("="*80)
    print(config)
    print("="*80)
    
    # Check if results already exist
    results_file = config.results_dir / "all_results.csv"
    if results_file.exists() and not config.overwrite:
        print(f"Results file {results_file} already exists. Use --overwrite to rerun.")
        return
    
    # Load data once for all experiments
    print("Loading data sources...")
    config_data = load_data_sources()
    
    # Generate parameter grid
    experiments = generate_parameter_grid(config)
    print(f"Generated {len(experiments)} experiments")
    
    if config.dry_run:
        print("DRY RUN - Would run the following experiments:")
        for exp in experiments[:5]:  # Show first 5
            print(f"  k_diag={exp['k_diag']}, k_bert={exp['k_bert']}, "
                  f"method={exp['method']}, rewiring={exp['rewiring']}")
        if len(experiments) > 5:
            print(f"  ... and {len(experiments) - 5} more")
        return
    
    # Add config data to each experiment
    for exp in experiments:
        exp['config_data'] = config_data
    
    # Run experiments
    print(f"Running {len(experiments)} experiments with {config.num_workers} workers...")
    
    results = []
    
    if config.num_workers == 1:
        # Sequential execution with progress bar
        for exp in tqdm(experiments, desc="Running experiments"):
            result = run_one_graph_experiment(exp)
            results.append(result)
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
            # Submit all experiments
            future_to_exp = {
                executor.submit(run_one_graph_experiment, exp): exp 
                for exp in experiments
            }
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_exp), total=len(experiments), desc="Running experiments"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    exp = future_to_exp[future]
                    print(f"Experiment failed: {exp['method']}-{exp['rewiring']}-k{exp['k_diag']}_{exp['k_bert']}: {e}")
                    # Add failed result
                    results.append({
                        'k_diag': exp['k_diag'], 'k_bert': exp['k_bert'],
                        'method': exp['method'], 'rewiring': exp['rewiring'],
                        'status': 'failed', 'error': str(e),
                        'r2_mean': np.nan, 'n_edges': 0, 'density': 0.0
                    })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(results_file, index=False)
    print(f"Saved all results to {results_file}")
    
    # Generate summary statistics
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    total_exp = len(df)
    completed = len(df[df['status'] == 'completed'])
    failed = len(df[df['status'].isin(['failed', 'training_failed', 'graph_construction_failed'])])
    
    print(f"Total experiments: {total_exp}")
    print(f"Completed successfully: {completed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {completed/total_exp*100:.1f}%")
    
    if completed > 0:
        valid_results = df[df['status'] == 'completed']
        print(f"\nR² Statistics (successful experiments):")
        print(f"  Mean: {valid_results['r2_mean'].mean():.4f} ± {valid_results['r2_mean'].std():.4f}")
        print(f"  Range: [{valid_results['r2_mean'].min():.4f}, {valid_results['r2_mean'].max():.4f}]")
        
        print(f"\nGraph Size Statistics:")
        print(f"  Edges - Mean: {valid_results['n_edges'].mean():.0f}, Range: [{valid_results['n_edges'].min():.0f}, {valid_results['n_edges'].max():.0f}]")
        print(f"  Density - Mean: {valid_results['density'].mean():.6f}, Range: [{valid_results['density'].min():.6f}, {valid_results['density'].max():.6f}]")
        
        # Save pivot tables for visualization
        save_results_matrices(valid_results, config.results_dir)
        
        # Best configuration
        best_idx = valid_results['r2_mean'].idxmax()
        best_config = valid_results.loc[best_idx]
        print(f"\nBest Configuration:")
        print(f"  k_diag={best_config['k_diag']}, k_bert={best_config['k_bert']}")
        print(f"  method={best_config['method']}, rewiring={best_config['rewiring']}")
        print(f"  R² = {best_config['r2_mean']:.4f} ± {best_config['r2_std']:.4f}")
        print(f"  Edges = {best_config['n_edges']}, Density = {best_config['density']:.6f}")
    
    print("\nStudy completed! Results saved to:", config.results_dir)


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Comprehensive graph construction study for Mamba-GPS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Parameter grid arguments
    parser.add_argument('--k_diag', type=str, default=None,
                        help='Comma-separated k_diag values (e.g., "1,3,5")')
    parser.add_argument('--k_bert', type=str, default=None,
                        help='Comma-separated k_bert values (e.g., "1,3,5")')
    parser.add_argument('--methods', type=str, default=None,
                        help='Comma-separated methods (e.g., "faiss,tfidf,penalize")')
    parser.add_argument('--rewiring', type=str, default=None,
                        help='Comma-separated rewiring methods (e.g., "none,gdc_light,mst")')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated random seeds (e.g., "42,43,44")')
    
    # Execution arguments
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--results_dir', type=str, default='results/graph',
                        help='Directory to save results')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print experiment list without running')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing results')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = GraphStudyConfig(args)
    
    # Set random seed for reproducibility
    seed_everything(42)
    
    # Run the study
    run_graph_study(config)


if __name__ == "__main__":
    main()