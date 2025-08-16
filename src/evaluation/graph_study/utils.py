"""
Utility functions for graph construction study
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns


def load_experiment_results(results_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load experiment results from CSV file
    
    Args:
        results_dir: Directory containing results
        
    Returns:
        DataFrame with experiment results
    """
    results_file = Path(results_dir) / "all_results.csv"
    
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    df = pd.read_csv(results_file)
    
    # Filter successful experiments
    successful = df[df['status'] == 'completed'].copy()
    
    print(f"Loaded {len(df)} total experiments, {len(successful)} successful")
    
    return successful


def create_summary_statistics(df: pd.DataFrame) -> Dict:
    """
    Generate comprehensive summary statistics
    
    Args:
        df: DataFrame with experiment results
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_experiments': len(df),
        'parameter_coverage': {
            'k_diag_values': sorted(df['k_diag'].unique()),
            'k_bert_values': sorted(df['k_bert'].unique()),
            'methods': sorted(df['method'].unique()),
            'rewiring': sorted(df['rewiring'].unique())
        }
    }
    
    # Performance statistics
    summary['performance'] = {
        'r2_mean': df['r2_mean'].mean(),
        'r2_std': df['r2_mean'].std(),
        'r2_min': df['r2_mean'].min(),
        'r2_max': df['r2_mean'].max(),
        'r2_median': df['r2_mean'].median()
    }
    
    # Graph statistics
    summary['graph_stats'] = {
        'edges_mean': df['n_edges'].mean(),
        'edges_std': df['n_edges'].std(),
        'edges_min': df['n_edges'].min(),
        'edges_max': df['n_edges'].max(),
        'density_mean': df['density'].mean(),
        'density_std': df['density'].std(),
        'density_min': df['density'].min(),
        'density_max': df['density'].max()
    }
    
    # Best configurations
    best_idx = df['r2_mean'].idxmax()
    best_config = df.loc[best_idx]
    
    summary['best_configuration'] = {
        'k_diag': int(best_config['k_diag']),
        'k_bert': int(best_config['k_bert']),
        'method': best_config['method'],
        'rewiring': best_config['rewiring'],
        'r2_mean': float(best_config['r2_mean']),
        'r2_std': float(best_config['r2_std']),
        'n_edges': int(best_config['n_edges']),
        'density': float(best_config['density'])
    }
    
    # Method comparisons
    method_comparison = df.groupby('method')['r2_mean'].agg(['mean', 'std', 'count']).to_dict('index')
    summary['method_comparison'] = method_comparison
    
    # Rewiring comparisons
    rewiring_comparison = df.groupby('rewiring')['r2_mean'].agg(['mean', 'std', 'count']).to_dict('index')
    summary['rewiring_comparison'] = rewiring_comparison
    
    return summary


def analyze_correlation_patterns(df: pd.DataFrame) -> Dict:
    """
    Analyze correlations between parameters and performance
    
    Args:
        df: DataFrame with experiment results
        
    Returns:
        Dictionary with correlation analysis
    """
    # Encode categorical variables
    df_numeric = df.copy()
    df_numeric['method_encoded'] = pd.Categorical(df['method']).codes
    df_numeric['rewiring_encoded'] = pd.Categorical(df['rewiring']).codes
    
    # Calculate correlations
    correlation_vars = ['k_diag', 'k_bert', 'method_encoded', 'rewiring_encoded', 
                       'n_edges', 'density', 'r2_mean']
    corr_matrix = df_numeric[correlation_vars].corr()
    
    # Focus on correlations with performance
    r2_correlations = corr_matrix['r2_mean'].drop('r2_mean').to_dict()
    
    analysis = {
        'correlation_matrix': corr_matrix.to_dict(),
        'r2_correlations': r2_correlations,
        'strongest_positive_correlation': max(r2_correlations.items(), key=lambda x: x[1]),
        'strongest_negative_correlation': min(r2_correlations.items(), key=lambda x: x[1])
    }
    
    return analysis


def identify_pareto_frontier(df: pd.DataFrame, 
                           performance_col: str = 'r2_mean',
                           efficiency_col: str = 'density',
                           maximize_performance: bool = True,
                           minimize_efficiency: bool = True) -> pd.DataFrame:
    """
    Identify configurations on the Pareto frontier for performance vs efficiency
    
    Args:
        df: DataFrame with experiment results
        performance_col: Column name for performance metric
        efficiency_col: Column name for efficiency metric (e.g., density, edges)
        maximize_performance: Whether to maximize performance
        minimize_efficiency: Whether to minimize efficiency metric
        
    Returns:
        DataFrame with Pareto optimal configurations
    """
    df_copy = df.copy()
    
    # Adjust signs for minimization/maximization
    perf_values = df_copy[performance_col].values
    eff_values = df_copy[efficiency_col].values
    
    if not maximize_performance:
        perf_values = -perf_values
    if not minimize_efficiency:
        eff_values = -eff_values
    
    # Find Pareto frontier
    pareto_mask = np.ones(len(df_copy), dtype=bool)
    
    for i in range(len(df_copy)):
        if pareto_mask[i]:
            # Check if any other point dominates this one
            dominated = (
                (perf_values >= perf_values[i]) & 
                (eff_values <= eff_values[i]) &
                ((perf_values > perf_values[i]) | (eff_values < eff_values[i]))
            )
            pareto_mask[dominated] = False
    
    pareto_df = df_copy[pareto_mask].copy()
    pareto_df = pareto_df.sort_values(performance_col, ascending=not maximize_performance)
    
    return pareto_df


def generate_ranking_analysis(df: pd.DataFrame) -> Dict:
    """
    Generate ranking analysis for different aspects
    
    Args:
        df: DataFrame with experiment results
        
    Returns:
        Dictionary with ranking information
    """
    rankings = {}
    
    # Overall performance ranking
    df_sorted = df.sort_values('r2_mean', ascending=False)
    rankings['performance'] = df_sorted[['k_diag', 'k_bert', 'method', 'rewiring', 'r2_mean']].head(10).to_dict('records')
    
    # Efficiency ranking (high performance, low edges)
    df['efficiency_score'] = df['r2_mean'] / (df['n_edges'] / 1000)  # Normalize edges
    df_eff_sorted = df.sort_values('efficiency_score', ascending=False)
    rankings['efficiency'] = df_eff_sorted[['k_diag', 'k_bert', 'method', 'rewiring', 'r2_mean', 'n_edges', 'efficiency_score']].head(10).to_dict('records')
    
    # Robustness ranking (low standard deviation)
    df_robust = df[df['r2_std'].notna()].sort_values('r2_std', ascending=True)
    rankings['robustness'] = df_robust[['k_diag', 'k_bert', 'method', 'rewiring', 'r2_mean', 'r2_std']].head(10).to_dict('records')
    
    # Method-specific rankings
    rankings['by_method'] = {}
    for method in df['method'].unique():
        method_df = df[df['method'] == method].sort_values('r2_mean', ascending=False)
        rankings['by_method'][method] = method_df[['k_diag', 'k_bert', 'rewiring', 'r2_mean']].head(5).to_dict('records')
    
    return rankings


def export_results_for_publication(df: pd.DataFrame, output_dir: Union[str, Path]):
    """
    Export results in publication-ready formats
    
    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save publication files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Main results table
    main_results = df[['k_diag', 'k_bert', 'method', 'rewiring', 
                      'r2_mean', 'r2_std', 'n_edges', 'density']].copy()
    main_results.columns = ['k_diag', 'k_bert', 'Method', 'Rewiring', 
                           'R²_mean', 'R²_std', 'Edges', 'Density']
    main_results.to_csv(output_dir / 'main_results.csv', index=False)
    
    # 2. Summary statistics
    summary = create_summary_statistics(df)
    with open(output_dir / 'summary_statistics.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # 3. Correlation analysis
    correlations = analyze_correlation_patterns(df)
    with open(output_dir / 'correlation_analysis.json', 'w') as f:
        json.dump(correlations, f, indent=2, default=str)
    
    # 4. Rankings
    rankings = generate_ranking_analysis(df)
    with open(output_dir / 'rankings.json', 'w') as f:
        json.dump(rankings, f, indent=2, default=str)
    
    # 5. Pareto frontier
    pareto_performance_density = identify_pareto_frontier(df, 'r2_mean', 'density')
    pareto_performance_edges = identify_pareto_frontier(df, 'r2_mean', 'n_edges')
    
    pareto_performance_density.to_csv(output_dir / 'pareto_frontier_density.csv', index=False)
    pareto_performance_edges.to_csv(output_dir / 'pareto_frontier_edges.csv', index=False)
    
    # 6. Method comparison table
    method_stats = df.groupby('method').agg({
        'r2_mean': ['mean', 'std', 'min', 'max', 'count'],
        'n_edges': ['mean', 'std'],
        'density': ['mean', 'std']
    }).round(4)
    method_stats.to_csv(output_dir / 'method_comparison.csv')
    
    # 7. Rewiring comparison table
    rewiring_stats = df.groupby('rewiring').agg({
        'r2_mean': ['mean', 'std', 'min', 'max', 'count'],
        'n_edges': ['mean', 'std'],
        'density': ['mean', 'std']
    }).round(4)
    rewiring_stats.to_csv(output_dir / 'rewiring_comparison.csv')
    
    print(f"Publication-ready results exported to {output_dir}")


def validate_experiment_completeness(df: pd.DataFrame, 
                                   expected_k_diag: List[int],
                                   expected_k_bert: List[int],
                                   expected_methods: List[str],
                                   expected_rewiring: List[str]) -> Dict:
    """
    Validate that all expected parameter combinations were tested
    
    Args:
        df: DataFrame with experiment results
        expected_*: Lists of expected parameter values
        
    Returns:
        Dictionary with validation results
    """
    # Generate expected combinations
    expected_combinations = []
    for k_d in expected_k_diag:
        for k_b in expected_k_bert:
            for method in expected_methods:
                for rewiring in expected_rewiring:
                    expected_combinations.append((k_d, k_b, method, rewiring))
    
    # Get actual combinations
    actual_combinations = set(
        zip(df['k_diag'], df['k_bert'], df['method'], df['rewiring'])
    )
    expected_combinations = set(expected_combinations)
    
    missing = expected_combinations - actual_combinations
    extra = actual_combinations - expected_combinations
    
    validation = {
        'total_expected': len(expected_combinations),
        'total_actual': len(actual_combinations),
        'missing_combinations': list(missing),
        'extra_combinations': list(extra),
        'completion_rate': len(actual_combinations) / len(expected_combinations),
        'is_complete': len(missing) == 0
    }
    
    return validation


def create_results_summary_report(results_dir: Union[str, Path]) -> str:
    """
    Create a comprehensive text summary report
    
    Args:
        results_dir: Directory containing results
        
    Returns:
        String with formatted report
    """
    df = load_experiment_results(results_dir)
    summary = create_summary_statistics(df)
    rankings = generate_ranking_analysis(df)
    
    report = []
    report.append("GRAPH CONSTRUCTION STUDY - RESULTS SUMMARY")
    report.append("=" * 60)
    report.append("")
    
    # Overall statistics
    report.append(f"Total successful experiments: {summary['total_experiments']}")
    report.append(f"Parameter ranges tested:")
    report.append(f"  k_diag: {summary['parameter_coverage']['k_diag_values']}")
    report.append(f"  k_bert: {summary['parameter_coverage']['k_bert_values']}")
    report.append(f"  Methods: {summary['parameter_coverage']['methods']}")
    report.append(f"  Rewiring: {summary['parameter_coverage']['rewiring']}")
    report.append("")
    
    # Performance summary
    perf = summary['performance']
    report.append("PERFORMANCE SUMMARY")
    report.append("-" * 30)
    report.append(f"R² Mean: {perf['r2_mean']:.4f} ± {perf['r2_std']:.4f}")
    report.append(f"R² Range: [{perf['r2_min']:.4f}, {perf['r2_max']:.4f}]")
    report.append(f"R² Median: {perf['r2_median']:.4f}")
    report.append("")
    
    # Graph statistics
    graph = summary['graph_stats']
    report.append("GRAPH SIZE STATISTICS")
    report.append("-" * 30)
    report.append(f"Edges - Mean: {graph['edges_mean']:.0f} ± {graph['edges_std']:.0f}")
    report.append(f"Edges - Range: [{graph['edges_min']:.0f}, {graph['edges_max']:.0f}]")
    report.append(f"Density - Mean: {graph['density_mean']:.6f} ± {graph['density_std']:.6f}")
    report.append(f"Density - Range: [{graph['density_min']:.6f}, {graph['density_max']:.6f}]")
    report.append("")
    
    # Best configuration
    best = summary['best_configuration']
    report.append("BEST CONFIGURATION")
    report.append("-" * 30)
    report.append(f"k_diag={best['k_diag']}, k_bert={best['k_bert']}")
    report.append(f"Method: {best['method']}, Rewiring: {best['rewiring']}")
    report.append(f"Performance: R² = {best['r2_mean']:.4f} ± {best['r2_std']:.4f}")
    report.append(f"Graph: {best['n_edges']} edges, density = {best['density']:.6f}")
    report.append("")
    
    # Method comparison
    report.append("METHOD COMPARISON")
    report.append("-" * 30)
    for method, stats in summary['method_comparison'].items():
        report.append(f"{method:>10}: R² = {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
    report.append("")
    
    # Rewiring comparison
    report.append("REWIRING COMPARISON")
    report.append("-" * 30)
    for rewiring, stats in summary['rewiring_comparison'].items():
        report.append(f"{rewiring:>10}: R² = {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
    report.append("")
    
    # Top 5 configurations
    report.append("TOP 5 CONFIGURATIONS")
    report.append("-" * 30)
    for i, config in enumerate(rankings['performance'][:5], 1):
        report.append(f"{i}. k_diag={config['k_diag']}, k_bert={config['k_bert']}, "
                     f"{config['method']}, {config['rewiring']}: R² = {config['r2_mean']:.4f}")
    
    return "\n".join(report)

def seed_everything(seed: int = 42):
    """Set random seed for reproducibility across libraries"""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For PyTorch Lightning, if used
    try:
        import pytorch_lightning as pl
        pl.seed_everything(seed)
    except ImportError:
        pass

    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_pivot_tables_from_dataframe(df: pd.DataFrame, output_dir: Union[str, Path]):
    """
    Save pivot tables for all method-rewiring combinations
    
    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save pivot tables
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save overall pivot tables
    overall_r2 = df.pivot_table(index='k_diag', columns='k_bert', values='r2_mean', aggfunc='mean')
    overall_edges = df.pivot_table(index='k_diag', columns='k_bert', values='n_edges', aggfunc='mean')
    overall_density = df.pivot_table(index='k_diag', columns='k_bert', values='density', aggfunc='mean')
    
    overall_r2.to_csv(output_dir / 'overall_r2.csv')
    overall_edges.to_csv(output_dir / 'overall_edges.csv')
    overall_density.to_csv(output_dir / 'overall_density.csv')
    
    # Save method-rewiring specific tables
    for (method, rewiring), group in df.groupby(['method', 'rewiring']):
        if len(group) == 0:
            continue
            
        try:
            prefix = f"{method}_{rewiring}"
            
            r2_pivot = group.pivot(index='k_diag', columns='k_bert', values='r2_mean')
            edges_pivot = group.pivot(index='k_diag', columns='k_bert', values='n_edges')
            density_pivot = group.pivot(index='k_diag', columns='k_bert', values='density')
            
            r2_pivot.to_csv(output_dir / f'r2_{prefix}.csv')
            edges_pivot.to_csv(output_dir / f'edges_{prefix}.csv')
            density_pivot.to_csv(output_dir / f'density_{prefix}.csv')
            
        except Exception as e:
            print(f"Warning: Could not create pivot table for {method}-{rewiring}: {e}")
    
    print(f"Pivot tables saved to {output_dir}")