# src/evaluation/tracking/aggregate.py
"""
Simple results aggregator for baseline comparison
Handles the actual complex directory structure with minimal code
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats


def find_result_files(base_dir: Path) -> Dict[str, List[Dict]]:
    """
    Find all result files in the complex directory structure
    
    Returns:
        Dict mapping model_name -> list of seed results
    """
    results = {}
    
    # Handle baselines directory
    baselines_dir = base_dir / "baselines"
    if baselines_dir.exists():
        for model_dir in baselines_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            results[model_name] = []
            
            # Find seed directories
            for seed_dir in model_dir.glob("seed_*"):
                seed = seed_dir.name.split('_')[1]
                
                # Find result files with comprehensive search
                result_files = {}

                # 1. Look for runtime.json in seed directory (priority)
                runtime_candidates = [
                    seed_dir / "runtime.json",  # Direct in seed dir
                    seed_dir / "runtime"        # Also check without .json
                ]
                
                for runtime_file in runtime_candidates:
                    if runtime_file.exists() and runtime_file.is_file():
                        result_files["runtime"] = runtime_file
                        break
                
                # 2. If not found, search recursively
                if "runtime" not in result_files:
                    for runtime_file in seed_dir.rglob("runtime*"):
                        if runtime_file.is_file() and not runtime_file.name.startswith('.'):
                            result_files["runtime"] = runtime_file
                            break
                
                # 3. Find test and validation results
                for json_file in seed_dir.rglob("*.json"):
                    if json_file.name == "test_results.json" and "test_results.json" not in result_files:
                        result_files["test_results.json"] = json_file
                    elif json_file.name == "valid_results.json" and "valid_results.json" not in result_files:
                        result_files["valid_results.json"] = json_file
                
                if result_files:
                    results[model_name].append({
                        'seed': seed,
                        'files': result_files
                    })
    
    # Handle mamba-gps directory (check multiple possible locations)
    mamba_gps_locations = [
        base_dir / "mamba-gps",
        base_dir / "mamba_gps", 
        base_dir / "baselines" / "mamba_gps",
        base_dir / "baselines" / "mamba-gps"
    ]
    
    for mamba_gps_dir in mamba_gps_locations:
        if mamba_gps_dir.exists():
            if 'mamba_gps' not in results:
                results['mamba_gps'] = []
                
            for seed_dir in mamba_gps_dir.glob("seed_*"):
                seed = seed_dir.name.split('_')[1]
                
                result_files = {}
                
                # Same runtime search logic
                runtime_candidates = [
                    seed_dir / "runtime.json",
                    seed_dir / "runtime"
                ]
                
                for runtime_file in runtime_candidates:
                    if runtime_file.exists() and runtime_file.is_file():
                        result_files["runtime"] = runtime_file
                        break
                
                if "runtime" not in result_files:
                    for runtime_file in seed_dir.rglob("runtime*"):
                        if runtime_file.is_file() and not runtime_file.name.startswith('.'):
                            result_files["runtime"] = runtime_file
                            break
                
                # Find test and validation results
                for json_file in seed_dir.rglob("*.json"):
                    if json_file.name == "test_results.json" and "test_results.json" not in result_files:
                        result_files["test_results.json"] = json_file
                    elif json_file.name == "valid_results.json" and "valid_results.json" not in result_files:
                        result_files["valid_results.json"] = json_file
                
                if result_files:
                    results['mamba_gps'].append({
                        'seed': seed,
                        'files': result_files
                    })
            break  # Stop after finding the first valid mamba-gps directory
    
    return results


def load_json_safe(filepath: Path) -> Dict:
    """Safely load JSON file"""
    try:
        if not filepath or not filepath.exists() or filepath.is_dir():
            return {}
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return {}


def extract_best_metrics(test_data: Dict, valid_data: Dict) -> Dict[str, float]:
    """
    Extract the best metrics from test/valid results
    For dual models (lstm+gnn), use the combined/gnn results as primary
    """
    metrics = {}
    
    # Primary metrics (use test data, fallback to valid)
    metric_names = ['r2', 'mse', 'mad', 'mape', 'kappa', 'msle']
    
    for metric in metric_names:
        # Try different naming patterns
        candidates = [
            f'test_{metric}',           # Standard test metric
            f'test_{metric}_gnn',       # GNN component for dual models
            f'valid_{metric}',          # Fallback to validation
            f'{metric}',                # Direct metric name
        ]
        
        value = None
        for candidate in candidates:
            if candidate in test_data:
                value = test_data[candidate]
                break
            elif candidate in valid_data:
                value = valid_data[candidate]
                break
        
        metrics[metric] = float(value) if value is not None else 0.0
    
    return metrics


def extract_runtime_stats(runtime_data: Dict) -> Dict[str, float]:
    """Extract runtime statistics from either format of runtime file"""
    if not runtime_data:
        return {
            'train_hours': 0.0,
            'peak_vram_GB': 0.0,
            'total_params': 0,
            'inference_ms': 0.0,
            'model_size_mb': 0.0,
        }
    
    return {
        'train_hours': runtime_data.get('train_hours', 0.0),
        'peak_vram_GB': runtime_data.get('peak_vram_GB', 0.0),
        'total_params': runtime_data.get('total_params', 0),
        'inference_ms': runtime_data.get('inference_ms_per_patient', 0.0),
        'model_size_mb': runtime_data.get('model_size_mb', 0.0),
    }


def calculate_stats(values: List[float]) -> Dict[str, float]:
    """Calculate mean, std, and format for display"""
    if not values:
        return {'mean': 0.0, 'std': 0.0, 'formatted': '0.000±0.000', 'values': []}
    
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
    
    return {
        'mean': mean_val,
        'std': std_val,
        'formatted': f"{mean_val:.3f}±{std_val:.3f}",
        'values': values
    }


def format_params(param_count: int) -> str:
    """Format parameter count"""
    if param_count >= 1_000_000:
        return f"{param_count / 1_000_000:.1f}M"
    elif param_count >= 1_000:
        return f"{param_count / 1_000:.1f}K"
    else:
        return str(param_count)


def perform_significance_test(values1: List[float], values2: List[float]) -> float:
    """Perform paired t-test"""
    if len(values1) < 2 or len(values2) < 2:
        return 1.0
    try:
        _, p_value = stats.ttest_ind(values1, values2)
        return p_value
    except:
        return 1.0


def aggregate_all_results(results_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main aggregation function
    
    Returns:
        summary_df: Main comparison table
        scatter_df: Scatter plot data
    """
    print(f"🔍 Scanning results in {results_dir}")
    
    # Find all result files
    all_results = find_result_files(results_dir)
    
    if not all_results:
        print("❌ No results found!")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"📊 Found results for {len(all_results)} models")
    
    # Debug: Show what files were found
    for model_name, seed_results in all_results.items():
        print(f"  • {model_name}: {len(seed_results)} seeds")
        for seed_data in seed_results:
            files = seed_data['files']
            file_list = list(files.keys())
            print(f"    - Seed {seed_data['seed']}: {file_list}")
            
            # Show runtime file path if found
            if 'runtime' in files:
                runtime_path = files['runtime']
                print(f"      Runtime: {runtime_path}")
            else:
                print(f"      ⚠️  No runtime file found")
    print()
    
    # Process each model
    summary_data = []
    
    for model_name, seed_results in all_results.items():
        if not seed_results:
            continue
        
        # Collect metrics across seeds
        model_metrics = {metric: [] for metric in ['r2', 'mse', 'mad', 'mape', 'kappa', 'msle']}
        runtime_stats = {'train_hours': [], 'peak_vram_GB': [], 'total_params': [], 'inference_ms': []}
        
        for seed_data in seed_results:
            files = seed_data['files']
            
            # Load data safely
            test_data = load_json_safe(files.get('test_results.json'))
            valid_data = load_json_safe(files.get('valid_results.json'))
            runtime_data = load_json_safe(files.get('runtime'))
            
            # Debug print to check data loading
            if not test_data and not valid_data:
                print(f"  Warning: No test/valid data for {model_name} seed {seed_data['seed']}")
                continue
                
            # Extract metrics
            metrics = extract_best_metrics(test_data, valid_data)
            runtime = extract_runtime_stats(runtime_data)
            
            # Collect values
            for metric in model_metrics:
                model_metrics[metric].append(metrics[metric])
            
            for stat in runtime_stats:
                runtime_stats[stat].append(runtime[stat])
        
        # Calculate statistics
        row = {'Model': model_name}
        
        # Add formatted metrics
        for metric in model_metrics:
            stats_dict = calculate_stats(model_metrics[metric])
            row[f'{metric.upper()}'] = stats_dict['formatted']
            row[f'{metric}_mean'] = stats_dict['mean']
            row[f'{metric}_values'] = stats_dict['values']
        
        # Add runtime info - GPU-h with 3 decimal places, no inference time
        row['Params'] = format_params(int(np.mean(runtime_stats['total_params']))) if runtime_stats['total_params'] else 'N/A'
        row['GPU-h'] = f"{np.mean(runtime_stats['train_hours']):.3f}" if runtime_stats['train_hours'] else 'N/A'
        row['VRAM (GB)'] = f"{np.mean(runtime_stats['peak_vram_GB']):.1f}" if runtime_stats['peak_vram_GB'] else 'N/A'
        
        # Store raw values for significance testing
        row['peak_vram_mean'] = np.mean(runtime_stats['peak_vram_GB']) if runtime_stats['peak_vram_GB'] else 0.0
        
        summary_data.append(row)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    if summary_df.empty:
        print("❌ No valid data to aggregate")
        return pd.DataFrame(), pd.DataFrame()
    
    # Add significance markers
    summary_df = add_significance_markers(summary_df)
    
    # Create scatter data
    scatter_data = []
    for _, row in summary_df.iterrows():
        scatter_data.append({
            'model': row['Model'],
            'peak_vram_GB': row['peak_vram_mean'],
            'r2_mean': row['r2_mean'],
            'r2_std': calculate_stats(row['r2_values'])['std']
        })
    
    scatter_df = pd.DataFrame(scatter_data)
    
    # Clean up summary table for final output - removed 'Inference (ms)'
    display_cols = ['Model', 'Params', 'GPU-h', 'MSE', 'MAD', 'MAPE', 'R2', 'KAPPA', 'MSLE', 'VRAM (GB)']
    final_summary = summary_df[display_cols].copy()
    
    print(f"✅ Aggregated results for {len(final_summary)} models")
    return final_summary, scatter_df


def add_significance_markers(df: pd.DataFrame) -> pd.DataFrame:
    """Add † markers for statistically significant best results"""
    df = df.copy()
    
    # For higher-is-better metrics
    for metric in ['R2', 'KAPPA']:
        mean_col = f'{metric.lower()}_mean'
        values_col = f'{metric.lower()}_values'
        
        if mean_col in df.columns and values_col in df.columns:
            # Sort by mean performance
            sorted_df = df.sort_values(mean_col, ascending=False)
            
            if len(sorted_df) >= 2:
                best_row = sorted_df.iloc[0]
                second_best_row = sorted_df.iloc[1]
                
                # Perform significance test
                p_value = perform_significance_test(
                    best_row[values_col], 
                    second_best_row[values_col]
                )
                
                if p_value < 0.05:
                    # Add † to best model
                    best_idx = df[df['Model'] == best_row['Model']].index[0]
                    current_value = df.loc[best_idx, metric]
                    df.loc[best_idx, metric] = current_value + '†'
    
    # For lower-is-better metrics
    for metric in ['MSE', 'MAD', 'MAPE', 'MSLE']:
        mean_col = f'{metric.lower()}_mean'
        values_col = f'{metric.lower()}_values'
        
        if mean_col in df.columns and values_col in df.columns:
            # Sort by mean performance (ascending for lower-is-better)
            sorted_df = df.sort_values(mean_col, ascending=True)
            
            if len(sorted_df) >= 2:
                best_row = sorted_df.iloc[0]
                second_best_row = sorted_df.iloc[1]
                
                # Perform significance test
                p_value = perform_significance_test(
                    best_row[values_col], 
                    second_best_row[values_col]
                )
                
                if p_value < 0.05:
                    # Add † to best model
                    best_idx = df[df['Model'] == best_row['Model']].index[0]
                    current_value = df.loc[best_idx, metric]
                    df.loc[best_idx, metric] = current_value + '†'
    
    return df


def save_results(summary_df: pd.DataFrame, scatter_df: pd.DataFrame, output_dir: Path):
    """Save aggregated results"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main table
    summary_df.to_csv(output_dir / 'summary.csv', index=False)
    print(f"📄 Main table saved: {output_dir / 'summary.csv'}")
    
    # Save scatter data
    scatter_df.to_csv(output_dir / 'scatter_data.csv', index=False)
    print(f"📈 Scatter data saved: {output_dir / 'scatter_data.csv'}")
    
    # Print summary
    print(f"\n📊 Summary ({len(summary_df)} models):")
    print(summary_df.to_string(index=False))


def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Aggregate experimental results')
    parser.add_argument('--results_dir', type=Path, default='results',
                       help='Base results directory')
    parser.add_argument('--output_dir', type=Path, default='results/sum_results',
                       help='Output directory for summary tables')
    parser.add_argument('--debug', action='store_true',
                       help='Show detailed file discovery information')
    
    args = parser.parse_args()
    
    # Check if results directory exists
    if not args.results_dir.exists():
        print(f"❌ Results directory not found: {args.results_dir}")
        return
    
    if args.debug:
        print("🔍 Debug mode: Checking directory structure...")
        debug_directory_structure(args.results_dir)
        return
    
    # Aggregate results
    summary_df, scatter_df = aggregate_all_results(args.results_dir)
    
    if summary_df.empty:
        print("❌ No results to aggregate")
        return
    
    # Save results
    save_results(summary_df, scatter_df, args.output_dir)
    
    print(f"\n🎉 Aggregation complete!")
    print(f"   Summary table: {args.output_dir / 'summary.csv'}")
    print(f"   Scatter data: {args.output_dir / 'scatter_data.csv'}")


def debug_directory_structure(results_dir: Path):
    """Debug function to show directory structure and found files"""
    print(f"Scanning directory: {results_dir}")
    
    # Check baselines
    baselines_dir = results_dir / "baselines"
    if baselines_dir.exists():
        print(f"\n📁 Baselines directory: {baselines_dir}")
        for model_dir in sorted(baselines_dir.iterdir()):
            if model_dir.is_dir():
                print(f"  • {model_dir.name}/")
                for seed_dir in sorted(model_dir.glob("seed_*")):
                    print(f"    - {seed_dir.name}/")
                    
                    # Check for runtime files with detailed search
                    runtime_files_found = []
                    
                    # Check direct files
                    for candidate in ["runtime.json", "runtime"]:
                        candidate_path = seed_dir / candidate
                        if candidate_path.exists():
                            runtime_files_found.append(f"✅ {candidate} (direct)")
                    
                    # Check recursive files
                    for runtime_file in seed_dir.rglob("runtime*"):
                        if runtime_file.is_file():
                            rel_path = runtime_file.relative_to(seed_dir)
                            runtime_files_found.append(f"📁 {runtime_file.name}: {rel_path}")
                    
                    if runtime_files_found:
                        for rf in runtime_files_found:
                            print(f"      {rf}")
                    else:
                        print(f"      ❌ No runtime files found")
                    
                    # Check for result files
                    test_files = list(seed_dir.rglob("test_results.json"))
                    valid_files = list(seed_dir.rglob("valid_results.json"))
                    
                    if test_files:
                        print(f"      ✅ test_results.json: {test_files[0].relative_to(seed_dir)}")
                    else:
                        print(f"      ❌ test_results.json not found")
                        
                    if valid_files:
                        print(f"      ✅ valid_results.json: {valid_files[0].relative_to(seed_dir)}")
                    else:
                        print(f"      ❌ valid_results.json not found")
    else:
        print(f"❌ Baselines directory not found: {baselines_dir}")
    
    # Check mamba-gps locations
    print(f"\n📁 Mamba-GPS search:")
    mamba_gps_locations = [
        results_dir / "mamba-gps",
        results_dir / "mamba_gps", 
        results_dir / "baselines" / "mamba_gps",
        results_dir / "baselines" / "mamba-gps"
    ]
    
    found_mamba = False
    for location in mamba_gps_locations:
        if location.exists():
            print(f"  ✅ Found: {location}")
            found_mamba = True
            # Show contents
            for seed_dir in sorted(location.glob("seed_*")):
                print(f"    - {seed_dir.name}/")
        else:
            print(f"  ❌ Not found: {location}")
    
    if not found_mamba:
        print(f"  ⚠️  No mamba-gps directory found in any expected location")
        
    # Show overall directory structure
    print(f"\n📂 Overall structure in {results_dir}:")
    for item in sorted(results_dir.iterdir()):
        if item.is_dir():
            print(f"  📁 {item.name}/")
        else:
            print(f"  📄 {item.name}")



if __name__ == "__main__":
    main()