# src/evaluation/tracking/tracking_viz.py
"""
Publication-quality visualization script for baseline comparison results
Modified to directly read existing table data without regeneration
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from matplotlib import rcParams
import matplotlib.patches as mpatches

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')

# Configure matplotlib for modern academic publications
rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 15,
    'text.usetex': False,
    'axes.linewidth': 0.8,
    'axes.edgecolor': 'black',
    'axes.facecolor': 'white',
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
})

# Academic color palette based on provided scheme
ACADEMIC_COLORS = [
    '#E74C3C',  # R:231, G:76,  B:60  - Red-orange
    '#EB6535',  # R:235, G:101, B:53  - Deep orange transition
    '#EF7F39',  # R:239, G:127, B:57  - Orange
    '#F59B46',  # R:245, G:155, B:90  - Warm orange transition
    '#F7AF58',  # R:247, G:175, B:88  - Light orange
    '#FFD166',  # R:255, G:209, B:102 - Yellow
    '#E8B85C',  # R:232, G:184, B:92  - Muted sunset yellow
    '#B5C9C8',  # R:181, G:201, B:200 - Dusky transition blue
    '#72BFCF',  # R:114, G:191, B:207 - Medium blue
    '#4F8DA6',  # R:079, G:141, B:166 - Blue
    '#37679F',  # R:055, G:103, B:159 - Dark blue
    '#2A4C7B',  # R:042, G:076, B:123 - Twilight navy transition
    '#1E3A5F',  # R:030, G:058, B:095 - Very dark blue
    '#152B4A',  # R:021, G:043, B:074 - Deep blue-black transition
    '#0D1A2E',  # R:013, G:026, B:046 - Nightfall blue-black
    '#090E1B'   # R:007, G:014, B:027 - Near black, midnight tone
]



# Specialized colors for highlighting
COLORS = {
    'highlight': '#E74C3C',    # Red-orange for S²G-Net (most prominent)
    'primary': '#4F8DA6',      # Blue for primary elements
    'secondary': '#37679F',    # Dark blue for secondary
    'neutral': '#72BFCF',      # Medium blue for neutral
    'background': '#F8F9FA',   # Light background
}

# Model color assignment (ensuring S²G-Net gets the highlight color)
MODEL_COLORS = ACADEMIC_COLORS.copy()


def load_results(results_dir: Path = Path('results/sum_results')):
    """
    Load existing aggregated results directly from CSV files
    No data regeneration - uses existing processed data
    """
    summary_path = results_dir / 'summary.csv'
    scatter_path = results_dir / 'scatter_data.csv'
    
    # Check if required files exist
    if not summary_path.exists():
        print(f"❌ Summary file not found: {summary_path}")
        print(f"Expected files in {results_dir}:")
        print("  - summary.csv (formatted results)")
        print("  - scatter_data.csv (VRAM vs R² data)")
        return None, None
    
    try:
        # Load summary data
        print(f"📊 Loading summary data from: {summary_path}")
        summary_df = pd.read_csv(summary_path)
        print(f"   Loaded {len(summary_df)} models")
        
        # Load scatter data if available
        scatter_df = None
        if scatter_path.exists():
            print(f"📈 Loading scatter data from: {scatter_path}")
            scatter_df = pd.read_csv(scatter_path)
            print(f"   Loaded scatter data for {len(scatter_df)} models")
        else:
            print(f"⚠️  Scatter data not found: {scatter_path}")
        
        # Model name replacement mapping for display purposes
        model_name_mapping = {
            'mamba-gps': 'S²G-Net',
            'mamba_gps': 'S²G-Net',
            'dynamic_lstmgnn_gat': 'DyLSTM-GAT',
            'dynamic_lstmgnn_mpnn': 'DyLSTM-MPNN', 
            'dynamic_lstmgnn_gcn': 'DyLSTM-GCN',
            'bilstm': 'BiLSTM',
            'gnn_gat': 'GAT',
            'gnn_sage': 'GraphSAGE',
            'gnn_mpnn': 'MPNN',
            'lstmgnn_gat': 'LSTM-GAT',
            'lstmgnn_gcn': 'LSTM-GCN', 
            'lstmgnn_sage': 'LSTM-SAGE',
            'lstmgnn_mpnn': 'LSTM-MPNN',
            'mamba': 'Mamba',
            'graphgps': 'GraphGPS',
            'transformer': 'Transformer',
            'tcn': 'TCN',
            'gru': 'GRU',
            'lstm': 'LSTM',
            'rnn': 'RNN'
        }
        
        # Apply model name mapping only if needed
        if summary_df is not None:
            # Check if we have 'Model' column, if not check for 'model'
            model_col = 'Model' if 'Model' in summary_df.columns else 'model'
            if model_col in summary_df.columns:
                summary_df[model_col] = summary_df[model_col].replace(model_name_mapping)
                # Ensure we use 'Model' as the standard column name
                if model_col == 'model':
                    summary_df = summary_df.rename(columns={'model': 'Model'})
        
        if scatter_df is not None:
            # Check scatter dataframe model column
            model_col = 'model' if 'model' in scatter_df.columns else 'Model'
            if model_col in scatter_df.columns:
                scatter_df[model_col] = scatter_df[model_col].replace(model_name_mapping)
                # Ensure we use 'model' as the standard column name for scatter data
                if model_col == 'Model':
                    scatter_df = scatter_df.rename(columns={'Model': 'model'})
        
        # Standardize column names if needed
        if summary_df is not None:
            # Common column name standardizations
            column_mappings = {
                'peak_vram_mean': 'VRAM (GB)',
                'vram_gb': 'VRAM (GB)',
                'gpu_hours': 'GPU-h',
                'params_formatted': 'Params'
            }
            summary_df = summary_df.rename(columns=column_mappings)
        
        if scatter_df is not None:
            # Standardize scatter data column names
            scatter_column_mappings = {
                'peak_vram_GB': 'peak_vram_GB',  # Keep as is
                'vram_gb': 'peak_vram_GB'
            }
            scatter_df = scatter_df.rename(columns=scatter_column_mappings)
        
        print("✅ Data loaded successfully")
        print(f"📋 Summary columns: {list(summary_df.columns) if summary_df is not None else 'None'}")
        if scatter_df is not None:
            print(f"📋 Scatter columns: {list(scatter_df.columns)}")
        
        return summary_df, scatter_df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Please check that the CSV files are properly formatted")
        return None, None


def check_data_integrity(summary_df, scatter_df):
    """Check data integrity and provide helpful information"""
    if summary_df is None:
        return False
    
    print("\n🔍 Data Integrity Check:")
    print(f"📊 Summary data shape: {summary_df.shape}")
    print(f"📊 Required columns present:")
    
    required_summary_cols = ['Model', 'R2', 'MSE', 'MAD', 'MAPE', 'KAPPA']
    for col in required_summary_cols:
        present = col in summary_df.columns
        print(f"   {col}: {'✅' if present else '❌'}")
        if not present and col == 'MAD' and 'MAE' in summary_df.columns:
            print(f"   Found MAE instead of MAD - will use MAE")
    
    if scatter_df is not None:
        print(f"📈 Scatter data shape: {scatter_df.shape}")
        required_scatter_cols = ['model', 'peak_vram_GB', 'r2_mean', 'r2_std']
        for col in required_scatter_cols:
            present = col in scatter_df.columns
            alt_name = col.replace('peak_vram_GB', 'peak_vram_mean').replace('_GB', '_mean')
            if not present and alt_name in scatter_df.columns:
                print(f"   {col}: ✅ (found as {alt_name})")
                # Rename the column
                scatter_df.rename(columns={alt_name: col}, inplace=True)
            else:
                print(f"   {col}: {'✅' if present else '❌'}")
    
    # Check for S²G-Net presence
    s2g_present = False
    if 'Model' in summary_df.columns:
        models = summary_df['Model'].values
        s2g_variants = ['S²G-Net', 'mamba-gps', 'mamba_gps']
        s2g_present = any(variant in models for variant in s2g_variants)
    
    print(f"🎯 S²G-Net present: {'✅' if s2g_present else '❌'}")
    
    return True


def get_model_sort_key(model_name):
    custom_order = [
        'S²G-Net',
        'GraphGPS',
        'GAT',
        'MPNN',
        'GraphSAGE',
        'Mamba',
        'Transformer',
        'BiLSTM',
        'RNN',
        'LSTM-MPNN',
        'LSTM-GAT',
        'LSTM-SAGE',
        'DyLSTM-GAT',
        'DyLSTM-MPNN',
        'DyLSTM-GCN'
    ]
    try:
        return (custom_order.index(model_name), model_name)
    except ValueError:
        # Push unknown models to the end
        return (len(custom_order), model_name)


def plot_summary_table(summary_df: pd.DataFrame, save_path: Path = None):
    """Display summary table in a clean format"""
    print("📊 BASELINE COMPARISON SUMMARY")
    print("=" * 80)
    
    # Sort by R² (descending)
    display_df = summary_df.copy()
    
    # Extract numeric values for sorting (remove ± and †)
    r2_values = []
    for val in display_df['R2']:
        if isinstance(val, str):
            numeric_part = val.split('±')[0].replace('†', '')
            r2_values.append(float(numeric_part))
        else:
            r2_values.append(float(val))
    
    display_df['_r2_sort'] = r2_values
    display_df = display_df.sort_values('_r2_sort', ascending=False)
    display_df = display_df.drop('_r2_sort', axis=1)
    
    print(display_df.to_string(index=False))
    print("\n† = Statistically significant (p < 0.05)")
    
    if save_path:
        display_df.to_csv(save_path, index=False)
        print(f"💾 Table saved: {save_path}")


def plot_memory_accuracy_tradeoff(scatter_df: pd.DataFrame, save_path: Path = None):
    """Create conference-quality VRAM vs R² scatter plot with legend"""
    if scatter_df is None or scatter_df.empty:
        print("⚠️  No scatter data available")
        return
    
    # Check for required columns and handle alternatives
    vram_col = 'peak_vram_GB'
    if vram_col not in scatter_df.columns:
        alt_cols = ['peak_vram_mean', 'vram_gb', 'VRAM (GB)']
        for alt_col in alt_cols:
            if alt_col in scatter_df.columns:
                scatter_df = scatter_df.rename(columns={alt_col: vram_col})
                break
        else:
            print(f"❌ No VRAM data column found. Available: {list(scatter_df.columns)}")
            return
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    # Sort models by grouping same first letters together
    unique_models = sorted(scatter_df['model'].unique(), key=get_model_sort_key)
    
    # Prepare colors for different models
    model_colors = {}
    color_idx = 0
    for model in unique_models:
        if model == 'S²G-Net':
            model_colors[model] = COLORS['highlight']
        else:
            model_colors[model] = ACADEMIC_COLORS[color_idx % len(ACADEMIC_COLORS)]
            color_idx += 1
    
    # Plot all models except S²G-Net first (background layer)
    for model in unique_models:
        if model == 'S²G-Net':
            continue
        model_data = scatter_df[scatter_df['model'] == model]
        ax.errorbar(model_data[vram_col], model_data['r2_mean'], 
                   yerr=model_data['r2_std'], fmt='o', capsize=3, 
                   markersize=8, alpha=0.8, color=model_colors[model],
                   linewidth=1.2, elinewidth=1.2, zorder=1, label=model)
    
    # Plot S²G-Net on top (foreground layer)
    s2g_data = scatter_df[scatter_df['model'] == 'S²G-Net']
    if not s2g_data.empty:
        ax.errorbar(s2g_data[vram_col], s2g_data['r2_mean'], 
                   yerr=s2g_data['r2_std'], fmt='s', capsize=4, 
                   markersize=12, alpha=0.95, color=model_colors['S²G-Net'],
                   markeredgewidth=1.5, markeredgecolor='white',
                   linewidth=2.5, elinewidth=2.5, zorder=3, label='S²G-Net')
    
    ax.set_xlabel('Peak VRAM Usage (GB)')
    ax.set_ylabel('R² Score')
    
    # Refined grid
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.4, color='gray')
    ax.set_axisbelow(True)
    
    # Keep all spines for full frame
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['top'].set_linewidth(0.8)
    ax.spines['right'].set_linewidth(0.8)
    
    # Add legend with sorted model order
    handles, labels = ax.get_legend_handles_labels()
    if 'S²G-Net' in labels:
        s2g_idx = labels.index('S²G-Net')
        # Move S²G-Net to front
        handles = [handles[s2g_idx]] + [h for i, h in enumerate(handles) if i != s2g_idx]
        labels = [labels[s2g_idx]] + [l for i, l in enumerate(labels) if i != s2g_idx]
    
    legend = ax.legend(handles, labels, loc='best', frameon=True, framealpha=0.95,
                      edgecolor='lightgray', fancybox=False, shadow=False,
                      fontsize=10, ncol=1 if len(labels) <= 6 else 2)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(0.5)
    
    # Tight axis limits
    x_range = scatter_df[vram_col].max() - scatter_df[vram_col].min()
    y_range = scatter_df['r2_mean'].max() - scatter_df['r2_mean'].min()
    ax.set_xlim(scatter_df[vram_col].min() - x_range * 0.05, 
                scatter_df[vram_col].max() + x_range * 0.05)
    ax.set_ylim(scatter_df['r2_mean'].min() - y_range * 0.05, 
                scatter_df['r2_mean'].max() + y_range * 0.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='pdf', 
                   facecolor='white', edgecolor='none', dpi=300)
        print(f"💾 VRAM plot saved: {save_path}")
    
    plt.show()


def plot_performance_landscape(summary_df: pd.DataFrame, save_path: Path = None):
    """Create conference-quality multi-dimensional performance comparison"""
    # Extract numeric values
    def extract_numeric(col_name):
        values = []
        for val in summary_df[col_name]:
            if isinstance(val, str) and '±' in val:
                numeric_part = val.split('±')[0].replace('†', '')
                values.append(float(numeric_part))
            elif isinstance(val, (int, float)):
                values.append(float(val))
            else:
                values.append(0.0)
        return values
    
    # Extract parameter counts
    def extract_params(param_str):
        if isinstance(param_str, str):
            if 'M' in param_str:
                return float(param_str.replace('M', ''))
            elif 'K' in param_str:
                return float(param_str.replace('K', '')) / 1200
            else:
                try:
                    return float(param_str) / 1200000  # Assume raw number is in units
                except:
                    return 0.0
        elif isinstance(param_str, (int, float)):
            return float(param_str) / 1200000  # Convert to millions
        else:
            return 0.0
    
    r2_vals = extract_numeric('R2')
    mse_vals = extract_numeric('MSE')
    
    # Extract runtime data
    gpu_hours = []
    params_m = []
    
    for _, row in summary_df.iterrows():
        try:
            gpu_hours.append(float(row['GPU-h']))
        except:
            gpu_hours.append(0.0)
        
        try:
            params_m.append(extract_params(row['Params']))
        except:
            params_m.append(0.0)
    
    # Create consistent color mapping for all subplots with sorted model order
    unique_models = sorted(summary_df['Model'].unique(), key=get_model_sort_key)
    model_colors = {}
    color_idx = 0
    
    for model in unique_models:
        if model == 'S²G-Net':
            model_colors[model] = COLORS['highlight']  # Red-orange for S²G-Net
        else:
            model_colors[model] = ACADEMIC_COLORS[color_idx % len(ACADEMIC_COLORS)]
            color_idx += 1
    
    # Create figure with adjusted spacing for conference papers
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Plot 1: Training Cost vs Performance
    for i, model in enumerate(summary_df['Model']):
        color = model_colors[model]
        if model == 'S²G-Net':
            marker, size, alpha, edgecolor, linewidth = 's', 120, 0.95, 'white', 1.5
            zorder = 3
        else:
            marker, size, alpha, edgecolor, linewidth = 'o', 90, 0.75, 'none', 0
            zorder = 1
        
        axes[0, 0].scatter(gpu_hours[i], r2_vals[i], color=color, s=size, 
                          alpha=alpha, edgecolors=edgecolor, linewidth=linewidth,
                          marker=marker, zorder=zorder)
    
    # Add labels for top performers and S²G-Net
    for i, model in enumerate(summary_df['Model']):
        if r2_vals[i] > np.percentile(r2_vals, 75) or model == 'S²G-Net':
            axes[0, 0].annotate(model, (gpu_hours[i], r2_vals[i]), 
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=10, ha='left', va='bottom',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                       alpha=0.8, edgecolor='lightgray', linewidth=0.5))
    
    axes[0, 0].set_xlabel('Training Time (GPU-hours)')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('(a) Training Efficiency')
    axes[0, 0].grid(True, alpha=0.25, linewidth=0.4)
    
    # Plot 2: Model Size vs Performance
    for i, model in enumerate(summary_df['Model']):
        color = model_colors[model]
        if model == 'S²G-Net':
            marker, size, alpha, edgecolor, linewidth = 's', 120, 0.95, 'white', 1.5
            zorder = 3
        else:
            marker, size, alpha, edgecolor, linewidth = 'o', 90, 0.75, 'none', 0
            zorder = 1
        
        axes[0, 1].scatter(params_m[i], r2_vals[i], color=color, s=size, 
                          alpha=alpha, edgecolors=edgecolor, linewidth=linewidth,
                          marker=marker, zorder=zorder)
    
    # Add labels for top performers and S²G-Net
    for i, model in enumerate(summary_df['Model']):
        if r2_vals[i] > np.percentile(r2_vals, 75) or model == 'S²G-Net':
            axes[0, 1].annotate(model, (params_m[i], r2_vals[i]), 
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=10, ha='left', va='bottom',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                       alpha=0.8, edgecolor='lightgray', linewidth=0.5))
    
    axes[0, 1].set_xlabel('Model Parameters (M)')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].set_title('(b) Model Complexity')
    axes[0, 1].grid(True, alpha=0.25, linewidth=0.4)
    
    # Plot 3: R² vs MSE (Performance consistency)
    for i, model in enumerate(summary_df['Model']):
        color = model_colors[model]
        if model == 'S²G-Net':
            marker, size, alpha, edgecolor, linewidth = 's', 120, 0.95, 'white', 1.5
            zorder = 3
        else:
            marker, size, alpha, edgecolor, linewidth = 'o', 90, 0.75, 'none', 0
            zorder = 1
            
        axes[1, 0].scatter(r2_vals[i], mse_vals[i], color=color, s=size, 
                          alpha=alpha, edgecolors=edgecolor, linewidth=linewidth,
                          marker=marker, zorder=zorder)
    
    # Add labels for top performers and S²G-Net  
    for i, model in enumerate(summary_df['Model']):
        if r2_vals[i] > np.percentile(r2_vals, 75) or model == 'S²G-Net':
            offset = (5, 5)
            if model == 'Mamba':
                offset = (7, -5)
    
            axes[1, 0].annotate(model, (r2_vals[i], mse_vals[i]),
                                xytext=offset, textcoords='offset points',
                                fontsize=10, ha='left', va='bottom',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                          alpha=0.8, edgecolor='lightgray', linewidth=0.5))

    
    axes[1, 0].set_xlabel('R² Score')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('(c) Accuracy vs. Error')
    axes[1, 0].grid(True, alpha=0.25, linewidth=0.4)
    
    # Plot 4: Horizontal bar chart - top 8 models only, ordered from top (highest) to bottom (lowest)
    sorted_indices = sorted(
        range(len(r2_vals)),
        key=lambda i: (-r2_vals[i], summary_df['Model'].iloc[i].lower())
    )[:8]  # Top 8 only, highest to lowest
    sorted_models = [summary_df['Model'].iloc[i] for i in sorted_indices]
    sorted_r2 = [r2_vals[i] for i in sorted_indices]
    sorted_colors = [model_colors[model] for model in sorted_models]
    
    # Reverse the order for plotting (highest at top, lowest at bottom)
    sorted_models = sorted_models[::-1]
    sorted_r2 = sorted_r2[::-1]
    sorted_colors = sorted_colors[::-1]
    
    y_pos = np.arange(len(sorted_models))
    bars = axes[1, 1].barh(y_pos, sorted_r2, color=sorted_colors, alpha=0.8, 
                          edgecolor='white', linewidth=0.5, height=0.7)
    
    # Highlight S²G-Net bar
    for i, model in enumerate(sorted_models):
        if model == 'S²G-Net':
            bars[i].set_alpha(0.95)
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(1.5)
    
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels(sorted_models, fontsize=10)
    axes[1, 1].set_xlabel('R² Score')
    axes[1, 1].set_title('(d) Performance Ranking')
    axes[1, 1].grid(True, alpha=0.25, axis='x', linewidth=0.4)
    
    # Fix text positioning to stay within plot boundaries
    max_r2 = max(sorted_r2)
    axes[1, 1].set_xlim(0, max_r2 * 1.15)  # Add 15% padding for text
    
    # Add value labels inside bars to prevent overflow
    for i, (model, val) in enumerate(zip(sorted_models, sorted_r2)):
        # Position text inside the bar, slightly offset from the right edge
        text_x = val - max_r2 * 0.02  # 2% offset from bar end
        axes[1, 1].text(text_x, i, f'{val:.2f}', 
                       va='center', ha='right', fontsize=10, 
                       color='white', fontweight='bold')
    
    # Keep all spines for full frames on all subplots
    for ax in axes.flat:
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['top'].set_linewidth(0.8)
        ax.spines['right'].set_linewidth(0.8)

    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='pdf', 
                   facecolor='white', edgecolor='none', dpi=300)
        print(f"💾 Performance landscape saved: {save_path}")
    
    plt.show()


def plot_metric_comparison(summary_df: pd.DataFrame, save_path: Path = None):
    """Create conference-quality radar chart for metric comparison"""
    metrics = ['R2', 'MSE', 'MAD', 'MAPE', 'KAPPA']
    
    # Create a copy to avoid modifying original
    summary_df = summary_df.copy()
    
    # Handle MAE to MAD column renaming if needed
    if 'MAE' in summary_df.columns and 'MAD' not in summary_df.columns:
        summary_df['MAD'] = summary_df['MAE']
    
    # Extract numerical R2 values for sorting
    r2_numeric = []
    for val in summary_df['R2']:
        if isinstance(val, str) and '±' in val:
            numeric_part = val.split('±')[0].replace('†', '')
            r2_numeric.append(float(numeric_part))
        elif isinstance(val, (int, float)):
            r2_numeric.append(float(val))
        else:
            r2_numeric.append(0.0)
    
    summary_df['_R2_num'] = r2_numeric
    
    # Select top 5 models
    top_models = summary_df.nlargest(5, '_R2_num')
    
    # Prepare data for radar plot
    metric_data = {m: [] for m in metrics}
    for metric in metrics:
        for val in summary_df[metric]:
            if isinstance(val, str) and '±' in val:
                val_clean = float(val.split('±')[0].replace('†', ''))
                metric_data[metric].append(val_clean)
            elif isinstance(val, (int, float)):
                metric_data[metric].append(float(val))
            else:
                metric_data[metric].append(0.0)
    
    # Normalize metrics (0-1 scale)
    normalized_data = {}
    for metric in metrics:
        values = np.array(metric_data[metric])
        min_val, max_val = np.min(values), np.max(values)
        if max_val > min_val:
            # For error metrics (MSE, MAD, MAPE), invert the scale
            if metric in ['MSE', 'MAD', 'MAPE']:
                normalized = 1 - (values - min_val) / (max_val - min_val)
            else:
                normalized = (values - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(values)
        normalized_data[metric] = normalized
    
    # Setup radar chart
    labels = ['R²', 'MSE⁻¹', 'MAD⁻¹', 'MAPE⁻¹', 'Kappa']
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Prepare colors for models with sorted order
    model_colors = {}
    color_idx = 0
    for _, row in top_models.iterrows():
        model_name = row['Model']
        if model_name == 'S²G-Net':
            model_colors[model_name] = COLORS['highlight']
        else:
            model_colors[model_name] = ACADEMIC_COLORS[color_idx % len(ACADEMIC_COLORS)]
            color_idx += 1
    
    # Plot each top model
    for idx, (_, row) in enumerate(top_models.iterrows()):
        model_idx = summary_df.index.get_loc(row.name)
        model_name = row['Model']
        
        values = [normalized_data[m][model_idx] for m in metrics]
        values += values[:1]  # Complete the circle
        
        color = model_colors[model_name]
        
        # Style based on whether it's S²G-Net
        if model_name == 'S²G-Net':
            linewidth = 3.5
            alpha_line = 1.0
            alpha_fill = 0.2
            markersize = 8
            marker = 's'
        else:
            linewidth = 2.2
            alpha_line = 0.8
            alpha_fill = 0.1
            markersize = 6
            marker = 'o'
        
        ax.plot(angles, values, marker=marker, linewidth=linewidth, 
               label=model_name, color=color, alpha=alpha_line, markersize=markersize)
        ax.fill(angles, values, alpha=alpha_fill, color=color)
    
    # Customize the radar chart
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10, alpha=0.6)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    
    # Remove radial axis labels at 0 position
    ax.set_rlabel_position(45)
    
    # Clean legend - no title
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.05), 
                      frameon=True, framealpha=0.95, edgecolor='lightgray',
                      fancybox=False, shadow=False)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='pdf', 
                   facecolor='white', edgecolor='none', dpi=300)
        print(f"💾 Metric comparison saved: {save_path}")
    
    plt.show()
    
    # Clean up
    summary_df.drop(columns=['_R2_num'], inplace=True)


def generate_all_plots(results_dir: Path = Path('results/sum_results'), 
                      output_dir: Path = Path('results/sum_results/plots')):
    """Generate all conference-quality visualization plots from existing data"""
    print("📈 Generating conference-quality visualization plots from existing data...")
    print(f"📂 Reading data from: {results_dir}")
    
    # Load existing data (no regeneration)
    summary_df, scatter_df = load_results(results_dir)
    
    if summary_df is None:
        print("❌ Failed to load data. Please check that the required CSV files exist.")
        return
    
    # Check data integrity
    if not check_data_integrity(summary_df, scatter_df):
        print("❌ Data integrity check failed")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📊 Found {len(summary_df)} models to visualize")
    print(f"💾 Output directory: {output_dir}")
    
    # Generate plots (only PDF format, conference-ready)
    print("\n🎨 Generating visualizations...")
    
    try:
        plot_summary_table(summary_df, output_dir / 'summary_formatted.csv')
        
        if scatter_df is not None and not scatter_df.empty:
            plot_memory_accuracy_tradeoff(scatter_df, output_dir / 'fig1_memory_efficiency.pdf')
        else:
            print("⚠️  Skipping memory efficiency plot - no scatter data available")
        
        plot_performance_landscape(summary_df, output_dir / 'fig2_performance_analysis.pdf')
        
        plot_metric_comparison(summary_df, output_dir / 'fig3_metric_comparison.pdf')
        
        print(f"\n🎉 All conference-quality figures generated in: {output_dir}")
        print("📄 Generated files:")
        print("  • summary_formatted.csv - Formatted results table")
        if scatter_df is not None:
            print("  • fig1_memory_efficiency.pdf - Memory vs. accuracy trade-off")
        print("  • fig2_performance_analysis.pdf - Multi-dimensional analysis")
        print("  • fig3_metric_comparison.pdf - Top-5 models radar chart")
        print("\n✅ All figures are conference-ready with academic color scheme")
        print("✅ S²G-Net is highlighted across all visualizations")
        print("✅ No overlapping labels, clean layouts, professional styling")
        
    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate publication-quality visualization plots from existing data')
    parser.add_argument('--results_dir', type=Path, default='results/sum_results',
                       help='Directory with existing CSV results')
    parser.add_argument('--output_dir', type=Path, default='results/sum_results/plots',
                       help='Directory to save plots')
    parser.add_argument('--plot_type', choices=['summary', 'scatter', 'landscape', 'metrics', 'all'],
                       default='all', help='Type of plot to generate')
    
    args = parser.parse_args()
    
    if args.plot_type == 'all':
        generate_all_plots(args.results_dir, args.output_dir)
    else:
        summary_df, scatter_df = load_results(args.results_dir)
        if summary_df is None:
            exit(1)
        
        # Check data integrity
        check_data_integrity(summary_df, scatter_df)
        
        if args.plot_type == 'summary':
            plot_summary_table(summary_df)
        elif args.plot_type == 'scatter':
            if scatter_df is not None:
                plot_memory_accuracy_tradeoff(scatter_df)
            else:
                print("❌ Scatter plot requires scatter_data.csv")
        elif args.plot_type == 'landscape':
            plot_performance_landscape(summary_df)
        elif args.plot_type == 'metrics':
            plot_metric_comparison(summary_df)