# src/evaluation/tracking/tracking_viz.py
"""
Publication-quality visualization script for baseline comparison results
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
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
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
    '#E74C3C',  # R:231, G:98, B:84 - Red-orange
    '#EF7F39',  # R:239, G:138, B:71 - Orange  
    '#F7AF58',  # R:247, G:170, B:88 - Light orange
    '#FFD166',  # R:255, G:208, B:111 - Yellow
    '#FFF2B7',  # R:255, G:230, B:183 - Light yellow
    '#AADCE0',  # R:170, G:220, B:224 - Light blue
    '#72BFCF',  # R:114, G:188, B:213 - Medium blue
    '#4F8DA6',  # R:082, G:143, B:173 - Blue
    '#37679F',  # R:055, G:103, B:149 - Dark blue
    '#1E3A5F'   # R:030, G:070, B:110 - Very dark blue
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
    """Load aggregated results"""
    summary_path = results_dir / 'summary.csv'
    scatter_path = results_dir / 'scatter_data.csv'
    
    if not summary_path.exists():
        print(f"❌ Summary file not found: {summary_path}")
        print("Run: python -m src.evaluation.tracking.aggregate first")
        return None, None
    
    summary_df = pd.read_csv(summary_path)
    scatter_df = pd.read_csv(scatter_path) if scatter_path.exists() else None
    
    # Model name replacement mapping
    model_name_mapping = {
        'mamba-gps': 'S²G-Net',
        'mamba_gps': 'S²G-Net',
        'dynamic_lstmgnn_gat': 'DyLSTM-GAT',
        'dynamic_lstmgnn_mpnn': 'DyLSTM-MPNN', 
        'dynamic_lstmgnn_gcn': 'DyLSTM-GCN',
        'bilstm': 'BiLSTM',
        'gnn_gat': 'GAT',
        'gnn_gcn': 'GCN',
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
    
    # Replace model names in summary dataframe
    if summary_df is not None:
        if 'Model' in summary_df.columns:
            summary_df['Model'] = summary_df['Model'].replace(model_name_mapping)
    
    # Replace model names in scatter dataframe
    if scatter_df is not None:
        if 'model' in scatter_df.columns:
            scatter_df['model'] = scatter_df['model'].replace(model_name_mapping)
    
    return summary_df, scatter_df


def get_model_sort_key(model_name):
    """Get sort key for model names, grouping by first letter"""
    # Extract the first letter/prefix for grouping
    if model_name.startswith('DyLSTM'):
        return ('DyLSTM', model_name)
    elif model_name.startswith('LSTM'):
        return ('LSTM', model_name)
    elif model_name.startswith('GCN') or model_name.startswith('GAT') or model_name.startswith('GRU'):
        return (model_name[0], model_name)
    else:
        return (model_name[0], model_name)


def plot_summary_table(summary_df: pd.DataFrame, save_path: Path = None):
    """Display summary table in a clean format"""
    print("📊 BASELINE COMPARISON SUMMARY")
    print("=" * 80)
    
    # Sort by R² (descending)
    display_df = summary_df.copy()
    
    # Extract numeric values for sorting (remove ± and †)
    r2_values = []
    for val in display_df['R2']:
        numeric_part = val.split('±')[0].replace('†', '')
        r2_values.append(float(numeric_part))
    
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
        ax.errorbar(model_data['peak_vram_GB'], model_data['r2_mean'], 
                   yerr=model_data['r2_std'], fmt='o', capsize=3, 
                   markersize=8, alpha=0.8, color=model_colors[model],
                   linewidth=1.2, elinewidth=1.2, zorder=1, label=model)
    
    # Plot S²G-Net on top (foreground layer)
    s2g_data = scatter_df[scatter_df['model'] == 'S²G-Net']
    if not s2g_data.empty:
        ax.errorbar(s2g_data['peak_vram_GB'], s2g_data['r2_mean'], 
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
                      fontsize=9, ncol=1 if len(labels) <= 6 else 2)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(0.5)
    
    # Tight axis limits
    x_range = scatter_df['peak_vram_GB'].max() - scatter_df['peak_vram_GB'].min()
    y_range = scatter_df['r2_mean'].max() - scatter_df['r2_mean'].min()
    ax.set_xlim(scatter_df['peak_vram_GB'].min() - x_range * 0.05, 
                scatter_df['peak_vram_GB'].max() + x_range * 0.05)
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
            else:
                values.append(0.0)
        return values
    
    # Extract parameter counts
    def extract_params(param_str):
        if 'M' in param_str:
            return float(param_str.replace('M', ''))
        elif 'K' in param_str:
            return float(param_str.replace('K', '')) / 1000
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
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Plot 1: Training Cost vs Performance
    for i, model in enumerate(summary_df['Model']):
        color = model_colors[model]
        if model == 'S²G-Net':
            marker, size, alpha, edgecolor, linewidth = 's', 100, 0.95, 'white', 1.5
            zorder = 3
        else:
            marker, size, alpha, edgecolor, linewidth = 'o', 70, 0.75, 'none', 0
            zorder = 1
        
        axes[0, 0].scatter(gpu_hours[i], r2_vals[i], color=color, s=size, 
                          alpha=alpha, edgecolors=edgecolor, linewidth=linewidth,
                          marker=marker, zorder=zorder)
    
    # Add labels for top performers and S²G-Net
    for i, model in enumerate(summary_df['Model']):
        if r2_vals[i] > np.percentile(r2_vals, 75) or model == 'S²G-Net':
            axes[0, 0].annotate(model, (gpu_hours[i], r2_vals[i]), 
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=8, ha='left', va='bottom',
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
            marker, size, alpha, edgecolor, linewidth = 's', 100, 0.95, 'white', 1.5
            zorder = 3
        else:
            marker, size, alpha, edgecolor, linewidth = 'o', 70, 0.75, 'none', 0
            zorder = 1
        
        axes[0, 1].scatter(params_m[i], r2_vals[i], color=color, s=size, 
                          alpha=alpha, edgecolors=edgecolor, linewidth=linewidth,
                          marker=marker, zorder=zorder)
    
    # Add labels for top performers and S²G-Net
    for i, model in enumerate(summary_df['Model']):
        if r2_vals[i] > np.percentile(r2_vals, 75) or model == 'S²G-Net':
            axes[0, 1].annotate(model, (params_m[i], r2_vals[i]), 
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=8, ha='left', va='bottom',
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
            marker, size, alpha, edgecolor, linewidth = 's', 100, 0.95, 'white', 1.5
            zorder = 3
        else:
            marker, size, alpha, edgecolor, linewidth = 'o', 70, 0.75, 'none', 0
            zorder = 1
        
        axes[1, 0].scatter(r2_vals[i], mse_vals[i], color=color, s=size, 
                          alpha=alpha, edgecolors=edgecolor, linewidth=linewidth,
                          marker=marker, zorder=zorder)
    
    # Add labels for top performers and S²G-Net  
    for i, model in enumerate(summary_df['Model']):
        if r2_vals[i] > np.percentile(r2_vals, 75) or model == 'S²G-Net':
            axes[1, 0].annotate(model, (r2_vals[i], mse_vals[i]), 
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=8, ha='left', va='bottom',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                       alpha=0.8, edgecolor='lightgray', linewidth=0.5))
    
    axes[1, 0].set_xlabel('R² Score')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('(c) Accuracy vs. Error')
    axes[1, 0].grid(True, alpha=0.25, linewidth=0.4)
    
    # Plot 4: Horizontal bar chart - top 8 models only, ordered from top (highest) to bottom (lowest)
    sorted_indices = np.argsort(r2_vals)[::-1][:8]  # Top 8 only, highest to lowest
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
    axes[1, 1].set_yticklabels(sorted_models, fontsize=9)
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
        axes[1, 1].text(text_x, i, f'{val:.3f}', 
                       va='center', ha='right', fontsize=8, 
                       color='white', fontweight='bold')
    
    # Keep all spines for full frames on all subplots
    for ax in axes.flat:
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['top'].set_linewidth(0.8)
        ax.spines['right'].set_linewidth(0.8)
    
    plt.tight_layout()
    
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
    summary_df['_R2_num'] = (
        summary_df['R2']
        .str.split('±').str[0]
        .str.replace('†', '', regex=False)
        .astype(float)
    )
    
    # Select top 5 models
    top_models = summary_df.nlargest(5, '_R2_num')
    
    # Prepare data for radar plot
    metric_data = {m: [] for m in metrics}
    for metric in metrics:
        for val in summary_df[metric]:
            if isinstance(val, str) and '±' in val:
                val_clean = float(val.split('±')[0].replace('†', ''))
                metric_data[metric].append(val_clean)
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
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9, alpha=0.6)
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
    """Generate all conference-quality visualization plots"""
    print("📈 Generating conference-quality visualization plots...")
    
    # Load data
    summary_df, scatter_df = load_results(results_dir)
    
    if summary_df is None:
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Found {len(summary_df)} models to visualize")
    
    # Generate plots (only PDF format, conference-ready)
    plot_summary_table(summary_df, output_dir / 'summary_formatted.csv')
    
    plot_memory_accuracy_tradeoff(scatter_df, output_dir / 'fig1_memory_efficiency.pdf')
    
    plot_performance_landscape(summary_df, output_dir / 'fig2_performance_analysis.pdf')
    
    plot_metric_comparison(summary_df, output_dir / 'fig3_metric_comparison.pdf')
    
    print(f"\n🎉 All conference-quality figures generated in: {output_dir}")
    print("📄 Generated files:")
    print("  • summary_formatted.csv - Formatted results table")
    print("  • fig1_memory_efficiency.pdf - Memory vs. accuracy trade-off")
    print("  • fig2_performance_analysis.pdf - Multi-dimensional analysis")
    print("  • fig3_metric_comparison.pdf - Top-5 models radar chart")
    print("\n✅ All figures are conference-ready with academic color scheme")
    print("✅ S²G-Net is highlighted across all visualizations")
    print("✅ No overlapping labels, clean layouts, professional styling")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate publication-quality visualization plots')
    parser.add_argument('--results_dir', type=Path, default='results/sum_results',
                       help='Directory with aggregated results')
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
        
        if args.plot_type == 'summary':
            plot_summary_table(summary_df)
        elif args.plot_type == 'scatter':
            plot_memory_accuracy_tradeoff(scatter_df)
        elif args.plot_type == 'landscape':
            plot_performance_landscape(summary_df)
        elif args.plot_type == 'metrics':
            plot_metric_comparison(summary_df)