#!/usr/bin/env python3
"""
Optimized publication-ready figure generation for S²G-Net ICU LOS prediction
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
from matplotlib.colors import LinearSegmentedColormap

# Academic styling
plt.style.use('seaborn-v0_8-whitegrid')

# Academic colors from tracking_viz
ACADEMIC_COLORS = [
    '#E74C3C',  # Red-orange
    '#EF7F39',  # Orange  
    '#F7AF58',  # Light orange
    '#FFD166',  # Yellow
    '#FFF2B7',  # Light yellow
    '#AADCE0',  # Light blue
    '#72BFCF',  # Medium blue
    '#4F8DA6',  # Blue
    '#37679F',  # Dark blue
    '#1E3A5F'   # Very dark blue
]

# Method colors using tracking_viz scheme
METHOD_COLORS = [ACADEMIC_COLORS[7], ACADEMIC_COLORS[1], ACADEMIC_COLORS[0]]  # Blue, Orange, Red-orange

# Minimal matplotlib config
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.linewidth': 0.8,
    'grid.alpha': 0.25,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf'
})

def load_results(results_dir):
    """Load experimental results"""
    results_file = Path(results_dir) / "all_results.csv"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    df = pd.read_csv(results_file)
    df = df[df['status'] == 'completed'].copy()
    
    if len(df) == 0:
        raise ValueError("No successful experiments found!")
    
    print(f"📊 Loaded {len(df)} successful experiments")
    return df

def create_blue_colormap():
    """Create blue-based colormap for heatmap (single color system)"""
    # Use blue tones from tracking_viz for consistency
    colors = [ACADEMIC_COLORS[5], ACADEMIC_COLORS[7], ACADEMIC_COLORS[9]]  # Dark blue to medium blue
    return LinearSegmentedColormap.from_list('blue_system', colors, N=100)

def figure1_and_3_combined(df, output_dir):
    """Combined hyperparameter heatmap and method comparison"""
    # Prepare data for heatmap
    best_row = df.loc[df['r2_mean'].idxmax()]
    subset = df[(df['method'] == best_row['method']) & 
                (df['rewiring'] == best_row['rewiring'])]
    
    if len(subset) < 6:
        pivot_data = df.groupby(['k_diag', 'k_bert'])['r2_mean'].mean().reset_index()
        heatmap_data = pivot_data.pivot(index='k_diag', columns='k_bert', values='r2_mean')
    else:
        heatmap_data = subset.pivot(index='k_diag', columns='k_bert', values='r2_mean')
    
    # Prepare data for bar chart
    method_stats = df.groupby('method')['r2_mean'].agg(['mean', 'std']).reset_index()
    
    # Create combined figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Heatmap (blue color system)
    sns.heatmap(heatmap_data, 
                annot=True, fmt='.3f', cmap=create_blue_colormap(),
                cbar_kws={'label': 'R² Score', 'shrink': 0.8},
                ax=ax1, square=True, linewidths=0.8, linecolor='white',
                annot_kws={'fontsize': 12, 'color': 'white', 'fontweight': 'bold'})
    
    ax1.set_xlabel('$k_{\\mathrm{BERT}}$ (Text-based Neighbors)')
    ax1.set_ylabel('$k_{\\mathrm{diag}}$ (Diagnosis-based Neighbors)')
    ax1.set_title('(a) k-Parameter Optimization', pad=20, fontsize=13)
    
    # Subplot 2: Bar chart (thinner bars, tracking_viz colors)
    x_pos = range(len(method_stats))
    bars = ax2.bar(x_pos, method_stats['mean'], 
                   yerr=method_stats['std'], capsize=5,
                   color=METHOD_COLORS, 
                   alpha=0.8, edgecolor='white', linewidth=1,
                   width=0.6)  # Thinner bars
    
    # Add value labels on bars
    for i, (bar, mean_val) in enumerate(zip(bars, method_stats['mean'])):
        ax2.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + method_stats['std'].iloc[i] + 0.002,
                f'{mean_val:.3f}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    ax2.set_xlabel('Similarity Method')
    ax2.set_ylabel('Average R² Score')
    ax2.set_title('(b) Method Comparison', pad=20, fontsize=13)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([method.capitalize() for method in method_stats['method']])
    ax2.grid(True, alpha=0.25, axis='y', linewidth=0.4)
    
    # Set y-axis limits for better visualization
    y_max = (method_stats['mean'] + method_stats['std']).max()
    ax2.set_ylim(0, y_max * 1.15)
    
    plt.tight_layout()
    save_path = output_dir / "figure1_combined_analysis.pdf"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Figure 1 (Combined) saved: {save_path}")
    plt.close()

def figure2_pareto_frontier(df, output_dir):
    """Pareto frontier analysis with fixed density display"""
    def compute_pareto_frontier(points):
        """Compute Pareto frontier indices"""
        pareto_indices = []
        points_array = np.array(points)
        
        for i in range(len(points)):
            is_pareto = True
            r2_i, density_i = points_array[i]
            
            for j in range(len(points)):
                if i == j:
                    continue
                r2_j, density_j = points_array[j]
                
                if (r2_j >= r2_i and density_j <= density_i and 
                    (r2_j > r2_i or density_j < density_i)):
                    is_pareto = False
                    break
            
            if is_pareto:
                pareto_indices.append(i)
        
        return pareto_indices
    
    points = list(zip(df['r2_mean'].values, df['density'].values))
    pareto_indices = compute_pareto_frontier(points)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot by method with larger points (tracking_viz colors)
    methods = df['method'].unique()
    for i, method in enumerate(methods):
        method_data = df[df['method'] == method]
        ax.scatter(method_data['density'] * 1e6, method_data['r2_mean'],
                  c=METHOD_COLORS[i % len(METHOD_COLORS)], alpha=0.7, s=80,
                  label=method.capitalize(), edgecolors='white', linewidth=0.8)
    
    # Pareto frontier
    if len(pareto_indices) > 1:
        pareto_df = df.iloc[pareto_indices].sort_values('density')
        ax.plot(pareto_df['density'] * 1e6, pareto_df['r2_mean'], 
               color=ACADEMIC_COLORS[0], linewidth=2.5, alpha=0.9, zorder=10)
    
    ax.scatter(df.iloc[pareto_indices]['density'] * 1e6, 
              df.iloc[pareto_indices]['r2_mean'],
              c=ACADEMIC_COLORS[0], s=120, marker='D',
              edgecolors='white', linewidth=1.5,
              label='Pareto Optimal', zorder=11)
    
    # Best point
    best_idx = df['r2_mean'].idxmax()
    best = df.loc[best_idx]
    ax.scatter(best['density'] * 1e6, best['r2_mean'], 
              c=ACADEMIC_COLORS[3], s=180, marker='*',
              edgecolors='black', linewidth=1.5,
              label='Best Performance', zorder=12)
    
    # Fixed density label with proper scientific notation
    ax.set_xlabel('Graph Density (×10$^{-6}$)')
    ax.set_ylabel('R² Score')
    ax.legend(loc='lower right', frameon=True, framealpha=0.95,
             edgecolor='lightgray', fancybox=False, shadow=False)
    ax.grid(True, alpha=0.25, linewidth=0.4)
    
    plt.tight_layout()
    save_path = output_dir / "figure2_pareto_frontier.pdf"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Figure 2 saved: {save_path}")
    plt.close()

def table1_top_configurations(df, output_dir, top_n=10):
    """Generate top configurations table (without std column)"""
    top_configs = df.nlargest(top_n, 'r2_mean').reset_index(drop=True)
    
    table_data = []
    for idx, (_, row) in enumerate(top_configs.iterrows(), 1):
        table_data.append({
            'Rank': idx,
            'k_diag': int(row['k_diag']),
            'k_BERT': int(row['k_bert']),
            'Method': row['method'].capitalize(),
            'Rewiring': row['rewiring'].replace('_', ' ').title(),
            'R²': f"{row['r2_mean']:.4f}",
            'Edges': f"{int(row['n_edges']):,}",
            'Density': f"{row['density']*1e6:.2f}"
        })
    
    results_df = pd.DataFrame(table_data)
    
    # Save CSV and LaTeX
    csv_path = output_dir / "table1_top_configurations.csv"
    results_df.to_csv(csv_path, index=False)
    
    latex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Top-10 Graph Construction Configurations for ICU Length-of-Stay Prediction. " +
        "Density values are scaled by $10^6$ for readability.}",
        "\\label{tab:graph_configs}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{@{}cccccccc@{}}",
        "\\toprule",
        "Rank & $k_{\\text{diag}}$ & $k_{\\text{BERT}}$ & Method & Rewiring & R² & \\#Edges & Density \\\\",
        "\\midrule"
    ]
    
    for _, row in results_df.iterrows():
        line = (f"{row['Rank']} & {row['k_diag']} & {row['k_BERT']} & "
               f"{row['Method']} & {row['Rewiring']} & {row['R²']} & "
               f"{row['Edges']} & {row['Density']} \\\\")
        latex_lines.append(line)
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}"
    ])
    
    latex_path = output_dir / "table1_top_configurations.tex"
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"✅ Table 1 saved: {csv_path} and {latex_path}")
    return results_df

def table_appendix_full_ranking(df, output_dir):
    """Generate full ranking table for appendix"""
    full_ranking = df.sort_values('r2_mean', ascending=False).reset_index(drop=True)
    
    table_data = []
    for idx, (_, row) in enumerate(full_ranking.iterrows(), 1):
        table_data.append({
            'Rank': idx,
            'k_diag': int(row['k_diag']),
            'k_BERT': int(row['k_bert']),
            'Method': row['method'].capitalize(),
            'Rewiring': row['rewiring'].replace('_', ' ').title(),
            'R²': f"{row['r2_mean']:.4f}",
            'Std': f"{row['r2_std']:.4f}" if pd.notna(row['r2_std']) else "N/A",
            'Edges': f"{int(row['n_edges']):,}",
            'Density': f"{row['density']*1e6:.2f}"
        })
    
    results_df = pd.DataFrame(table_data)
    
    # Save CSV
    csv_path = output_dir / "appendix_full_ranking.csv"
    results_df.to_csv(csv_path, index=False)
    
    # Generate LaTeX for appendix
    latex_lines = [
        "\\begin{longtable}{@{}ccccccccc@{}}",
        "\\caption{Complete Ranking of All Graph Construction Configurations} \\\\",
        "\\toprule",
        "Rank & $k_{\\text{diag}}$ & $k_{\\text{BERT}}$ & Method & Rewiring & R² & Std & \\#Edges & Density \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\multicolumn{9}{c}%",
        "{\\tablename\\ \\thetable\\ -- \\textit{Continued from previous page}} \\\\",
        "\\toprule",
        "Rank & $k_{\\text{diag}}$ & $k_{\\text{BERT}}$ & Method & Rewiring & R² & Std & \\#Edges & Density \\\\",
        "\\midrule",
        "\\endhead",
        "\\midrule \\multicolumn{9}{r}{\\textit{Continued on next page}} \\\\",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot"
    ]
    
    for _, row in results_df.iterrows():
        line = (f"{row['Rank']} & {row['k_diag']} & {row['k_BERT']} & "
               f"{row['Method']} & {row['Rewiring']} & {row['R²']} & "
               f"{row['Std']} & {row['Edges']} & {row['Density']} \\\\")
        latex_lines.append(line)
    
    latex_lines.append("\\end{longtable}")
    
    latex_path = output_dir / "appendix_full_ranking.tex"
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"✅ Appendix table saved: {csv_path} and {latex_path}")
    return results_df

def generate_summary_stats(df):
    """Generate key statistics in the requested format"""
    best = df.loc[df['r2_mean'].idxmax()]
    
    # Format top-10 configurations (without std in display)
    top_configs = df.nlargest(10, 'r2_mean')
    
    print(f"\n📋 Top-10 Configurations:")
    print("-" * 60)
    for i, (_, row) in enumerate(top_configs.iterrows(), 1):
        method_str = f"{row['method'].capitalize():8s}"
        rewiring_str = f"{row['rewiring'].replace('_', ' ').title():10s}"
        print(f"{i:2d}. k_diag={int(row['k_diag']):2d}, k_BERT={int(row['k_bert']):2d}, "
              f"{method_str}, {rewiring_str} → {row['r2_mean']:.3f}")
    
    print(f"\n📊 Key Statistics for Paper:")
    print("=" * 50)
    print(f"• Total configurations tested: {len(df)}")
    print(f"• Performance range: {df['r2_mean'].min():.3f} - {df['r2_mean'].max():.3f}")
    improvement = ((df['r2_mean'].max() - df['r2_mean'].min()) / df['r2_mean'].min() * 100)
    print(f"• Relative improvement: {improvement:.1f}%")
    print(f"• Graph size range: {int(df['n_edges'].min()):,} - {int(df['n_edges'].max()):,} edges")
    print(f"• Best configuration: k_diag={int(best['k_diag'])}, k_BERT={int(best['k_bert'])}, {best['method']}, {best['rewiring']}")
    
    # Method comparison
    method_stats = df.groupby('method')['r2_mean'].agg(['mean', 'std', 'count'])
    print(f"\n🔧 Method Performance:")
    for method, stats in method_stats.iterrows():
        print(f"• {method:8s}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})")
    
    # Rewiring comparison
    rewiring_stats = df.groupby('rewiring')['r2_mean'].agg(['mean', 'std', 'count'])
    print(f"\n🕸️ Rewiring Performance:")
    for rewiring, stats in rewiring_stats.iterrows():
        formatted_name = rewiring.replace('_', ' ').title()
        print(f"• {formatted_name:10s}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Generate S²G-Net publication figures")
    parser.add_argument('--results_dir', type=str, default='results/graph',
                       help='Directory containing all_results.csv')
    parser.add_argument('--top_n', type=int, default=10,
                       help='Number of top configurations for table')
    
    args = parser.parse_args()
    
    print("🎨 S²G-Net Publication Figure Generator")
    print("=" * 50)
    
    try:
        # Load results
        df = load_results(args.results_dir)
        
        # Create output directory
        output_dir = Path(args.results_dir) / 'graph_output'
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate figures
        print(f"📈 Generating figures...")
        figure1_and_3_combined(df, output_dir)
        figure2_pareto_frontier(df, output_dir)
        
        # Generate tables
        print(f"📋 Generating tables...")
        table1_top_configurations(df, output_dir, args.top_n)
        table_appendix_full_ranking(df, output_dir)
        
        # Summary statistics
        generate_summary_stats(df)
        
        print(f"\n✅ All materials saved to: {output_dir}/")
        print(f"📄 Generated files:")
        print(f"  • figure1_combined_analysis.pdf")
        print(f"  • figure2_pareto_frontier.pdf")
        print(f"  • table1_top_configurations.csv/.tex")
        print(f"  • appendix_full_ranking.csv/.tex")
        
        print(f"\n🎯 Ready for publication!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())