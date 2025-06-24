import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import pandas as pd
from scipy import stats
import os
import colorsys

# Set publication-quality defaults
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],  # Fallback fonts
    'font.size': 8,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'axes.linewidth': 0.5,
    'lines.linewidth': 1,
    'patch.linewidth': 0.5,
    'grid.alpha': 0.3,
    'axes.grid': True,
    'axes.axisbelow': True,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#000000',
    'text.color': '#000000',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
})

# Create save directory
save_dir = 'results/ablations/ablations_viz'
os.makedirs(save_dir, exist_ok=True)


# Define color palette based on your image
def lighten_color(color, amount=0.05):
    """Lighten a color by mixing with white"""
    import matplotlib.colors as mc
    import colorsys
    c = mc.cnames.get(color, color)
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], min(1, c[1] + amount), c[2])


color_palette = {
    # Temporal - Blue series (lightened)
    'temporal_1': '#5A8CA4',  # was #4A7C94
    'temporal_2': '#8AC8E3',  # was #7AB8D3
    'temporal_3': '#B8E5F5',  # was #A8D5E5

    # Feature - Green series (lightened)
    'feature_1': '#3E8D42',  # was #2E7D32
    'feature_2': '#5CBF60',  # was #4CAF50
    'feature_3': '#91D794',  # was #81C784

    # Architecture - Yellow/Orange series (lightened)
    'arch_1': '#FFAC73',  # was #F39C63
    'arch_2': '#FFCA73',  # was #FAB763

    # Graph - Red series (lightened)
    'graph_1': '#E33F3F',  # was #D32F2F
    'graph_2': '#F86D5E',  # was #E85D4E
    'graph_3': '#FF6360',  # was #EF5350

    # Baseline
    'baseline': '#434343'  # was #333333
}

softer_colors = {
    'temporal_2': '#A5D3E8',  # Even softer blue
    'feature_1': '#5E9D62',  # Softer green 1
    'feature_2': '#7CCF80',  # Softer green 2
    'feature_3': '#A1E7A4',  # Softer green 3
    'graph_2': '#FF8D7E',  # Softer red
    'baseline': '#636363'  # Softer baseline
}

# Nature-style color palette (colorblind-friendly) - keeping as backup
nature_colors = {
    'blue': '#0173B2',  # Baseline/primary
    'orange': '#DE8F05',  # Temporal
    'green': '#029E73',  # Feature
    'red': '#CC78BC',  # Architecture
    'purple': '#949494',  # Graph
    'gray': '#656565'  # Reference
}

# Your data
ablation_results = {
    'baseline': {'r2': 0.411, 'kappa': 0.357, 'mse': 0.252, 'msle': 0.225, 'mad': 0.385, 'mape': 40.77},

    # Temporal Window Analysis
    'last6h': {'r2': 0, 'kappa': 0, 'mse': 0.670, 'msle': 0.399, 'mad': 0.654, 'mape': 65.07},
    'last24h': {'r2': 0.348, 'kappa': 0.315, 'mse': 0.282, 'msle': 0.258, 'mad': 0.405, 'mape': 42.64},
    'full48h': {'r2': 0.411, 'kappa': 0.357, 'mse': 0.252, 'msle': 0.225, 'mad': 0.385, 'mape': 40.77},

    # Feature Group Impact
    'remove_physio': {'r2': 0.350, 'kappa': 0.291, 'mse': 0.281, 'msle': 0.256, 'mad': 0.416, 'mape': 45.41},
    'remove_vitals': {'r2': 0.373, 'kappa': 0.318, 'mse': 0.271, 'msle': 0.232, 'mad': 0.402, 'mape': 42.89},
    'remove_ethnicity': {'r2': 0.409, 'kappa': 0.356, 'mse': 0.255, 'msle': 0.226, 'mad': 0.385, 'mape': 40.94},

    # Modality Analysis
    'static_only': {'r2': 0, 'kappa': 0, 'mse': 1.513, 'msle': 0.448, 'mad': 1.021, 'mape': 80.87},
    'no_static': {'r2': 0, 'kappa': 0, 'mse': 0.851, 'msle': 0.368, 'mad': 0.679, 'mape': 67.09},

    # Graph Robustness
    'drop_edges_30': {'r2': 0.411, 'kappa': 0.356, 'mse': 0.255, 'msle': 0.226, 'mad': 0.385, 'mape': 41.26},
    'drop_edges_50': {'r2': 0.402, 'kappa': 0.346, 'mse': 0.263, 'msle': 0.229, 'mad': 0.393, 'mape': 41.83},
    'drop_edges_70': {'r2': 0.385, 'kappa': 0.327, 'mse': 0.277, 'msle': 0.236, 'mad': 0.412, 'mape': 42.79}
}

ablation_stds = {
    'baseline': {'r2': 0.005, 'kappa': 0.007, 'mse': 0.003, 'msle': 0.001, 'mad': 0.002, 'mape': 0.341},

    # Temporal Window Analysis
    'last6h': {'r2': 0, 'kappa': 0, 'mse': 0.012, 'msle': 0.013, 'mad': 0.023, 'mape': 0.429},
    'last24h': {'r2': 0.002, 'kappa': 0.001, 'mse': 0.003, 'msle': 0.004, 'mad': 0.003, 'mape': 0.473},
    'full48h': {'r2': 0.005, 'kappa': 0.007, 'mse': 0.003, 'msle': 0.001, 'mad': 0.002, 'mape': 0.341},

    # Feature Group Impact
    'remove_physio': {'r2': 0.017, 'kappa': 0.001, 'mse': 0.000, 'msle': 0.005, 'mad': 0.005, 'mape': 0.793},
    'remove_vitals': {'r2': 0.009, 'kappa': 0.001, 'mse': 0.000, 'msle': 0.006, 'mad': 0.003, 'mape': 0.557},
    'remove_ethnicity': {'r2': 0.004, 'kappa': 0.001, 'mse': 0.003, 'msle': 0.002, 'mad': 0.003, 'mape': 0.472},

    # Modality Analysis
    'static_only': {'r2': 0, 'kappa': 0, 'mse': 0.198, 'msle': 0.095, 'mad': 0.112, 'mape': 3.685},
    'no_static': {'r2': 0, 'kappa': 0, 'mse': 0.019, 'msle': 0.045, 'mad': 0.031, 'mape': 1.431},

    # Graph Robustness
    'drop_edges_30': {'r2': 0.004, 'kappa': 0.003, 'mse': 0.001, 'msle': 0.010, 'mad': 0.002, 'mape': 0.524},
    'drop_edges_50': {'r2': 0.008, 'kappa': 0.002, 'mse': 0.005, 'msle': 0.001, 'mad': 0.001, 'mape': 0.812},
    'drop_edges_70': {'r2': 0.023, 'kappa': 0.003, 'mse': 0.003, 'msle': 0.002, 'mad': 0.004, 'mape': 0.792}
}

# Categorize ablations
ablation_categories = {
    'Temporal': ['last6h', 'last24h', 'full48h'],
    'Feature': ['remove_physio', 'remove_vitals', 'remove_ethnicity'],
    'Architecture': ['static_only', 'no_static'],
    'Graph': ['drop_edges_30', 'drop_edges_50', 'drop_edges_70']
}

# Better labels for display
ablation_labels = {
    'baseline': 'Baseline',
    'last6h': 'Last 6h',
    'last24h': 'Last 24h',
    'full48h': 'Full 48h',
    'remove_physio': 'No Physiology',
    'remove_vitals': 'No Vitals',
    'remove_ethnicity': 'No Ethnicity',
    'static_only': 'Static Only',
    'no_static': 'No Static',
    'drop_edges_30': 'Drop 30%',
    'drop_edges_50': 'Drop 50%',
    'drop_edges_70': 'Drop 70%'
}

# Metric labels
metric_labels = {
    'kappa': 'Kappa',  # Changed from Cohen's κ
    'mad': 'MAD',
    'mape': 'MAPE (%)',
    'mse': 'MSE',
    'msle': 'MSLE',
    'r2': 'R² Score'
}


def create_metrics_heatmap():
    # Prepare data matrix
    ablations = list(ablation_results.keys())
    metrics = ['kappa', 'r2', 'mse', 'mape', 'mad', 'msle']
    data_matrix = np.array([[ablation_results[abl][m] for m in metrics] for abl in ablations])

    # Normalize each metric to [0, 1] (error metrics inverted)
    error_metrics = {'mse', 'mape', 'mad', 'msle'}
    normalized = np.zeros_like(data_matrix)
    for j, m in enumerate(metrics):
        col = data_matrix[:, j]
        norm = (col - col.min()) / (col.max() - col.min())
        normalized[:, j] = 1 - norm if m in error_metrics else norm

    # Build a white-to-blue colormap using the same blue as 'last6h'
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        'gradient',
        ['#d7e1eb', '#afc3d8', color_palette['temporal_1']]
    )

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(normalized, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # Add text annotations (metric values)
    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                    ha='center', va='center', fontsize=8, color='black')

    # Configure ticks and labels
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(['Kappa', 'R² Score', 'MSE↓', 'MAPE↓', 'MAD↓', 'MSLE↓'], fontsize=11)
    ax.set_yticks(np.arange(len(ablations)))
    ax.set_yticklabels([ablation_labels[abl] for abl in ablations], fontsize=11)
    ax.set_xlabel('Metrics', fontsize=11)
    ax.set_ylabel('Configuration', fontsize=11)

    # Add a colorbar ranging from 0 to 1
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Performance', rotation=270, labelpad=20)
    cbar.set_ticks(np.linspace(0, 1, 11))

    # Remove all spines for a cleaner look
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/ablation_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.close()


# Call the function to create heatmap
create_metrics_heatmap()

# Figure 2: Key Metrics Comparison with all metrics (2x3 layout) - UPDATED WITH LARGER LEGEND
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

metric_subtitles = [
    'R² Score',  # A
    "Kappa",  # B
    'Mean-Squared Error',  # C
    'Mean Absolute Deviation',  # D
    'Mean Absolute Percentage Error',  # E
    'Mean-Squared Log Error'  # F
]


def annotate_subplot(ax, letter, subtitle, pad=5):
    ax.text(0.5, 1.08, f"({letter}) {subtitle}",
            transform=ax.transAxes,
            fontsize=13,
            va='bottom', ha='center')


for idx, (ax, sub) in enumerate(zip(axes, metric_subtitles)):
    annotate_subplot(ax, chr(97 + idx), sub)

all_metrics = ['r2', 'kappa', 'mse', 'mad', 'mape', 'msle']


# Color mapping function with gradient colors
def get_color(abl):
    if abl == 'baseline':
        return color_palette['baseline']
    elif abl == 'last6h':
        return color_palette['temporal_1']
    elif abl == 'last24h':
        return color_palette['temporal_2']
    elif abl == 'full48h':
        return color_palette['temporal_3']
    elif abl == 'remove_physio':
        return color_palette['feature_1']
    elif abl == 'remove_vitals':
        return color_palette['feature_2']
    elif abl == 'remove_ethnicity':
        return color_palette['feature_3']
    elif abl == 'static_only':
        return color_palette['arch_1']
    elif abl == 'no_static':
        return color_palette['arch_2']
    elif abl == 'drop_edges_30':
        return color_palette['graph_1']
    elif abl == 'drop_edges_50':
        return color_palette['graph_2']
    elif abl == 'drop_edges_70':
        return color_palette['graph_3']
    else:
        return '#666666'


for idx, metric in enumerate(all_metrics):
    ax = axes[idx]

    # Prepare data
    x_labels = []
    values = []
    errors = []
    colors_list = []

    # Add all ablations
    for abl in ablation_results.keys():
        x_labels.append(ablation_labels[abl])
        values.append(ablation_results[abl][metric])
        errors.append(ablation_stds[abl][metric])
        colors_list.append(get_color(abl))

    # Create bar plot
    x_pos = np.arange(len(x_labels))
    bars = ax.bar(x_pos, values, yerr=errors, capsize=4,
                  color=colors_list, edgecolor='black', linewidth=0.5,
                  error_kw={'linewidth': 1, 'ecolor': 'black'})

    # Add baseline line
    ax.axhline(y=ablation_results['baseline'][metric],
               color='black', linestyle='--', alpha=0.5, linewidth=1)

    # Formatting
    ax.set_xlabel('')
    ax.set_ylabel(metric_labels[metric], fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=11)  # Increased label size

    # Add significance stars (positioned higher to avoid overlap)
    baseline_val = ablation_results['baseline'][metric]
    baseline_std = ablation_stds['baseline'][metric]

    for i, (val, err) in enumerate(zip(values[1:], errors[1:]), 1):
        # Simple significance test (2 std rule)
        if abs(val - baseline_val) > 2 * (baseline_std + err):
            # Calculate star position based on bar height and error bar
            star_y = val + err + 0.05 * (max(values) - min(values))
            ax.text(i, star_y, '*', ha='center', va='bottom', fontsize=12)

# Add legend with light gray box and larger font
handles = [
    mpatches.Patch(color=color_palette['baseline'], label='Baseline'),
    mpatches.Patch(color=color_palette['temporal_2'], label='Temporal'),
    mpatches.Patch(color=color_palette['feature_2'], label='Feature'),
    mpatches.Patch(color=color_palette['arch_1'], label='Architecture'),
    mpatches.Patch(color=color_palette['graph_2'], label='Graph')
]

# Position legend above plots with light gray box
legend = fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1),
                    ncol=5, frameon=True, columnspacing=2, fontsize=12,  # Increased font size
                    edgecolor='#E0E0E0', framealpha=0.9)  # Light gray box
legend.get_frame().set_linewidth(0.5)  # Thin border

plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Make more room for legend
plt.savefig(f'{save_dir}/ablation_key_metrics.pdf', bbox_inches='tight', dpi=600)
plt.show()

# Figure 3: Grouped Performance Drop - Fixed alignment
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate performance drops relative to baseline
performance_drops = {}
for cat_name, cat_ablations in ablation_categories.items():
    drops = []
    labels = []
    for abl in cat_ablations:
        # Use R² as primary metric
        baseline_r2 = ablation_results['baseline']['r2']
        abl_r2 = ablation_results[abl]['r2']
        drop = (baseline_r2 - abl_r2) / abs(baseline_r2) * 100
        drops.append(drop)
        labels.append(ablation_labels[abl])
    performance_drops[cat_name] = (drops, labels)

# Plot grouped bars with proper alignment
categories = list(ablation_categories.keys())
n_categories = len(categories)
bar_width = 0.25
group_spacing = 0.1

# Calculate positions for each group
group_positions = []
current_pos = 0
for i, cat in enumerate(categories):
    n_bars = len(performance_drops[cat][0])
    group_positions.append(current_pos)
    current_pos += n_bars * bar_width + group_spacing

# Plot bars
for i, cat_name in enumerate(categories):
    drops, labels = performance_drops[cat_name]

    # Use gradient colors for each category
    if cat_name == 'Temporal':
        cat_colors = [color_palette['temporal_1'], color_palette['temporal_2'], color_palette['temporal_3']]
    elif cat_name == 'Feature':
        cat_colors = [color_palette['feature_1'], color_palette['feature_2'], color_palette['feature_3']]
    elif cat_name == 'Architecture':
        cat_colors = [color_palette['arch_1'], color_palette['arch_2']]
    else:  # Graph
        cat_colors = [color_palette['graph_1'], color_palette['graph_2'], color_palette['graph_3']]

    # Plot bars for this category
    for j, (drop, label) in enumerate(zip(drops, labels)):
        x_pos = group_positions[i] + j * bar_width
        ax.bar(x_pos, drop, bar_width, label=label,
               color=cat_colors[j % len(cat_colors)],
               edgecolor='black', linewidth=0.5)

# Calculate center positions for category labels
category_centers = []
for i, cat in enumerate(categories):
    n_bars = len(performance_drops[cat][0])
    center = group_positions[i] + (n_bars - 1) * bar_width / 2
    category_centers.append(center)

# Formatting
ax.set_ylabel('Performance Drop from Baseline (%)', fontsize=11)
ax.set_xticks(category_centers)
ax.set_xticklabels(categories)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{save_dir}/ablation_grouped_impact.pdf', bbox_inches='tight', dpi=600)
plt.show()

# Figure 4: Component Analysis (without overall title) - UPDATED WITH DASHED LINES AND SOFTER COLORS
fig, axes = plt.subplots(2, 3, figsize=(14, 6))

# First row: R² Score
# Panel A: Time window effect (R²)
ax = axes[0, 0]
time_windows = [6, 24, 48]
time_r2 = [ablation_results['last6h']['r2'],
           ablation_results['last24h']['r2'],
           ablation_results['full48h']['r2']]

ax.set_title('(a) Effect of Time Window on R²', fontsize=13)
ax.plot(time_windows, time_r2, marker='o', markersize=6,
        color=softer_colors['temporal_2'], linewidth=2, linestyle='--',  # Dashed and thicker
        markeredgecolor='black', markeredgewidth=0.5)
ax.axhline(y=ablation_results['baseline']['r2'],
           color=softer_colors['baseline'], linestyle='--', linewidth=0.5)
ax.set_ylabel('R² Score', fontsize=11)
ax.set_xlim(0, 54)
ax.set_xticks([6, 24, 48])
ax.grid(True, alpha=0.3)

# Panel B: Feature importance (R²)
ax = axes[0, 1]
features = ['Phys', 'Vital', 'Ethn']
feature_r2_drop = [
    (ablation_results['baseline']['r2'] - ablation_results['remove_physio']['r2']),
    (ablation_results['baseline']['r2'] - ablation_results['remove_vitals']['r2']),
    (ablation_results['baseline']['r2'] - ablation_results['remove_ethnicity']['r2'])
]

x_pos = np.arange(len(features))
ax.set_title('(b) Static Feature Importance (R²)', fontsize=13)
bars = ax.bar(x_pos, feature_r2_drop,
              color=[softer_colors['feature_1'], softer_colors['feature_2'], softer_colors['feature_3']],
              edgecolor='black', linewidth=0.5, width=0.6)

# Add value labels
for i, v in enumerate(feature_r2_drop):
    ax.text(i, v + 0.002, f'{v:.3f}', ha='center', va='bottom', fontsize=7)

ax.set_ylabel('R² Score', fontsize=11)
ax.set_xticks(x_pos)
ax.set_xticklabels(features)
ax.set_ylim(0, max(feature_r2_drop) * 1.2)
ax.grid(axis='y', alpha=0.3)

# Panel C: Edge dropping effect (R²) - without error bars
ax = axes[0, 2]
drop_rates = [30, 50, 70]
drop_r2 = [ablation_results['drop_edges_30']['r2'],
           ablation_results['drop_edges_50']['r2'],
           ablation_results['drop_edges_70']['r2']]

ax.set_title('(c) Graph Edge Removal Impact on R²', fontsize=13)
ax.plot(drop_rates, drop_r2, marker='s', markersize=6,
        color=softer_colors['graph_2'], linewidth=2, linestyle='--',  # Dashed and thicker
        markeredgecolor='black', markeredgewidth=0.5)
ax.axhline(y=ablation_results['baseline']['r2'],
           color=softer_colors['baseline'], linestyle='--', linewidth=0.5)
ax.set_ylabel('R² Score', fontsize=11)
ax.set_xlim(20, 80)
ax.set_xticks([30, 50, 70])
ax.grid(True, alpha=0.3)

# Second row: MSE
# Panel D: Time window effect (MSE)
ax = axes[1, 0]
time_mse = [ablation_results['last6h']['mse'],
            ablation_results['last24h']['mse'],
            ablation_results['full48h']['mse']]

ax.set_title('(d) Effect of Time Window on MSE', fontsize=13)
ax.plot(time_windows, time_mse, marker='o', markersize=6,
        color=softer_colors['temporal_2'], linewidth=2, linestyle='--',  # Dashed and thicker
        markeredgecolor='black', markeredgewidth=0.5)
ax.axhline(y=ablation_results['baseline']['mse'],
           color=softer_colors['baseline'], linestyle='--', linewidth=0.5)
ax.set_xlabel('Time Window (h)', fontsize=11)
ax.set_ylabel('MSE', fontsize=11)
ax.set_xlim(0, 54)
ax.set_xticks([6, 24, 48])
ax.grid(True, alpha=0.3)

# Panel E: Feature importance (MSE)
ax = axes[1, 1]
feature_mse_increase = [
    (ablation_results['remove_physio']['mse'] - ablation_results['baseline']['mse']),
    (ablation_results['remove_vitals']['mse'] - ablation_results['baseline']['mse']),
    (ablation_results['remove_ethnicity']['mse'] - ablation_results['baseline']['mse'])
]

ax.set_title('(e) Static Feature Impact (MSE)', fontsize=13)
bars = ax.bar(x_pos, feature_mse_increase,
              color=[softer_colors['feature_1'], softer_colors['feature_2'], softer_colors['feature_3']],
              edgecolor='black', linewidth=0.5, width=0.6)

# Add value labels
for i, v in enumerate(feature_mse_increase):
    ax.text(i, v + 0.002, f'{v:.3f}', ha='center', va='bottom', fontsize=7)

ax.set_xlabel('Feature Group', fontsize=11)
ax.set_ylabel('MSE', fontsize=11)
ax.set_xticks(x_pos)
ax.set_xticklabels(features)
ax.set_ylim(0, max(feature_mse_increase) * 1.2)
ax.grid(axis='y', alpha=0.3)

# Panel F: Edge dropping effect (MSE)
ax = axes[1, 2]
drop_mse = [ablation_results['drop_edges_30']['mse'],
            ablation_results['drop_edges_50']['mse'],
            ablation_results['drop_edges_70']['mse']]

ax.set_title('(f) Graph Edge Removal Impact on MSE', fontsize=13)
ax.plot(drop_rates, drop_mse, marker='s', markersize=6,
        color=softer_colors['graph_2'], linewidth=2, linestyle='--',  # Dashed and thicker
        markeredgecolor='black', markeredgewidth=0.5)
ax.axhline(y=ablation_results['baseline']['mse'],
           color=softer_colors['baseline'], linestyle='--', linewidth=0.5)
ax.set_xlabel('Edge Drop Rate (%)', fontsize=11)
ax.set_ylabel('MSE', fontsize=11)
ax.set_xlim(20, 80)
ax.set_xticks([30, 50, 70])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{save_dir}/figure_ablation_details.pdf', bbox_inches='tight', dpi=600)
plt.show()

print("All figures have been generated successfully!")
print(f"\nFigures saved to: {save_dir}/")
print("1. ablation_heatmap.pdf - Performance heatmap with gradient colors")
print("2. ablation_key_metrics.pdf - All metrics comparison (with larger legend in light gray box)")
print("3. ablation_grouped_impact.pdf - Grouped performance impact")
print("4. figure_ablation_details.pdf - Component analysis (with dashed lines and softer colors)")