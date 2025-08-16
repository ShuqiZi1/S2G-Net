# src/evaluation/explain/viz_style.py

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# -------- Publication defaults --------
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
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

COLOR = dict(
    temporal = ['#1E466E', '#376895', '#528FAD', '#8AC8E3', '#B8E5F5'],
    feature  = ['#326935', '#36993B', '#72BFCF', '#37679F', '#c8e8c9'],
    arch     = ['#FFAC73', '#FFCA73'],
    graph    = ['#E33F3F', '#F86D5E', '#FF6360'],
    baseline = ['#434343']
)
    
def white_to(c):
    return mcolors.LinearSegmentedColormap.from_list(f'white_{c}', ['white', c])
