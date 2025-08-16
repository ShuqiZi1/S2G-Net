# src/evaluation/explain/graph_explain.py
"""
Graph explainability using GNNExplainer - Complete Replacement with Publication Styling
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import seaborn as sns
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer

# Apply publication-quality settings
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
})


class FlatWrapper(torch.nn.Module):
    def __init__(self, model, flat_feats, sub_nodes):
        super().__init__()
        self.model = model
        self.flat_feats = flat_feats
        self.sub_nodes = sub_nodes

    def forward(self, x, edge_index):
        sub_flat = self.flat_feats[self.sub_nodes]
        return self.model(x, edge_index, edge_attr=None, batch=None, flat=sub_flat)


def extract_subgraph(edge_index: torch.Tensor, node_idx: int, 
                     num_hops: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract k-hop subgraph around a node"""
    device = edge_index.device
    
    # Start with the target node
    subgraph_nodes = {node_idx}
    
    # Expand for k hops
    for _ in range(num_hops):
        new_nodes = set()
        for node in subgraph_nodes:
            # Find neighbors
            mask = (edge_index[0] == node) | (edge_index[1] == node)
            neighbors = edge_index[:, mask].flatten().unique().tolist()
            new_nodes.update(neighbors)
        subgraph_nodes.update(new_nodes)
    
    # Create node mapping
    subgraph_nodes = sorted(list(subgraph_nodes))
    node_mapping = {old: new for new, old in enumerate(subgraph_nodes)}
    
    # Extract edges
    subgraph_mask = torch.zeros(edge_index.shape[1], dtype=torch.bool, device=device)
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        if src in subgraph_nodes and dst in subgraph_nodes:
            subgraph_mask[i] = True
    
    subgraph_edge_index = edge_index[:, subgraph_mask].clone()
    
    # Remap nodes
    remapped_edges = torch.tensor(
        [[node_mapping[src.item()] for src in subgraph_edge_index[0]],
         [node_mapping[dst.item()] for dst in subgraph_edge_index[1]]],
        device=device
    )
    
    return remapped_edges, torch.tensor(subgraph_nodes, device=device)


def visualize_subgraph_explanation(edge_index, edge_importance, nodes, 
                                   target_idx, original_idx, save_path):
    """Visualize explained subgraph with publication-ready styling"""
    
    if edge_importance is not None:
        topk = max(1, int(0.2 * edge_importance.numel()))
        top_indices = edge_importance.topk(topk).indices
        edge_index = edge_index[:, top_indices]
        edge_importance = edge_importance[top_indices]
    
    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    
    # Add edges with importance weights
    edge_list = []
    edge_weights = []
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        if src < dst:  # Avoid duplicate edges
            edge_list.append((src, dst))
            edge_weights.append(edge_importance[i].item())
    
    G.add_edges_from(edge_list)
    
    # Layout
    N = len(nodes)
    if N > 50:
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, k=1/np.sqrt(N), iterations=30)
    
    # Calculate node importance
    node_scores = edge_importance.abs() if edge_importance is not None else torch.ones(edge_index.shape[1])
    node_importance = torch.zeros(len(nodes))
    for i in range(edge_index.shape[1]):
        node_importance[edge_index[0, i]] += node_scores[i]
        node_importance[edge_index[1, i]] += node_scores[i]
    top_nodes = node_importance.topk(min(10, len(nodes))).indices.tolist()
    
    # Create figure with consistent styling
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Use consistent colors from palette
    node_colors = []
    for i in range(len(nodes)):
        if i == target_idx:
            node_colors.append('#E33F3F')  # Red for target
        elif i in top_nodes:
            node_colors.append('#FFAC73')  # Orange for important
        else:
            node_colors.append('#E0E0E0')  # Light gray for others
    
    # Draw nodes with consistent style
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                          node_size=[300 if i in top_nodes else 80 for i in range(len(nodes))],
                          alpha=0.9, edgecolors='black', linewidths=0.5)
    
    # Draw edges with importance-based width
    edge_widths = [1 + 3 * (w / max(edge_weights)) for w in edge_weights]
    edge_colors = ['#5A8CA4' if w > np.median(edge_weights) else '#B8E5F5' 
                   for w in edge_weights]
    
    nx.draw_networkx_edges(G, pos, edge_list, width=edge_widths, 
                          edge_color=edge_colors, alpha=0.7)
    
    # Labels only for important nodes
    labels = {i: str(nodes[i].item()) for i in top_nodes}
    if target_idx not in top_nodes:
        labels[target_idx] = f'{nodes[target_idx].item()}'
    
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, 
                           font_family='sans-serif')
    
    # Title and formatting
    ax.set_title(f'Subgraph Explanation for Node {original_idx}', 
                fontsize=11, pad=10)
    ax.axis('off')
    
    # Remove the frame
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=600, bbox_inches='tight', 
                facecolor='white')
    plt.close()


def create_edge_density_plot(explanations, save_dir):
    """Create edge importance density plot with consistent styling"""
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Collect all edge importance values
    all_importances = []
    for node_idx, exp in explanations.items():
        if 'edge_mask' in exp:
            all_importances.extend(exp['edge_mask'].cpu().numpy().flatten())
    
    # Create histogram with our color palette
    ax.hist(all_importances, bins=30, alpha=0.7, 
            color='#5A8CA4', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Edge Importance Score', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Distribution of Edge Importance Scores', fontsize=11, pad=10)
    
    # Add statistics
    mean_imp = np.mean(all_importances)
    median_imp = np.median(all_importances)
    
    ax.axvline(mean_imp, color='#E33F3F', linestyle='--', 
               linewidth=2, label=f'Mean: {mean_imp:.3f}')
    ax.axvline(median_imp, color='#FFAC73', linestyle='--', 
               linewidth=2, label=f'Median: {median_imp:.3f}')
    
    ax.legend(frameon=True, edgecolor='#E0E0E0', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/edge_importance_distribution.pdf", 
                dpi=600, bbox_inches='tight')
    plt.close()


def run_graph_explanation(model, dataset, device='cuda', save_dir='./figs/explain',
                          target_nodes=None, n_nodes=5):
    """
    Run graph explanation analysis
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Get graph data
    edge_index = dataset.data.edge_index.to(device)
    edge_attr = dataset.data.edge_attr.to(device) if hasattr(dataset.data, 'edge_attr') else None
    x = dataset.data.x.to(device)
    
    with torch.no_grad():
        seq_full  = dataset.data.x.to(device)                 # [N, T, 14]
        mask_full = (seq_full.abs().sum(-1) > 0)
        last, _ = model.mamba_encoder(seq_full, mask_full)  # last is [N, d]
        last    = model.mamba_norm(last)                    # just normalise
        gps_ts_all = model.mamba_to_gps(last)                 # [N, 128]
    
        flat_feats = dataset.data.flat.to(device)
        gps_node_feats = gps_ts_all                           # 128-D
        
    # Select target nodes if not specified
    if target_nodes is None:
        # Select nodes randomly from test set
        test_mask = dataset.data.test_mask
        test_indices = torch.where(test_mask)[0]
        if len(test_indices) > n_nodes:
            selected = torch.randperm(len(test_indices))[:n_nodes]
            target_nodes = test_indices[selected].tolist()
        else:
            target_nodes = test_indices.tolist()
    
    explanations = {}
    
    # Initialize GNNExplainer
    try:
        from torch_geometric.explain.algorithm import GNNExplainer as AlgoGNNExplainer
        from torch_geometric.explain import Explainer
        from torch_geometric.explain.config import ModelConfig
    
        algo = AlgoGNNExplainer(
            epochs=200,
            lr=0.01,
            feat_mask_type='individual_feature'
        )
    
        model_cfg = ModelConfig(
            mode='regression', 
            task_level='node',
            return_type='raw'
        )
        
    except (ImportError, TypeError):
        from torch_geometric.nn.models import GNNExplainer as LegacyGNNExplainer
        explainer = LegacyGNNExplainer(
            model.graphgps_encoder,
            epochs=50,
            
            lr=0.01,
            return_type='raw',
            feat_mask_type='individual_feature'
        )

    for i, node_idx in enumerate(target_nodes):
        print(f"Explaining node {node_idx} ({i+1}/{len(target_nodes)})...")
        
        # Extract subgraph
        sub_edge_index, sub_nodes = extract_subgraph(edge_index, node_idx)
    
        # Create explainer using per-node FlatWrapper
        wrapped_model = FlatWrapper(model.graphgps_encoder, flat_feats, sub_nodes)
        explainer = Explainer(
            model=wrapped_model,
            algorithm=algo,
            model_config=model_cfg,
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object'
        )
            
        # Get node features for subgraph
        sub_x   = gps_node_feats[sub_nodes]
        sub_flat = flat_feats[sub_nodes]
        if sub_x.dim() == 3:              # [N, T, d]
            sub_x = sub_x[:, -1, :] 
        
        # Map target node to subgraph
        target_in_subgraph = (sub_nodes == node_idx).nonzero(as_tuple=True)[0].item()
        
        # Explain
        try:
            explanation = explainer(
                x=gps_node_feats[sub_nodes],
                edge_index=sub_edge_index
            )                            
            edge_mask = explanation.edge_mask if hasattr(explanation, "edge_mask") else explanation.edge_mask_dict["object"]
            k = max(1, int(0.2 * edge_mask.numel()))      # keep top-20 %
            topk = edge_mask.topk(k).indices
            keep_mask = torch.zeros_like(edge_mask, dtype=torch.bool)
            keep_mask[topk] = True

            # forward with *masked* edges → measure ΔRMSE
            with torch.no_grad():
                full_pred = wrapped_model(gps_node_feats[sub_nodes], sub_edge_index)
                sparse_pred = wrapped_model(gps_node_feats[sub_nodes], sub_edge_index[:, keep_mask])

            fidelity = torch.sqrt(((full_pred - sparse_pred) ** 2).mean()).item()
            sparsity = 1.0 - k / edge_mask.numel()

            # -------------- Edge-type statistics --------------           # <<< NEW
            if edge_attr is not None and edge_attr.size(1) > 1:
                edge_types_sub = edge_attr[sub_edge_index[0]]       # assumes type one-hot
                is_cross = (edge_types_sub[:, 1] > 0) if edge_types_sub.size(1) > 1 else None
                if is_cross is not None:
                    same_imp  = edge_mask[~is_cross].mean().item()
                    cross_imp = edge_mask[is_cross].mean().item()
            else:
                same_imp = cross_imp = None

            node_feat_mask = explanation.get('node_mask')
            edge_mask      = explanation.get('edge_mask')
        except Exception as e:
            print(f"Error explaining node {node_idx}: {e}")
            continue
        
        # Store explanation
        explanations[node_idx] = {
            'edge_mask': edge_mask.detach().cpu(),
            'fidelity': fidelity,
            'sparsity': sparsity,
            'imp_same_diag': same_imp,
            'imp_cross_diag': cross_imp
        }
        
        # Visualize subgraph
        visualize_subgraph_explanation(
            sub_edge_index, edge_mask, sub_nodes, 
            target_in_subgraph, node_idx,
            save_path=f"{save_dir}/subgraph_node{node_idx}.png"
        )
    
    # Create edge density analysis
    if explanations:
        create_edge_density_plot(explanations, save_dir)

    # === Save fidelity & sparsity summary to CSV ===
    import pandas as pd
    summary = {
        'node_idx': [],
        'fidelity': [],
        'sparsity': [],
        'imp_same_diag': [],
        'imp_cross_diag': []
    }
    for node_id, result in explanations.items():
        summary['node_idx'].append(node_id)
        summary['fidelity'].append(result['fidelity'])
        summary['sparsity'].append(result['sparsity'])
        summary['imp_same_diag'].append(result['imp_same_diag'])
        summary['imp_cross_diag'].append(result['imp_cross_diag'])
    
    pd.DataFrame(summary).to_csv(f"{save_dir}/explanation_summary.csv", index=False)

    print(f"Graph explanation complete. Figures saved to {save_dir}")
    
    return explanations