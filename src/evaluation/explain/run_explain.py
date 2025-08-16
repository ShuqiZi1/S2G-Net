# src/evaluation/explain/run_explain.py
"""
Main script to run all interpretability experiments
"""
import argparse
import torch
from pathlib import Path
import json
import sys

# Import implementations
from src.evaluation.explain.global_shap import run_global_shap
from src.evaluation.explain.temporal_attr import run_temporal_attribution
from src.evaluation.explain.graph_explain import run_graph_explanation
from src.evaluation.ablation.ablate import run_all_ablations, run_ablation_experiment
from src.evaluation.explain.plotter import (
    plot_lambda_dynamics, plot_reliability_diagram, 
    plot_umap_embeddings
)


def run_interpretability_experiments(model, dataset, loader, subgraph_loader, 
                                   ts_loader, config, device='cuda', 
                                   results_dir='results'):
    """Run all interpretability experiments"""
    print("\n" + "="*60)
    print("Running interpretability experiments")
    print("="*60)
    
    explain_dir = Path(results_dir) / 'explain' / 'explain_viz'
    explain_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Global SHAP
    print("\n1. Running Global SHAP analysis...")
    try:
        shap_results = run_global_shap(
            model, ts_loader, str(config.get("data_dir")),
            device=device, save_dir=explain_dir
        )
    except Exception as e:
        print(f"Error in SHAP analysis: {e}")
        shap_results = None
    
    # 2. Temporal Attribution
    print("\n2. Running Temporal Attribution...")
    try:
        temporal_results = run_temporal_attribution(
            model, ts_loader, device=device,
            save_dir=explain_dir
        )
    except Exception as e:
        print(f"Error in temporal attribution: {e}")
        temporal_results = None
        
    # 3. Additional plots
    print("\n3. Creating reliability and UMAP Embeddings visualizations...")
    
    # === 1) Reliability Diagram ===
    print("Plotting reliability diagram...")
    try:
        from src.evaluation.explain.plotter import plot_reliability_diagram
    
        # Collect predictions and true values from ts_loader
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            seq        = dataset.data.x.to(device)                  # [N, 48, 14]
            flat       = dataset.data.flat.to(device)               # [N, d]
            edge_index = dataset.data.edge_index.to(device)         # [2, E]
            y_true     = dataset.data.y.to(device)                  # [N]
    
            preds, _ = model(seq, flat, [(edge_index, None, None)], batch_size=seq.size(0))
            preds = torch.expm1(preds)
            y_true = torch.expm1(y_true)
            
            y_pred = preds.squeeze().cpu().numpy()
            y_true = y_true.squeeze().cpu().numpy()
    
            plot_reliability_diagram(y_true, y_pred, save_dir=explain_dir)

    except Exception as e:
        print(f"Error plotting reliability diagram: {e}")
    
    # === 2) UMAP Embeddings ===
    print("Plotting UMAP embeddings...")
    try:
        from src.evaluation.explain.plotter import plot_umap_embeddings

        model.eval()
        with torch.no_grad():
            node_feats = model.mamba_to_gps(model.mamba_norm(
                model.mamba_encoder(dataset.data.x.to(device),
                                    (dataset.data.x.abs().sum(-1) > 0).to(device))[0]
            )).cpu().numpy()
            import numpy as np
            labels = np.log1p(dataset.data.y)
            
            plot_umap_embeddings(node_feats, labels, save_dir=explain_dir)
    except Exception as e:
        print(f"Error plotting UMAP: {e}")
    
    # 4. Graph Explanation
    print("\n4. Running Graph Explanation...")
    try:
        graph_results = run_graph_explanation(
            model, dataset, device=device,
            save_dir=explain_dir
        )
    except Exception as e:
        print(f"Error in graph explanation: {e}")
        graph_results = None
        
    print("\n" + "="*60)
    print("Interpretability experiments complete!")
    print(f"Results saved to {explain_dir}")
    print("="*60)
    
    return {
        'shap': shap_results,
        'temporal': temporal_results,
        'graph': graph_results
    }


def main():
    """Main entry point - can be called from training script"""
    parser = argparse.ArgumentParser(description='Run interpretability experiments')
    parser.add_argument('--mode', type=str, choices=['explain', 'ablate'], 
                        required=True, help='Execution mode')
    parser.add_argument('--abl', type=str, default=None,
                        help='Specific ablation to run')
    parser.add_argument('--model_path', type=str, 
                        help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str,
                        help='Path to model configuration')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    
    # Allow being called with pre-loaded model and config
    if len(sys.argv) == 1:
        # Being called from another script
        return
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config_path:
        with open(args.config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Note: Model loading would be done here in full implementation
    print(f"Mode: {args.mode}")
    print(f"Results will be saved to: {results_dir}")
    
    if args.mode == 'explain':
        print("Run interpretability experiments...")
        # Would call run_interpretability_experiments with loaded model
    elif args.mode == 'ablate':
        if args.abl:
            print(f"Run specific ablation: {args.abl}")
        else:
            print("Run all ablations...")
        # Would call ablation functions with loaded model


if __name__ == '__main__':
    main()