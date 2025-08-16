# src/evaluation/explain/global_shap.py
"""
Global SHAP analysis for static features using DeepSHAP
"""
import torch
import torch.nn as nn
import numpy as np
import shap
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
import json   
from pathlib import Path 
from shap import kmeans
import re
import src.evaluation.explain.viz_style as vs
import matplotlib.patches as mpatches

class FlatBranch(nn.Module):
    """Wrapper for flat/static feature branch"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, flat_features: torch.Tensor) -> torch.Tensor:
        """Process only flat features through the model by zeroing other branches"""
        batch_size = flat_features.size(0)
        device = flat_features.device
        
        # Create dummy inputs for other branches
        # Assuming model expects: (seq, flat, edge_stuff...)
        dummy_seq = torch.zeros(batch_size, 48, self.model.mamba_encoder.input_dim, device=device)
        dummy_masks = torch.ones(batch_size, 48, device=device)
        
        # For GraphGPS, create minimal dummy graph inputs
        dummy_edge_index = torch.tensor([[0], [0]], device=device)
        dummy_batch_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Call the actual model with zeroed time series and graph inputs
        # This preserves the learned flat processing pathway
        with torch.no_grad():
            # Temporarily set lambda values to emphasize flat features
            def _get_param(obj, name):
                if hasattr(obj, name):
                    return getattr(obj, name)
                if hasattr(obj, "graphgps_encoder") and hasattr(obj.graphgps_encoder, name):
                    return getattr(obj.graphgps_encoder, name)
                return None
            
            lambda_ts  = _get_param(self.model, "lambda_ts")
            lambda_gps = _get_param(self.model, "lambda_gps")
            
            if lambda_ts is not None:
                orig_lambda_ts = lambda_ts.data.clone()
                lambda_ts.data.fill_(0.0)
            else:
                orig_lambda_ts = None
            
            if lambda_gps is not None:
                orig_lambda_gps = lambda_gps.data.clone()
                lambda_gps.data.fill_(0.0)
            else:
                orig_lambda_gps = None
        
        # Forward pass
        out, _ = self.model(
                dummy_seq, flat_features,
                [(dummy_edge_index, None, None)],
                batch_size, None)
        
        # Restore lambda values
        if lambda_ts is not None:
            lambda_ts.data.copy_(orig_lambda_ts)
        if lambda_gps is not None:
            lambda_gps.data.copy_(orig_lambda_gps)
        
        return out if out.dim() == 2 else out.unsqueeze(1)

FLAT_MAP = {
    "gender": "Gender",
    "age": "Age (Years)",
    "height": "Height (cm)",
    "weight": "Weight (kg)",
    "hour": "Admission Hour",
    "eyes": "GCS - Eye Opening",
    "motor": "GCS - Motor",
    "verbal": "GCS - Verbal",
    
    # Ethnicity
    "ethnicity_ASIAN": "Ethnicity: Asian",
    "ethnicity_ASIAN - CHINESE": "Ethnicity: Chinese",
    "ethnicity_BLACK/AFRICAN AMERICAN": "Ethnicity: Black/African American",
    "ethnicity_HISPANIC/LATINO - PUERTO RICAN": "Ethnicity: Hispanic/Latino",
    "ethnicity_OTHER": "Ethnicity: Other",
    "ethnicity_UNABLE TO OBTAIN": "Ethnicity: Unknown (Unable to Obtain)",
    "ethnicity_UNKNOWN": "Ethnicity: Unknown",
    "ethnicity_WHITE": "Ethnicity: White",
    "ethnicity_WHITE - OTHER EUROPEAN": "Ethnicity: White (European)",
    "ethnicity_misc": "Ethnicity: Other/Mixed",

    # Care Units
    "first_careunit_Cardiac Vascular Intensive Care Unit (CVICU)": "Unit: CVICU",
    "first_careunit_Coronary Care Unit (CCU)": "Unit: CCU",
    "first_careunit_Medical Intensive Care Unit (MICU)": "Unit: MICU",
    "first_careunit_Medical/Surgical Intensive Care Unit (MICU/SICU)": "Unit: MICU/SICU",
    "first_careunit_Neuro Intermediate": "Unit: Neuro-Intermediate",
    "first_careunit_Neuro Stepdown": "Unit: Neuro-Stepdown",
    "first_careunit_Neuro Surgical Intensive Care Unit (Neuro SICU)": "Unit: Neuro-SICU",
    "first_careunit_Surgical Intensive Care Unit (SICU)": "Unit: SICU",
    "first_careunit_Trauma SICU (TSICU)": "Unit: TSICU",
    "first_careunit_misc": "Unit: Other",

    # Admission Source
    "admission_location_CLINIC REFERRAL": "Clinic Referral",
    "admission_location_EMERGENCY ROOM": "Emergency Dept.",
    "admission_location_PHYSICIAN REFERRAL": "Physician Referral",
    "admission_location_PROCEDURE SITE": "Procedure Site",
    "admission_location_TRANSFER FROM HOSPITAL": "Hospital Transfer",
    "admission_location_TRANSFER FROM SKILLED NURSING FACILITY": "Nursing Facility",
    "admission_location_WALK-IN/SELF REFERRAL": "Walk-in/Self",
    "admission_location_misc": "Other Source",

    # Insurance
    "insurance_Medicaid": "Insurance: Medicaid",
    "insurance_Medicare": "Insurance: Medicare",
    "insurance_Other": "Insurance: Other",
    "insurance_Private": "Insurance: Private",
    "insurance_misc": "Insurance: Other/Mixed",
}


def prettify_flat(name: str) -> str:
    """Return concise, publication-ready name for a flat feature."""
    if name in FLAT_MAP:   
        return FLAT_MAP[name]
    if name.startswith("first_careunit_"):
        unit = re.sub(r'\s*\(.*\)', '', name.split('_', 1)[1])
        return f"CareUnit: {unit}"
    if name.startswith("admission_location_"):
        loc = name.split('_', 1)[1].title()
        loc = loc.replace("Emergency Room", "ED")\
                 .replace("Physician Referral", "Physician")
        return f"Admission: {loc}"
    # fallback
    return name

def wrap_feature_name(name, max_len=21):
    """Wrap long feature name into multiple lines for better plotting"""
    return '\n'.join([name[i:i+max_len] for i in range(0, len(name), max_len)])
    
def run_global_shap(
    model, loader, data_dir: str = None,
    device: str = "cuda", save_dir: str = "./figs/explain",
    n_background: int = 128, n_explain: int = 300,
    n_bootstrap: int = 10):

    """
    Run global SHAP analysis on static features
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Create wrapper for flat features
    wrapper = FlatBranch(model).to(device).eval()
    
    # Collect background and explain data
    background_list = []
    explain_list = []
    
    with torch.no_grad():
        for i, (inputs, labels, ids) in enumerate(loader):
            _, flat, _ = inputs
            flat = flat.to(device)
            
            # Collection logic
            if len(background_list) * loader.batch_size < n_background:
                background_list.append(flat)
            elif len(explain_list) * loader.batch_size < n_explain:
                explain_list.append(flat)
                
            # Check if we have enough data
            total_background = sum(b.size(0) for b in background_list)
            total_explain = sum(e.size(0) for e in explain_list)
            
            if total_background >= n_background and total_explain >= n_explain:
                break
    
    # Concatenate and trim to exact sizes
    background = torch.cat(background_list, dim=0)[:n_background]
    test_data = torch.cat(explain_list, dim=0)[:n_explain]
    
    print(f"Background data shape: {background.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # ---- 1. Background compression (K-medoids) ----
    bg_np = background.cpu().numpy()
    medoid_np = kmeans(background.cpu().numpy(), k=min(10, n_background)).data
    medoids = torch.tensor(medoid_np, dtype=torch.float32, device=device)
    explainer = shap.GradientExplainer(wrapper, medoids)

    
    # ---- 2. Bootstrap-SHAP for confidence interval ----
    print("Computing SHAP values with bootstrapping …")
    boots = []
    all_sv = []
    rng   = np.random.default_rng(0)
    
    for _ in range(n_bootstrap):
        idx     = rng.choice(test_data.shape[0], size=test_data.shape[0], replace=True)
        sv_raw  = explainer.shap_values(test_data[idx])
        sv      = sv_raw[0] if isinstance(sv_raw, list) else sv_raw
        
        all_sv.append(sv)
        boots.append(np.abs(sv).mean(0))
    
    boots = np.stack(boots, 0)                       # [B, F]
    shap_values = boots.mean(0)                      # final importance vector
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5], axis=0)
    
    # Ensure numpy array
    if isinstance(shap_values, torch.Tensor):
        shap_values = shap_values.cpu().numpy()
    
    # Get feature names
    feature_names = []  # will be filled, or fallback later
    try:
        if data_dir:          
            flat_info_path      = Path(data_dir) / "flat_info.json"
            diagnoses_info_path = Path(data_dir) / "diagnoses_info.json"
    
            if flat_info_path.exists():
                with open(flat_info_path) as f:
                    feature_names.extend(json.load(f)["columns"])          # 42
    
            if diagnoses_info_path.exists():
                with open(diagnoses_info_path) as f:
                    feature_names.extend(json.load(f)["columns"])          # 3266
    
        if len(feature_names) != test_data.shape[1]:
            raise ValueError(
                f"Expected {test_data.shape[1]} names, got {len(feature_names)}"
            )
    
    except Exception as e:
        print(f"[WARN] Unable to load real column names, using indices: {e}")
        feature_names = [f"Feature_{i}" for i in range(test_data.shape[1])]

    drop_features = ['nullheight'] 
    
    if drop_features:
        keep_idx = [i for i, n in enumerate(feature_names) if n not in drop_features]
        if len(keep_idx) != len(feature_names):

            feature_names = [feature_names[i] for i in keep_idx]
            all_sv = [sv[:, keep_idx] for sv in all_sv]

            shap_values = shap_values[keep_idx]
            ci_low      = ci_low[keep_idx]
            ci_high     = ci_high[keep_idx]

            test_data   = test_data[:, keep_idx]

            print(f"[INFO] Dropped features: {', '.join(drop_features)}")
    
    feature_pretty = [FLAT_MAP.get(n, n) for n in feature_names]
    
    # summary_plot expects matrix of [n_explain, n_features]
    plot_sv = all_sv[0]  # or np.mean(all_sv, axis=0) if needed
    if isinstance(plot_sv, torch.Tensor):
        plot_sv = plot_sv.cpu().numpy()
    
    # plot
    shap.summary_plot(plot_sv, test_data.cpu().numpy(),
                        feature_names=feature_pretty, show=False,
                      color=vs.white_to(vs.COLOR['temporal'][0]))
    plt.gcf().set_size_inches(10, 6)
    plt.subplots_adjust(right=0.85)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/beeswarm.pdf", dpi=600, bbox_inches='tight')
    plt.close()
    
    # Load flat feature list from file
    flat_field_set = set(json.load(open(Path(data_dir) / "flat_info.json"))["columns"])
    flat_idx = [i for i, name in enumerate(feature_names) if name in flat_field_set]
    diag_idx = [i for i in range(len(feature_names)) if i not in flat_idx]
    flat_set = set(flat_idx)
    
    
    # Select top 12 features by SHAP importance
    shap_importance = shap_values
    top_indices = np.argsort(-shap_importance)[:12]
    top_features = [wrap_feature_name(prettify_flat(feature_names[i])) for i in top_indices]
    top_importance = shap_importance[top_indices]
    top_low = ci_low[top_indices]
    top_high = ci_high[top_indices]
    
    
    # Assign color by feature type
    bar_colors = [
        vs.COLOR['feature'][3] if idx in flat_set else vs.COLOR['feature'][2]
        for idx in top_indices
    ]
    
    # Set plotting style
    plt.rcParams.update({
        'font.family': 'sans-serif',  # academic-style font
        'font.size': 14
    })
    
    x_pos = np.arange(len(top_features))
    
    plt.figure(figsize=(8, 4.2))
    
    bars = plt.bar(
        x_pos, top_importance,
        width=0.8,
        color=bar_colors,
        edgecolor='none',
        alpha=0.9,
        yerr=[top_importance - top_low, top_high - top_importance],
        capsize=6,
        error_kw=dict(ecolor='gray', lw=1)
    )
    

    plt.xticks(x_pos, top_features, rotation=45, ha='right', fontsize=10)
    plt.ylabel("Mean Absolute SHAP Value", labelpad=10, fontsize=11)
    plt.yticks(fontsize=10)
    plt.tick_params(axis='x', labelsize=10)
    plt.tick_params(axis='y', labelsize=10)

    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("gray")
    
    patch_flat = mpatches.Patch(color=vs.COLOR['feature'][3], label="Flat Feature")
    patch_diag = mpatches.Patch(color=vs.COLOR['feature'][2], label="Diagnosis Feature")
    plt.legend(handles=[patch_flat, patch_diag], loc='upper right', frameon=False, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/bar_top12.pdf", dpi=600, bbox_inches='tight')
    plt.close()
    
    def _plot_bar(idx_list: list, label: str):
        # Subset SHAP values and confidence intervals
        sv_sub = shap_values[idx_list]
        ci_low_sub, ci_high_sub = ci_low[idx_list], ci_high[idx_list]
    
        # Select top 20 features
        top = np.argsort(-sv_sub)[:20]
        top_feat = [wrap_feature_name(prettify_flat(feature_names[idx_list[i]])) for i in top]
        top_imp = sv_sub[top]
        top_low = ci_low_sub[top]
        top_high = ci_high_sub[top]
    
        # Set color by feature type
        color_map = {
            "flat": vs.COLOR['feature'][3],
            "diag": vs.COLOR['feature'][2],
        }
        bar_col = color_map[label]
    
        # Set academic-style font
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 11
        })
    
        y_pos = np.arange(len(top_feat)) * 1.3
    
        plt.figure(figsize=(6, 4.8))  # better aspect ratio
    
        # Create bar plot
        plt.barh(
            y_pos, top_imp,
            color=bar_col,
            alpha=0.9,
            capsize=4,
            height=0.8,
            edgecolor='none',
            error_kw=dict(ecolor='gray', lw=1)
        )
    
        # Set feature names and axis labels
        plt.yticks(y_pos, top_feat)
        plt.xlabel("Mean Absolute SHAP Value", labelpad=10)
        plt.gca().invert_yaxis()
    
        # Add gridlines for clarity
        plt.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.6)
    
        # Add border/spines
        for spine in plt.gca().spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("gray")
    
        # Final layout and export
        plt.tight_layout()
        plt.savefig(f"{save_dir}/bar_top20_{label}.pdf", dpi=600, bbox_inches="tight")
        plt.close()
    
    # Generate both flat and diagnosis plots
    _plot_bar(flat_idx, "flat")
    _plot_bar(diag_idx, "diag")

    total_abs     = np.abs(shap_values).sum()
    flat_contrib  = np.abs(shap_values[flat_idx]).sum()  / total_abs
    diag_contrib  = np.abs(shap_values[diag_idx]).sum()  / total_abs

    plt.figure(figsize=(4, 6))
    plt.bar(["Flat", "Diagnoses"], [flat_contrib, diag_contrib],
            color=["#4C72B0", "#55A868"])
    plt.ylabel("Relative |SHAP| contribution")
    plt.title("Modality contribution")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/modality_contribution.pdf", dpi=600,
                bbox_inches="tight")
    plt.close()
    
    
    # ---- 3. Hierarchical (ICD-block) summarisation ----     
    icd_grp = {}
    pattern = re.compile(r"^diag_(\w{3})")          # e.g. diag_J44_xxx → J44
    for i, name in enumerate(feature_names):
        m = pattern.match(name)
        key = m.group(1) if m else name             # non-diag keeps itself
        icd_grp.setdefault(key, []).append(i)

    grp_imp = {k: shap_importance[idxs].mean() for k, idxs in icd_grp.items()}
    grp_sorted = sorted(grp_imp.items(), key=lambda x: -x[1])[:20]
    plt.figure(figsize=(10, 6))
    bar_col = '#376b9e'
    plt.barh([g[0] for g in grp_sorted], [g[1] for g in grp_sorted], color=bar_col)
    plt.xlabel("Mean |SHAP|")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/icd_block_bar.pdf", dpi=600)
    plt.close()
    
    print(f"SHAP analysis complete. Figures saved to {save_dir}")
    
    return {
        "shap_values"  : shap_values,
        "feature_names": feature_names,
        "flat_idx"     : flat_idx,
        "diag_idx"     : diag_idx,
        "contrib"      : {"flat": float(flat_contrib),
                          "diag": float(diag_contrib)}
    }
