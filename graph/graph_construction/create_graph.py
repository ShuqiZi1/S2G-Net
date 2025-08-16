import numpy as np
from scipy import sparse
import pandas as pd
import torch
import argparse
import json
import os
import torch


def get_device_and_dtype():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    dtype = torch.cuda.sparse.ByteTensor if device.type == 'cuda' else torch.sparse.ByteTensor
    return device, dtype


def get_freqs(train_diagnoses):
    return train_diagnoses.sum()


def score_matrix(diagnoses, freq_adjustment=None, debug=False, batch_size=500):
    print('==> Making score matrix')
    diagnoses = np.array(diagnoses).astype(np.uint8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert to float32 for computation
    diagnoses = torch.tensor(diagnoses, device=device, dtype=torch.float32)

    if debug:
        diagnoses = diagnoses[:1000]

    print('==> Finding common diagnoses')

    if freq_adjustment is not None:
        freq_adjustment = torch.tensor(1 / freq_adjustment, device=device, dtype=torch.float32) * 1000 + 1
        diagnoses = diagnoses * freq_adjustment.unsqueeze(0)

    num_rows = diagnoses.shape[0]

    # Create a PyTorch tensor for scores
    scores = torch.zeros((num_rows, num_rows), device=device, dtype=torch.float32)

    print(f"Processing in batches of {batch_size} to avoid OOM errors...")

    for i in range(0, num_rows, batch_size):
        end_i = min(i + batch_size, num_rows)
        print(f"Processing batch {i} to {end_i}...")

        # Compute batch result
        batch_result = torch.matmul(diagnoses[i:end_i], diagnoses.T)
        
        # Store batch in the scores tensor
        scores[i:end_i, :] = batch_result
        
        del batch_result
        torch.cuda.empty_cache()

    print("✅ Score matrix completed.")
    return scores


def make_graph_penalise(diagnoses, scores_path, batch_size=1000, debug=False, k=3, mode='k_closest', save_edge_values=True):
    print('==> Getting edges')
    
    # Store the total number of patients BEFORE filtering
    total_patients = len(diagnoses)
    print(f"Total patients (including those with no diagnoses): {total_patients}")
    
    # Make sure we're working with a proper scores tensor
    if isinstance(scores_path, str):
        print(f"Loading scores from path: {scores_path}")
        try:
            # Calculate scores from scratch since loading isn't working
            print("Creating scores matrix from diagnoses...")
            scores = torch.zeros((len(diagnoses), len(diagnoses)), dtype=torch.float32)
            
            # Convert diagnoses to tensor for matrix multiplication
            diagnoses_array = np.array(diagnoses).astype(np.float32)
            diagnoses_tensor = torch.tensor(diagnoses_array)
            
            print(f"Diagnoses tensor shape: {diagnoses_tensor.shape}")
            
            # Calculate scores in batches
            for i in range(0, len(diagnoses), batch_size):
                end_i = min(i + batch_size, len(diagnoses))
                print(f"Processing batch {i} to {end_i}...")
                batch_scores = torch.matmul(diagnoses_tensor[i:end_i], diagnoses_tensor.T)
                scores[i:end_i] = batch_scores
            
            print(f"Successfully created scores matrix with shape: {scores.shape}")
        except Exception as e:
            print(f"Error creating scores matrix: {e}")
            raise
    else:
        scores = scores_path
    
    if debug:
        diagnoses = diagnoses[:1000]
        scores = scores[:1000, :1000]
    
    no_pts = len(diagnoses)
    print(f"Number of patients: {no_pts}")
    
    diags_per_pt = diagnoses.sum(axis=1)
    diags_per_pt = torch.tensor(diags_per_pt.values).type(torch.ShortTensor)
    
    # Track patients with no diagnoses
    patients_with_no_diags = (diags_per_pt == 0).nonzero(as_tuple=True)[0].cpu().numpy()
    print(f"Patients with no diagnoses: {len(patients_with_no_diags)}")
    
    del diagnoses
    
    if save_edge_values:
        edges_val = sparse.lil_matrix((no_pts, no_pts), dtype=np.int16)
    edges = sparse.lil_matrix((no_pts, no_pts), dtype=np.uint8)
    
    print(f"Setting diagonal of scores to zero")
    scores.fill_diagonal_(0)  # remove self scores on diagonal
    
    # Process in smaller batches to avoid memory issues
    batch_size = min(batch_size, 100)
    
    down = torch.split(diags_per_pt.repeat(no_pts, 1), batch_size, dim=0)
    across = torch.split(diags_per_pt.repeat(no_pts, 1).permute(1, 0), batch_size, dim=0)
    score_batches = torch.split(scores, batch_size, dim=0)
    
    prev_pts = 0
    for i, (d, a, s) in enumerate(zip(down, across, score_batches)):
        print(f'==> Processing batch {i+1}/{len(down)}, patients {prev_pts} to {prev_pts+len(d)}')
        total_combined_diags = d + a
        s_pen = 5 * s - total_combined_diags  # the 5 is fairly arbitrary but I don't want to penalise not sharing diagnoses too much
        
        if mode == 'k_closest':
            k_ = k
        else:
            k_ = 1  # make sure there is at least one edge for each node in the threshold graph
        
        for patient in range(len(d)):
            patient_idx = patient + prev_pts
            
            # For patients with no diagnoses, still try to find neighbors
            # They might connect to other patients with no diagnoses
            if patient_idx in patients_with_no_diags:
                # Find k patients with lowest diagnosis count (likely other zero-diagnosis patients)
                k_lowest_diag_indices = torch.argsort(diags_per_pt)[:k_]
                for j in k_lowest_diag_indices:
                    if j != patient_idx:  # Avoid self-loops
                        edges[patient_idx, j] = 1
                        if save_edge_values:
                            edges_val[patient_idx, j] = 1  # Small positive value
            else:
                # Original logic for patients with diagnoses
                k_highest_inds = torch.sort(s_pen[patient].flatten()).indices[-k_:]
                if save_edge_values:
                    k_highest_vals = torch.sort(s_pen[patient].flatten()).values[-k_:]
                    for j, val in zip(k_highest_inds, k_highest_vals):
                        if val == 0:  # these get removed if val is 0
                            val = 1
                        edges_val[patient_idx, j] = val
                for j in k_highest_inds:
                    edges[patient_idx, j] = 1
        
        prev_pts += len(d)
        
        if mode == 'threshold':
            scores_lower = torch.tril(s_pen, diagonal=-1)
            if i == 0:  # define threshold
                desired_no_edges = k * len(s_pen)
                threshold_value = torch.sort(scores_lower.flatten()).values[-desired_no_edges]
            
            for batch in torch.split(scores_lower, min(100, len(scores_lower)), dim=0):
                batch[batch < threshold_value] = 0
            
            batch_start = batch_size * i
            batch_end = batch_start + len(scores_lower)
            edges[batch_start:batch_end] = edges[batch_start:batch_end] + sparse.lil_matrix(scores_lower)
    
    del scores, d, a, s, total_combined_diags, s_pen
    
    print("Creating symmetric adjacency matrix")
    # make it symmetric again
    edges = edges + edges.transpose()
    if save_edge_values:
        edges_val = edges_val + edges_val.transpose()
        print("Processing edge values")
        for i, (edge, edge_val) in enumerate(zip(edges, edges_val)):
            if i % 1000 == 0:
                print(f"Processed {i}/{no_pts} rows")
            if len(edge.indices) > 0:
                edges_val[i, edge.indices] = edge_val.data // edge.data
        edges = edges_val
    
    print("Removing self edges and zeros")
    edges.setdiag(0)  # remove any left over self edges from patients without any diagnoses
    edges.eliminate_zeros()
    
    # do upper triangle again and then save
    print("Calculating final edge lists")
    edges = sparse.tril(edges, k=-1)
    v, u, vals = sparse.find(edges)
    
    print(f"Found {len(u)} edges")
    print(f"Final graph will have {total_patients} nodes (preserving all patients)")
    
    return u, v, vals, k


def make_graph(scores_path, threshold=True, k_closest=False, k=3):
    print('==> Getting edges')
    
    # Check if scores is a string (filepath) and load it
    if isinstance(scores_path, str):
        print(f"Creating scores matrix from scratch...")
        # The code to create a new scores matrix would go here
        # But for now, let's raise an error so we know to implement this
        raise NotImplementedError("Creating scores from scratch not implemented in make_graph function")
    else:
        scores = scores_path
    
    no_pts = len(scores)
    if k_closest:
        k_ = k
    else:
        k_ = 1 # ensure there is at least one edge per node in the threshold graph
    
    edges = sparse.lil_matrix((no_pts, no_pts), dtype=np.uint8)
    scores.fill_diagonal_(0)  # get rid of self connection scores
    
    for patient in range(no_pts):
        k_highest = torch.sort(scores[patient].flatten()).indices[-k_:]
        for i in k_highest:
            edges[patient, i] = 1
    
    del scores
    edges = edges + edges.transpose()  # make it symmetric again
    
    # do upper triangle again and then save
    edges = sparse.tril(edges, k=-1)
    
    if threshold:
        scores_lower = torch.tril(scores, diagonal=-1)
        del scores
        desired_no_edges = k * no_pts
        threshold_value = torch.sort(scores_lower.flatten()).values[-desired_no_edges]
        
        for batch in torch.split(scores_lower, 100, dim=0):
            batch[batch < threshold_value] = 0
        
        edges = edges + sparse.lil_matrix(scores_lower)
        del scores_lower
    
    v, u, _ = sparse.find(edges)
    return u, v, k


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--mode', type=str, default='k_closest', help='k_closest or threshold')
    parser.add_argument('--freq_adjust', action='store_true')
    parser.add_argument('--penalise_non_shared', action='store_true')
    parser.add_argument('--debug', action='store_true', help='Run with small subset of data')
    args = parser.parse_args()

    print(args)

    with open('paths.json', 'r') as f:
        data = json.load(f)
        MIMIC_path = data["MIMIC_path"]
        graph_dir = data["graph_dir"]
        print(graph_dir)

    device, dtype = get_device_and_dtype()
    adjust = '_adjusted' if args.freq_adjust else ''

    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)

    # get score matrix path
    scores_path = '{}scores{}.pt'.format(graph_dir, adjust)
    
    # Read diagnosis data
    print("Reading diagnoses data...")
    train_diagnoses = pd.read_csv('{}train/diagnoses.csv'.format(MIMIC_path), index_col='patient')
    val_diagnoses = pd.read_csv('{}val/diagnoses.csv'.format(MIMIC_path), index_col='patient')
    test_diagnoses = pd.read_csv('{}test/diagnoses.csv'.format(MIMIC_path), index_col='patient')
    all_diagnoses = pd.concat([train_diagnoses, val_diagnoses, test_diagnoses], sort=False)
    
    # Count total patients BEFORE filtering
    total_patients_before = len(all_diagnoses)
    print(f"Total patients before filtering: {total_patients_before}")
    
    # Instead of dropping patients with no diagnoses, keep them but mark them
    patients_with_diagnoses = all_diagnoses.sum(axis=1) > 0
    print(f"Patients with at least one diagnosis: {patients_with_diagnoses.sum()}")
    print(f"Patients with no diagnoses: {(~patients_with_diagnoses).sum()}")
    
    # Keep all patients
    print(f"Shape of all_diagnoses: {all_diagnoses.shape}")
    
    if args.debug:
        all_diagnoses = all_diagnoses.iloc[:1000]
        print(f"Debug mode: using {len(all_diagnoses)} samples")
    
    # Calculate frequency adjustment if needed
    if args.freq_adjust:
        freq_adjustment = get_freqs(train_diagnoses)
    else:
        freq_adjustment = None
    
    del train_diagnoses, val_diagnoses, test_diagnoses
    
    # make graph
    if args.penalise_non_shared:
        adjust = '_adjusted_ns'
        print(f"Creating graph with penalized non-shared diagnoses, k={args.k}, mode={args.mode}")
        u, v, vals, k = make_graph_penalise(all_diagnoses, scores_path, debug=args.debug, k=args.k, mode=args.mode)
    else:
        if args.mode == 'threshold':
            u, v, k = make_graph(scores_path, threshold=True, k_closest=False, k=args.k)
        else:
            u, v, k = make_graph(scores_path, threshold=False, k_closest=True, k=args.k)
    
    # Save results
    print(f"Saving results to {graph_dir}")
    np.savetxt('{}{}_u_k={}{}.txt'.format(graph_dir, args.mode, k, adjust), u.astype(int), fmt='%i')
    np.savetxt('{}{}_v_k={}{}.txt'.format(graph_dir, args.mode, k, adjust), v.astype(int), fmt='%i')
    if args.penalise_non_shared:
        np.savetxt('{}{}_scores_k={}{}.txt'.format(graph_dir, args.mode, k, adjust), vals.astype(int), fmt='%i')
    
    # Save node count information
    node_info = {
        'total_nodes': total_patients_before,
        'nodes_with_diagnoses': int(patients_with_diagnoses.sum()),
        'nodes_without_diagnoses': int((~patients_with_diagnoses).sum()),
        'edges': len(u)
    }
    with open(f'{graph_dir}{args.mode}_k={k}{adjust}_info.json', 'w') as f:
        json.dump(node_info, f, indent=2)
    
    print(f"Graph info saved to {graph_dir}{args.mode}_k={k}{adjust}_info.json")
    print("Done!")