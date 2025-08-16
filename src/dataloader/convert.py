import numpy as np
import pandas as pd
from pathlib import Path
from src.utils import write_json, write_pkl, load_json


def filter_and_convert_files(data_dir, save_dir, n_rows=100000):
    """
    Filter all CSV files to only include patients in diagnoses.csv and convert to mmap.
    """
    # First, collect all patient IDs from diagnoses.csv across all splits
    diagnoses_patients = set()
    for split in ['train', 'val', 'test']:
        csv_path = Path(data_dir) / split / 'diagnoses.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            diagnoses_patients.update(df[df.columns[0]].tolist())
    
    print(f"Found {len(diagnoses_patients)} unique patients in diagnoses files")
    
    # Process each file type, filtering to only include diagnoses patients
    for csv_name in ['diagnoses', 'flat', 'labels']:
        print(f"** Converting {csv_name} **")
        
        # Create temporary filtered CSV files
        for split in ['train', 'val', 'test']:
            original_path = Path(data_dir) / split / f'{csv_name}.csv'
            filtered_path = Path(data_dir) / split / f'{csv_name}_filtered.csv'
            
            if original_path.exists():
                df = pd.read_csv(original_path)
                patient_col = df.columns[0]
                filtered_df = df[df[patient_col].isin(diagnoses_patients)]
                
                print(f"{split}/{csv_name}: Original patients: {len(df)}, Filtered patients: {len(filtered_df)}")
                filtered_df.to_csv(filtered_path, index=False)
            else:
                print(f"Warning: {original_path} doesn't exist")
        
        # Convert the filtered files
        convert_filtered_into_mmap(data_dir, save_dir, csv_name, n_rows)
        
        # Clean up temporary files
        for split in ['train', 'val', 'test']:
            filtered_path = Path(data_dir) / split / f'{csv_name}_filtered.csv'
            if filtered_path.exists():
                filtered_path.unlink()
    
    # Handle timeseries separately due to its 3D structure
    print("** Converting time series **")
    convert_timeseries_filtered_into_mmap(data_dir, save_dir, diagnoses_patients, n_rows)


def convert_filtered_into_mmap(data_dir, save_dir, csv_name, n_rows=100000):
    """
    Convert filtered CSV files to mmap.
    """
    # First, check the actual number of columns in the first split
    first_csv_path = Path(data_dir) / 'train' / f'{csv_name}_filtered.csv'
    first_df = pd.read_csv(first_csv_path)
    actual_cols = first_df.shape[1] - 1  # Subtract 1 for patient column
    
    n_cols = actual_cols
    print(f"Using {n_cols} columns for {csv_name}.csv (actual column count)")
        
    shape = (n_rows, n_cols)

    save_path = Path(save_dir) / f'{csv_name}.dat'
    write_file = np.memmap(save_path, dtype=np.float32, mode='w+', shape=shape)

    info = {'name': csv_name, 'shape': shape}

    n = 0

    for split in ['train', 'val', 'test']:
        print('split: ', split)
        csv_path = Path(data_dir) / split / f'{csv_name}_filtered.csv'
        df = pd.read_csv(csv_path)
        
        # Apply log1p transformation to LOS column if this is labels data
        if csv_name == 'labels':
            # Find the LOS column
            los_column = None
            for col in df.columns[1:]:  # Skip patient ID column
                if 'los' in col.lower():
                    los_column = col
                    print(f"Applying log1p transformation to LOS column '{los_column}' in labels")
                    # Apply log1p transformation
                    df[los_column] = np.log1p(df[los_column])
                    break
            
            if los_column is None:
                print("Warning: No LOS column found, no log1p transformation applied")
        
        arr = df.values[:, 1:]  # cut out patient column
        arr_len = len(arr)
        write_file[n : n+arr_len, :] = arr  # write into mmap
        info[split + '_len'] = arr_len
        n += arr_len
        del arr
    
    # Adjust final size of the mmap if needed
    if n < n_rows:
        final_shape = (n, n_cols)
        info['shape'] = final_shape
        write_file.flush()
        # Create a new mmap with the correct size and copy the data
        final_file = np.memmap(save_path, dtype=np.float32, mode='r+', shape=final_shape)
        final_file[:] = write_file[:n]
        write_file = final_file
    
    info['total'] = n
    info['columns'] = list(df)[1:]
    
    # Record the transformation in info for labels data
    if csv_name == 'labels':
        # Find LOS column index
        los_column_idx = None
        los_column_name = None
        for i, col in enumerate(info['columns']):
            if 'los' in col.lower():
                los_column_idx = i
                los_column_name = col
                break
        
        if los_column_idx is not None:
            # Add transformation info to JSON
            if 'transformations' not in info:
                info['transformations'] = {}
            
            info['transformations'][los_column_name] = 'log1p'
            print(f"Recorded in info: LOS column '{los_column_name}' (index {los_column_idx}) has been log1p transformed")

    write_json(info, Path(save_dir) / f'{csv_name}_info.json')
    print(info)


def convert_timeseries_filtered_into_mmap(data_dir, save_dir, filtered_patients, n_rows=100000, max_timesteps=48):
    """
    Filter time series data to only include specified patients and convert to mmap.
    """
    save_path = Path(save_dir) / 'ts.dat'
    example_csv = pd.read_csv(Path(data_dir) / 'train' / 'timeseries.csv')
    n_features = example_csv.shape[1] - 1  # remove patient ID
    shape = (n_rows, max_timesteps, n_features)
    write_file = np.memmap(save_path, dtype=np.float32, mode='w+', shape=shape)
    ids = []
    n = 0
    info = {}
    info['name'] = 'ts'
    
    for split in ['train', 'val', 'test']:
        print('split: ', split)
        csv_path = Path(data_dir) / split / 'timeseries.csv'
        
        if not csv_path.exists():
            print(f"Warning: {csv_path} doesn't exist")
            continue
            
        df = pd.read_csv(csv_path)
        patient_col = df.columns[0]
        
        # Filter to only include patients in the diagnoses
        filtered_df = df[df[patient_col].isin(filtered_patients)]
        print(f"{split}/timeseries: Original records: {len(df)}, Filtered records: {len(filtered_df)}")
        
        # Group by patient ID
        patient_groups = filtered_df.groupby(patient_col)
        print(f"Number of patients in {split}: {len(patient_groups)}")
        
        split_patients = 0
        
        # Process each patient
        for patient_id, patient_df in patient_groups:
            # Get patient data (without ID column)
            patient_data = patient_df.iloc[:, 1:].values
            timesteps = len(patient_data)
            
            if timesteps > max_timesteps:
                # Truncate if too many timesteps
                patient_data = patient_data[:max_timesteps, :]
                actual_timesteps = max_timesteps
            else:
                # Pad with zeros if fewer timesteps
                padding = np.zeros((max_timesteps - timesteps, patient_data.shape[1]))
                patient_data = np.vstack([patient_data, padding])
                actual_timesteps = timesteps
                
            # Write to mmap
            write_file[n, :, :] = patient_data
            ids.append(patient_id)
            n += 1
            split_patients += 1
            
            # Expand mmap if needed
            if n >= write_file.shape[0]:
                old_shape = write_file.shape
                new_shape = (old_shape[0] + n_rows, old_shape[1], old_shape[2])
                write_file.flush()
                write_file = np.memmap(save_path, dtype=np.float32, mode='r+', shape=new_shape)
        
        info[split + '_len'] = split_patients
    
    # Resize mmap to actual used size
    if n < write_file.shape[0]:
        final_shape = (n, max_timesteps, n_features)
        write_file.flush()
        final_file = np.memmap(save_path, dtype=np.float32, mode='r+', shape=final_shape)
        final_file[:] = write_file[:n]
        write_file = final_file
        
    info['total'] = n
    info['shape'] = final_shape
    info['max_timesteps'] = max_timesteps
    info['columns'] = list(df)[1:]
    
    ids = np.array(ids)
    id2pos = {pid: pos for pos, pid in enumerate(ids)}
    pos2id = {pos: pid for pos, pid in enumerate(ids)}
    
    assert len(set(ids)) == len(ids)

    print('saving..')
    write_pkl(id2pos, Path(save_dir) / 'id2pos.pkl')
    write_pkl(pos2id, Path(save_dir) / 'pos2id.pkl')
    write_json(info, Path(save_dir) / 'ts_info.json')
    print(info)


def read_mm(datadir, name):
    """
    name can be one of {ts, diagnoses, labels, flat}.
    """
    info = load_json(Path(datadir) / (name + '_info.json'))
    dat_path = Path(datadir) / (name + '.dat')
    data = np.memmap(dat_path, dtype=np.float32, shape=tuple(info['shape']))
    return data, info


if __name__ == '__main__':
    paths = load_json('paths.json')
    data_dir = paths['MIMIC_path']
    save_dir = paths['data_dir']
    print(f'Load MIMIC processed data from {data_dir}')
    print(f'Saving mmap data in {save_dir}')
    print('--'*30)
    Path(save_dir).mkdir(exist_ok=True)
    
    # Use the new filtering and conversion function
    filter_and_convert_files(data_dir, save_dir)
    
    print('--'*30)
    print(f'Done! Saved data in {save_dir}')