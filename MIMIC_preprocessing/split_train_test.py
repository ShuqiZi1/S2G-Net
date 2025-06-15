"""
split_train_test.py
This script handles two main functions:
1. Full dataset splitting into train/val/test partitions including labels, flat, diagnoses, and timeseries tables.
2. Diagnoses-only splitting using pre-defined stays.txt files from existing partitions.
Usage:
  python split_dataset.py --mode full        # for full dataset split
  python split_dataset.py --mode diagnoses   # for diagnoses-only split
"""

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
import os
import argparse
import json

def create_folder(parent_path, folder):
    if not parent_path.endswith('/'):
        parent_path += '/'
    folder_path = parent_path + folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def shuffle_stays(stays, seed=9):
    return shuffle(stays, random_state=seed)

def process_table(table_name, table, stays, folder_path):
    # Only process patient IDs that exist in the table's index
    valid_stays = [stay for stay in stays if stay in table.index]
    if len(valid_stays) < len(stays):
        print(f"  - Warning: {len(stays) - len(valid_stays)} patients from stays list not found in {table_name}")
    
    table = table.loc[valid_stays].copy()
    table.to_csv('{}/{}.csv'.format(folder_path, table_name))
    print(f"  - Saved {table_name} with {len(table)} rows")
    return

def split_diagnoses_only(MIMIC_path, seed=9):
    """
    Only split the diagnoses dataset using existing train/val/test partitions
    """
    print('==> Loading diagnoses data for splitting...')
    diagnoses = pd.read_csv(MIMIC_path + 'preprocessed_diagnoses.csv')
    
    if 'patient' in diagnoses.columns:
        diagnoses.set_index('patient', inplace=True)
    else:
        diagnoses.set_index(diagnoses.columns[0], inplace=True)
        diagnoses.index.name = 'patient'
    
    diagnoses.index = diagnoses.index.astype(str)
    
    for partition_name in ['train', 'val', 'test']:
        print(f'==> Processing {partition_name} partition for diagnoses...')
        folder_path = MIMIC_path + partition_name
        stays_file = folder_path + '/stays.txt'
        
        if not os.path.exists(stays_file):
            print(f"  Warning: stays.txt not found at {stays_file}")
            continue
        
        stays = []
        with open(stays_file, 'r') as f:
            for line in f:
                stay = line.strip()
                stays.append(stay)
        
        stays_set = set(stays)
        matched_stays = [stay for stay in stays if stay in diagnoses.index]
        
        if matched_stays:
            matched_stays = shuffle_stays(matched_stays, seed=seed)
            diagnoses_subset = diagnoses.loc[matched_stays].copy()
            output_file = f'{folder_path}/diagnoses.csv'
            diagnoses_subset.to_csv(output_file)
            print(f"  Saved {len(diagnoses_subset)} diagnoses records to {output_file}")
        else:
            print(f"  No matching diagnoses found for {partition_name}")

def split_train_test(MIMIC_path, is_test=True, seed=9, cleanup=False):
    """
    Perform a complete train/test split of all datasets (diagnoses, labels, timeseries, flat)
    """
    print('==> Loading data for splitting...')
    
    # First load diagnoses table
    diagnoses = pd.read_csv(MIMIC_path + 'preprocessed_diagnoses.csv')
    diagnoses.set_index('patient', inplace=True)
    
    # Load labels table
    labels = pd.read_csv(MIMIC_path + 'preprocessed_labels.csv')
    labels.set_index('patient', inplace=True)
    
    # Only keep patients that exist in diagnoses table
    common_patients = set(labels.index) & set(diagnoses.index)
    print(f"==> Found {len(common_patients)} patients that exist in both labels and diagnoses")
    
    # Filter labels to only include patients that exist in diagnoses
    labels = labels[labels.index.isin(common_patients)]
    
    # Split based on uniquepid
    patients = labels.uniquepid.unique()
    print(f"==> Splitting {len(patients)} unique patients")
    
    train, test = train_test_split(patients, test_size=0.15, random_state=seed)
    train, val = train_test_split(train, test_size=0.15/0.85, random_state=seed)
    
    # Load other data tables
    if is_test:
        timeseries = pd.read_csv(MIMIC_path + 'preprocessed_timeseries.csv', nrows=999999)
    else:
        timeseries = pd.read_csv(MIMIC_path + 'preprocessed_timeseries.csv')
    timeseries.set_index('patient', inplace=True)
    
    flat_features = pd.read_csv(MIMIC_path + 'preprocessed_flat.csv')
    flat_features.set_index('patient', inplace=True)

    # Only delete source files if cleanup flag is set
    if cleanup:
        print("==> Cleanup flag is set. Source files will be deleted after processing.")
        # Add cleanup code here if needed

    for partition_name, partition in zip(['train', 'val', 'test'], [train, val, test]):
        print('==> Preparing {} data...'.format(partition_name))
        stays = labels.loc[labels['uniquepid'].isin(partition)].index
        folder_path = create_folder(MIMIC_path, partition_name)
        with open(folder_path + '/stays.txt', 'w') as f:
            for stay in stays:
                f.write("%s\n" % stay)
        stays = shuffle_stays(stays, seed=seed)

        for table_name, table in zip(['labels', 'flat', 'diagnoses', 'timeseries'],
                                     [labels, flat_features, diagnoses, timeseries]):
            process_table(table_name, table, stays, folder_path)

    return

if __name__=='__main__':
    with open('paths.json', 'r') as f:
        MIMIC_path = json.load(f)["MIMIC_path"]
    
    parser = argparse.ArgumentParser(description='Split MIMIC datasets into train/val/test partitions')
    parser.add_argument('--mode', type=str, choices=['full', 'diagnoses_only'], default='full',
                        help='Choose the split mode: full (all datasets) or diagnoses_only')
    parser.add_argument('--seed', type=int, default=9, help='Random seed for shuffling')
    parser.add_argument('--debug', action='store_true', help='Enable debug output (diagnoses_only mode)')
    parser.add_argument('--cleanup', action='store_true', help='Clean up source files after splitting (full mode)')
    parser.add_argument('--is_test', action='store_true', help='Run in test mode with limited timeseries rows (full mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        split_train_test(MIMIC_path, is_test=args.is_test, seed=args.seed, cleanup=args.cleanup)
    else:  # diagnoses_only
        split_diagnoses_only(MIMIC_path, seed=args.seed, debug=args.debug)