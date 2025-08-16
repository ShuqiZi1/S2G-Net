from MIMIC_preprocessing.timeseries import timeseries_main
from MIMIC_preprocessing.flat_and_labels import flat_and_labels_main
from MIMIC_preprocessing.split_train_test import split_train_test
from MIMIC_preprocessing.diagnoses import create_icd_hierarchy

import os
import json

with open('paths.json', 'r') as f:
    MIMIC_path = json.load(f)["MIMIC_path"]

if __name__=='__main__':
    print('==> Removing the stays.txt file if it exists...')
    try:
        os.remove(MIMIC_path + 'stays.txt')
    except FileNotFoundError:
        pass
    
    cut_off_prevalence = 0.001 
    
    timeseries_main(MIMIC_path, test=False)
    flat_and_labels_main(MIMIC_path)
    
    diagnoses_path = MIMIC_path + "diagnoses_icd.csv"
    labels_path = MIMIC_path + "labels.csv"
    output_path = MIMIC_path + "preprocessed_diagnoses.csv"
    create_icd_hierarchy(diagnoses_path, labels_path, output_path, cut_off_prevalence)
    
    split_train_test(MIMIC_path, is_test=False, cleanup=False)