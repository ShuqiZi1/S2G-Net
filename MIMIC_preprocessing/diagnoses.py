import pandas as pd
import numpy as np
import json

def extract_icd_hierarchy(icd_code, icd_version):
    """
    Extract hierarchy from ICD code
    
    ICD-9 hierarchy:
    - Level 1: Main categories (e.g., 001-139: Infectious and Parasitic Diseases)
    - Level 2: Three-digit codes (e.g., 001: Cholera)
    - Level 3: Four-digit codes (e.g., 0010: Cholera due to Vibrio cholerae)
    - Level 4: Five-digit codes (e.g., 00100: Cholera due to Vibrio cholerae, biotype cholerae)
    
    ICD-10 hierarchy:
    - Level 1: Chapter (e.g., A00-B99: Certain infectious and parasitic diseases)
    - Level 2: Category (e.g., A00: Cholera)
    - Level 3: Subcategory (e.g., A00.0: Cholera due to Vibrio cholerae O1)
    - Level 4: Seven-character modifier (e.g., S72.012A: Fracture of femoral neck, incomplete, right side, initial encounter)
    """
    # Handle V and E codes
    if icd_version == '9':
        if icd_code.startswith('V'):
            # V-code processing (e.g., V27.0 -> V27 -> V)
            if len(icd_code) >= 4 and icd_code[3] == '.':
                return [icd_code[0], icd_code[:3], icd_code]
            elif len(icd_code) >= 3:
                return [icd_code[0], icd_code[:3]]
            else:
                return [icd_code[0]]
        elif icd_code.startswith('E'):
            # E-code processing (e.g., E880.0 -> E880 -> E)
            if len(icd_code) >= 5 and icd_code[4] == '.':
                return [icd_code[0], icd_code[:4], icd_code]
            elif len(icd_code) >= 4:
                return [icd_code[0], icd_code[:4]]
            else:
                return [icd_code[0]]
        else:
            # Numeric code processing
            code = icd_code.replace('.', '')
            if len(code) == 3:
                chapter = determine_icd9_chapter(code)
                return [chapter, code]
            elif len(code) == 4:
                chapter = determine_icd9_chapter(code[:3])
                return [chapter, code[:3], code]
            elif len(code) == 5:
                chapter = determine_icd9_chapter(code[:3])
                return [chapter, code[:3], code[:4], code]
            else:
                return [icd_code]  # Non-standard format code
    
    elif icd_version == '10':
        if len(icd_code) >= 1:
            chapter = icd_code[0]  # Level 1: Letter represents chapter
            if len(icd_code) >= 3:
                category = icd_code[:3]  # Level 2: Three-character category
                if len(icd_code) > 3:
                    # Has subcategory or modifier
                    sub_parts = []
                    if len(icd_code) >= 4 and (icd_code[3] == '.' or icd_code[3].isdigit()):
                        # Process subcategory (4th position)
                        sub_category = icd_code[:4].replace('.', '')
                        sub_parts.append(sub_category)
                    
                    if len(icd_code) >= 5:
                        # Process modifier (if any)
                        full_code = icd_code.replace('.', '')
                        sub_parts.append(full_code)
                    
                    return [chapter, category] + sub_parts
                else:
                    return [chapter, category]
            else:
                return [chapter]
        else:
            return [icd_code]  # Empty code or non-standard format
    
    return [icd_code]  # Default: return entire code as a single level

def determine_icd9_chapter(code_prefix):
    """Determine which chapter an ICD-9 code belongs to"""
    code_num = int(code_prefix)
    if 1 <= code_num <= 139:
        return "01-Infectious"
    elif 140 <= code_num <= 239:
        return "02-Neoplasms"
    elif 240 <= code_num <= 279:
        return "03-Endocrine"
    elif 280 <= code_num <= 289:
        return "04-Blood"
    elif 290 <= code_num <= 319:
        return "05-Mental"
    elif 320 <= code_num <= 389:
        return "06-Nervous"
    elif 390 <= code_num <= 459:
        return "07-Circulatory"
    elif 460 <= code_num <= 519:
        return "08-Respiratory"
    elif 520 <= code_num <= 579:
        return "09-Digestive"
    elif 580 <= code_num <= 629:
        return "10-Genitourinary"
    elif 630 <= code_num <= 679:
        return "11-Pregnancy"
    elif 680 <= code_num <= 709:
        return "12-Skin"
    elif 710 <= code_num <= 739:
        return "13-Musculoskeletal"
    elif 740 <= code_num <= 759:
        return "14-Congenital"
    elif 760 <= code_num <= 779:
        return "15-Perinatal"
    elif 780 <= code_num <= 799:
        return "16-Symptoms"
    elif 800 <= code_num <= 999:
        return "17-Injury"
    else:
        return "18-Other"

def add_icd_codes(uniquepid, patientunitstayid, diagnosis_codes, icd_versions, mapping_dict, codes_dict, words_dict, count):
    """
    Add ICD codes for a patient
    
    Parameters:
    - uniquepid: Unique patient ID
    - patientunitstayid: Patient unit stay ID
    - diagnosis_codes: List of diagnosis codes for the patient
    - icd_versions: Corresponding list of ICD versions
    - mapping_dict: Mapping dictionary
    - codes_dict: Codes dictionary
    - words_dict: Words dictionary
    - count: Current code count
    
    Returns:
    - List of codes for the patient and updated count
    """
    patient_codes = []
    
    # Process all diagnosis codes for this patient
    for i, (icd_code, icd_version) in enumerate(zip(diagnosis_codes, icd_versions)):
        # Extract hierarchy
        hierarchy = extract_icd_hierarchy(icd_code, icd_version)
        codes_for_this_diagnosis = []
        
        # For each level in the hierarchy
        current_path = ""
        for level, part in enumerate(hierarchy):
            if level == 0:
                current_path = part
            else:
                current_path = f"{current_path}|{part}"
                
            # Check if this path already has a code
            try:
                # If this path is already coded, use existing code
                code_id = codes_dict[current_path][0]
                codes_dict[current_path][2] += 1  # Increment count
            except KeyError:
                # If this is a new path, create new code
                codes_dict[current_path] = [count, {}, 1]  # [code_id, children_dict, count]
                code_id = count
                words_dict[count] = current_path
                count += 1
                
            codes_for_this_diagnosis.append(code_id)
            
        # Store all codes for this diagnosis
        # We only save the full path code for the diagnosis, not each level
        if len(codes_for_this_diagnosis) > 0:
            full_code = codes_for_this_diagnosis[-1]
            patient_codes.append(full_code)
            
        # Store diagnosis mapping in mapping_dict - now using uniquepid and patientunitstayid
        diagnosis_key = f"{uniquepid}_{patientunitstayid}_{icd_code}"
        mapping_dict[diagnosis_key] = codes_for_this_diagnosis
            
    return patient_codes, count

def process_diagnoses(diagnoses_df, labels_df, cut_off_prevalence=0.001):
    """
    Process diagnosis data, build hierarchy, and create sparse matrix
    
    Parameters:
    - diagnoses_df: DataFrame containing diagnoses (with subject_id, icd_code, icd_version)
    - labels_df: DataFrame containing patient mapping (uniquepid to patientunitstayid)
    - cut_off_prevalence: Minimum prevalence to retain diagnoses
    
    Returns:
    - sparse_df: DataFrame in sparse matrix form
    - words_dict: Mapping from code to diagnosis name
    """
    # Create a mapping from uniquepid to patientunitstayid
    pid_to_unitstayid = dict(zip(labels_df['uniquepid'], labels_df['patientunitstayid']))
    
    # Initialize data structures
    mapping_dict = {}  # Mapping from diagnosis to code
    codes_dict = {}    # Hierarchical code dictionary
    words_dict = {}    # Mapping from code ID to diagnosis name
    count = 0          # Current code count
    
    # Organize data by patient ID (using uniquepid and getting patientunitstayid)
    diagnoses_by_patient = {}
    
    for _, row in diagnoses_df.iterrows():
        uniquepid = row['subject_id']  # Assuming 'subject_id' is the uniquepid in diagnoses_df
        
        # Skip if patient not in labels.csv
        if uniquepid not in pid_to_unitstayid:
            continue
            
        patientunitstayid = pid_to_unitstayid[uniquepid]
        icd_code = row['icd_code']
        icd_version = str(row['icd_version'])
        
        key = (uniquepid, patientunitstayid)
        if key not in diagnoses_by_patient:
            diagnoses_by_patient[key] = {'codes': [], 'versions': []}
            
        diagnoses_by_patient[key]['codes'].append(icd_code)
        diagnoses_by_patient[key]['versions'].append(icd_version)
    
    # Process diagnoses for each patient
    patient_diagnoses = {}
    
    for (uniquepid, patientunitstayid), diags in diagnoses_by_patient.items():
        patient_codes, count = add_icd_codes(
            uniquepid,
            patientunitstayid,
            diags['codes'], 
            diags['versions'], 
            mapping_dict, 
            codes_dict, 
            words_dict, 
            count
        )
        
        # Store diagnoses by patientunitstayid (not by uniquepid)
        patient_diagnoses[patientunitstayid] = patient_codes
    
    # Create sparse matrix using patientunitstayid as index
    unit_stay_ids = list(patient_diagnoses.keys())
    num_patients = len(unit_stay_ids)
    sparse_diagnoses = np.zeros((num_patients, count))
    
    for i, unit_stay_id in enumerate(unit_stay_ids):
        codes = patient_diagnoses[unit_stay_id]
        sparse_diagnoses[i, codes] = 1  # Set diagnoses this patient has to 1
    
    # Find codes with only one child (potentially redundant)
    pointless_codes = []
    def find_pointless_codes(diag_dict):
        result = []
        for key, value in diag_dict.items():
            # If there's only one child, the branch is linear and can be compressed
            if value[2] == 1:
                result.append(value[0])
            result.extend(find_pointless_codes(value[1]))
        return result
    
    pointless_codes = find_pointless_codes(codes_dict)
    
    # Calculate prevalence for each code, remove rare codes
    sparse_df = pd.DataFrame(sparse_diagnoses, index=unit_stay_ids, columns=range(count))
    cut_off = round(cut_off_prevalence * num_patients)
    prevalence = sparse_df.sum(axis=0)
    rare_codes = prevalence.loc[prevalence <= cut_off].index
    
    # Remove rare and redundant codes
    sparse_df.drop(columns=list(rare_codes) + pointless_codes, inplace=True)
    
    # Replace column names with diagnosis names
    sparse_df.rename(columns=words_dict, inplace=True)
    
    # Rename index to patient
    sparse_df.index.name = 'patient'
    
    return sparse_df, words_dict

# Main function
def create_icd_hierarchy(diagnoses_path, labels_path, output_path, cut_off_prevalence=0.001):
    """
    Create hierarchy for ICD codes and save processed data
    
    Parameters:
    - diagnoses_path: Path to diagnoses CSV file
    - labels_path: Path to labels CSV file with uniquepid to patientunitstayid mapping
    - output_path: Path to save output file
    - cut_off_prevalence: Minimum prevalence to retain diagnoses
    """
    print('==> Loading diagnosis and labels data...')
    diagnoses = pd.read_csv(diagnoses_path)
    labels = pd.read_csv(labels_path)
    
    print('==> Building diagnosis hierarchy...')
    sparse_df, words_dict = process_diagnoses(diagnoses, labels, cut_off_prevalence)
    
    print(f'==> Keeping {sparse_df.shape[1]} diagnoses with prevalence greater than {cut_off_prevalence*100}%...')
    
    print('==> Saving processed diagnosis data...')
    sparse_df.to_csv(output_path)
    
    # Optional: Save code to diagnosis name mapping
    pd.Series(words_dict).to_csv(output_path.replace('.csv', '_mapping.csv'))
    
    return sparse_df


if __name__ == '__main__':
    with open('paths.json', 'r') as f:
        MIMIC_path = json.load(f)["MIMIC_path"]
    diagnoses_path = MIMIC_path / "diagnoses_icd.csv"
    labels_path    = MIMIC_path / "labels.csv"
    output_path    = MIMIC_path / "preprocessed_diagnoses.csv"
    
    sparse_df = create_icd_hierarchy(diagnoses_path, labels_path, output_path, cut_off_prevalence)
    print('Done!')