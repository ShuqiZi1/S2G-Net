# S²G-Net MIMIC-IV Data Preprocessing

This repository contains the data preprocessing pipeline for S2G-Net (Mamba-GPS) applied to the MIMIC-IV dataset. The preprocessing transforms raw MIMIC-IV data into structured formats suitable for length-of-stay prediction and mortality prediction tasks.

## Overview

The preprocessing pipeline extracts and processes:
- **Labels**: ICU mortality and length-of-stay outcomes
- **Static Features**: Demographics, admission details, and baseline measurements
- **Time Series**: Vital signs, laboratory values, and clinical measurements
- **Diagnoses**: ICD-9/10 codes with hierarchical encoding

## Requirements

### Dependencies
```bash
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=0.24.0
psycopg2-binary>=2.8.0  # for PostgreSQL connection
```

### Database Setup
1. **MIMIC-IV Access**: Obtain access to MIMIC-IV dataset from PhysioNet: https://physionet.org/content/mimiciv/3.1/

2. **Database Installation Options**:
   
   **Option A: BigQuery (Recommended for small-scale use)**
   - Follow instructions: https://mimic-iv.mit.edu/docs/access/bigquery/
   - Note: Free tier has 1GB download limit; full preprocessing requires ~4.5GB
   - Modify table references (e.g., `patients` → `physionet-data.mimic_core.patients`)
   
   **Option B: Local PostgreSQL (Recommended for full preprocessing)**
   - Install PostgreSQL and create MIMIC-IV database
   - Use setup scripts from: https://github.com/EmmaRocheteau/MIMIC-IV-Postgres
   - For detailed instructions, see: https://mimic.physionet.org/tutorials/install-mimic-locally-ubuntu/ (adapt for MIMIC-IV)

## Configuration

### 1. Path Setup
Edit `paths.json` to specify your data directory:
```json
{
    "MIMIC_path": "/path/to/your/MIMIC_data/"
}
```

### 2. SQL Script Configuration
In `MIMIC_preprocessing/create_all_tables.sql`, update paths using find-and-replace:
- Find: `/MIMIC_preprocessing/`
- Replace: `/your/full/path/to/MIMIC_preprocessing/`
- Find: `/MIMIC_data/`
- Replace: `/your/full/path/to/MIMIC_data/`

**Important**: Keep trailing slashes in all paths.

## Usage

### Step 1: Database Data Extraction
Connect to your MIMIC-IV PostgreSQL database and run the SQL extraction scripts:

```bash
# Connect to database
psql 'dbname=mimic user=mimicuser options=--search_path=mimiciv'
```

In the psql console:
```sql
\i MIMIC_preprocessing/create_all_tables.sql
```

**Expected Duration**: 1-2 hours depending on system performance.

To exit psql:
```sql
\q
```

### Step 2: Data Preprocessing
Run the complete preprocessing pipeline:

```bash
python -m MIMIC_preprocessing.run_all_preprocessing
```

**Expected Duration**: 8-12 hours for full dataset processing.

### Alternative: Step-by-Step Processing
For debugging or partial processing:

```bash
# Process time series data
python -m MIMIC_preprocessing.timeseries

# Process static features and labels
python -m MIMIC_preprocessing.flat_and_labels

# Process diagnosis codes
python -m MIMIC_preprocessing.diagnoses

# Split into train/validation/test sets
python -m MIMIC_preprocessing.split_train_test
```

## Output Structure

The preprocessing generates the following directory structure:

```
MIMIC_data/
├── train/
│   ├── diagnoses.csv      # Training set diagnoses (hierarchical ICD codes)
│   ├── flat.csv          # Training set static features
│   ├── labels.csv        # Training set labels (mortality, LOS)
│   ├── stays.txt         # Training set patient IDs
│   └── timeseries.csv    # Training set time series data
├── val/
│   ├── diagnoses.csv      # Validation set (same structure as train)
│   ├── flat.csv
│   ├── labels.csv
│   ├── stays.txt
│   └── timeseries.csv
├── test/
│   ├── diagnoses.csv      # Test set (same structure as train)
│   ├── flat.csv
│   ├── labels.csv
│   ├── stays.txt
│   └── timeseries.csv
├── preprocessed_diagnoses.csv     # Full processed diagnoses
├── preprocessed_flat.csv          # Full processed static features
├── preprocessed_labels.csv        # Full processed labels
├── preprocessed_timeseries.csv    # Full processed time series
├── flat_features.csv              # Raw extracted static features
├── labels.csv                     # Raw extracted labels
├── timeseries.csv                 # Raw extracted chart events
└── timeserieslab.csv              # Raw extracted lab events
```

## Data Description

### Dataset Overview
- **Total Samples**: 65,347 ICU stays
- **Train/Validation/Test Split**: 45,742 / 9,802 / 9,803 samples
- **Split Strategy**: Patient-level stratification to prevent data leakage

### Labels (`labels.csv`)
- `patient`: Patient unit stay ID
- `uniquepid`: Unique patient identifier
- `patienthealthsystemstayid`: Hospital admission ID
- `actualhospitalmortality`: Hospital mortality (0/1)
- `actualiculos`: ICU length of stay (days)

### Static Features (`flat.csv`)
- **Total Features**: 42 features
- **Demographics**: age, gender, ethnicity (one-hot encoded with 10 categories)
- **Admission Details**: first care unit (8 categories), admission location (7 categories), insurance (5 categories) 
- **Baseline Measurements**: height, weight, Glasgow Coma Scale scores (eyes, motor, verbal)
- **Temporal Features**: admission hour
- **Missing Data Indicators**: nullheight flag
- **Encoding**: Categorical variables are one-hot encoded with rare categories grouped as "misc"

### Time Series (`timeseries.csv`)
- **Frequency**: Hourly measurements over 48-hour windows
- **Features**: Chart-based measurements and laboratory values (feature count varies by preprocessing parameters)
- **Format**: Patient × Time × Features with forward-fill imputation
- **Masking**: Binary masks indicating measurement availability

### Diagnoses (`diagnoses.csv`)
- **Encoding**: Hierarchical ICD-9/10 codes (3,266 features)
- **Structure**: Multi-level diagnosis categories with hierarchical encoding
- **Examples**: "09-Digestive|572|5723", "16-Symptoms|789|7895|78959"
- **Filtering**: Minimum prevalence threshold (0.1% by default)

## Data Filtering and Cohort Selection

### Inclusion Criteria
- **Age**: ≥18 years at ICU admission
- **ICU Stay**: ≥5 hours duration
- **Data Availability**: Complete time series, static features, and labels
- **Time Window**: Diagnoses within 24 hours of ICU admission

### Feature Selection
- **Time Series**: Features present in ≥25% of patients with sufficient measurements per patient on average
- **Static Features**: 42 processed features including demographics, admission details, and baseline measurements
- **Diagnoses**: 3,266 hierarchical ICD codes with prevalence ≥0.1% across the cohort

## Preprocessing Details

### Time Series Processing
1. **Resampling**: Irregular measurements → hourly intervals
2. **Normalization**: 5th-95th percentile scaling with outlier clipping
3. **Imputation**: Forward-fill with decay-based masking
4. **Windowing**: 48-hour observation windows

### Diagnosis Processing
1. **Hierarchy Construction**: Multi-level ICD code organization (Chapter|Category|Subcategory format)
2. **Prevalence Filtering**: Remove rare diagnoses (resulting in 3,266 features)
3. **Sparse Encoding**: Binary indicator matrices for each hierarchical diagnosis code

### Train/Validation/Test Split
- **Strategy**: Patient-level stratification
- **Actual Distribution**: 70.0% train (45,742), 15.0% validation (9,802), 15.0% test (9,803)
- **Seed**: Fixed for reproducibility (seed=9)

## Troubleshooting

### Common Issues

**Memory Errors**: 
- Large datasets may require ≥32GB RAM
- Consider processing in smaller chunks or using a high-memory system

**Database Connection Issues**:
- Verify PostgreSQL service is running
- Check database credentials and schema access
- Ensure `mimiciv` schema is properly set up

**Missing Files**:
- Verify all SQL scripts completed successfully
- Check file permissions in output directory
- Ensure sufficient disk space (~10GB for full dataset)

**Python Module Errors**:
- Run from project root directory
- Verify all dependencies are installed
- Check Python path configuration

## Performance Notes

- **SQL Extraction**: ~1-2 hours (database-dependent)
- **Python Processing**: ~8-12 hours (system-dependent)
- **Memory Usage**: Peak ~16-32GB RAM
- **Storage**: ~10GB total output data

## License

This code is released under the same license as the MIMIC-IV dataset. Users must obtain appropriate data use agreements before accessing MIMIC-IV data.

## Contact

For technical issues or questions about the preprocessing pipeline, please open an issue in this repository.