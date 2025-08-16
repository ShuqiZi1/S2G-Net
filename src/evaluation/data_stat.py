import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Publication‑quality style for all figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.transparent': True,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.6,
})

def _format_axes(ax):
    # Keep light grey grid and remove top/right spines
    ax.grid(color='lightgrey', linewidth=0.6, linestyle='-')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def load_all_stays_ids(data_path):
    """Load patient IDs from all stays.txt files"""
    splits = ['train', 'val', 'test']
    all_patient_ids = []
    
    for split in splits:
        stays_path = os.path.join(data_path, split, 'stays.txt')
        if os.path.exists(stays_path):
            stays_df = pd.read_csv(stays_path, header=None)
            # Get the first column (patient IDs)
            patient_ids = stays_df.iloc[:, 0].tolist()
            all_patient_ids.extend(patient_ids)
            print(f"Loaded {len(patient_ids)} patients from {split}/stays.txt")
        else:
            print(f"Warning: {stays_path} not found")
    
    # Remove duplicates and return
    unique_patient_ids = list(set(all_patient_ids))
    print(f"Total unique patients across all splits: {len(unique_patient_ids)}")
    return unique_patient_ids

def load_original_regression_features(data_path, patient_ids):
    """Load original regression features from MIMIC raw files"""
    print("==> Attempting to load original regression features from MIMIC raw files")
    
    # Define the regression features we want to extract
    target_features = ['age', 'height', 'weight', 'hour', 'eyes', 'motor', 'verbal']
    
    filename = 'flat_features.csv'
    original_data = None
    
    filepath = os.path.join(data_path, filename)
    if os.path.exists(filepath):
        print(f"  Found {filename}, checking for regression features...")
        try:
            df = pd.read_csv(filepath)
            
            # Find patient ID column
            patient_col = None
            if 'patientunitstayid' in df.columns:
                patient_col = 'patientunitstayid'
            elif 'patient' in df.columns:
                patient_col = 'patient'
            elif 'icustayid' in df.columns:
                patient_col = 'icustayid'
            
            if patient_col:
                print(f"    Found patient ID column: {patient_col}")
                
                # Check which target features are available
                available_features = []
                for feature in target_features:
                    if feature in df.columns:
                        available_features.append(feature)
                        print(f"    Found feature: {feature}")
                
                if available_features:
                    # Filter by patient IDs and get feature data
                    filtered_df = df[df[patient_col].isin(patient_ids)]
                    
                    if len(filtered_df) > 0:
                        # Select patient ID and available features
                        cols_to_select = [patient_col] + available_features
                        original_data = filtered_df[cols_to_select].copy()
                        original_data = original_data.rename(columns={patient_col: 'patient'})
                        
                        print(f"    Successfully loaded {len(original_data)} records with {len(available_features)} features")
                        
                        # Print summary statistics for each feature
                        for feature in available_features:
                            feature_data = original_data[feature].dropna()
                            if len(feature_data) > 0:
                                print(f"    {feature}: mean={feature_data.mean():.2f}, range=[{feature_data.min():.2f}, {feature_data.max():.2f}]")
                        
                        return original_data, available_features
                else:
                    print("    No target regression features found in the file")
            else:
                print("    No recognized patient ID column found")
                        
        except Exception as e:
            print(f"    Error reading {filename}: {e}")
    else:
        print(f"  {filename} not found")
    
    if original_data is None:
        print("  No original regression features found in raw MIMIC files")
        print("  Will use processed data for analysis")
    
    return original_data, []

def load_original_age_data(data_path, patient_ids):
    """Load original age data from MIMIC raw files (wrapper for backward compatibility)"""
    original_data, available_features = load_original_regression_features(data_path, patient_ids)
    
    if original_data is not None and 'age' in available_features:
        age_data = original_data[['patient', 'age']].copy()
        return age_data, 'age'
    else:
        return None, None

def load_combined_data(data_path, filename, patient_ids):
    """Load and combine data from all splits for a given filename"""
    splits = ['train', 'val', 'test']
    combined_df = []
    
    for split in splits:
        file_path = os.path.join(data_path, split, filename)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Filter by patient IDs
            df_filtered = df[df['patient'].isin(patient_ids)]
            if len(df_filtered) > 0:
                combined_df.append(df_filtered)
                print(f"  {split}: {len(df_filtered)} records")
    
    if combined_df:
        return pd.concat(combined_df, ignore_index=True)
    else:
        return None

def inspect_demographics_and_outcomes_combined(data_path, patient_ids, out_dir):
    """Analyze combined demographics and outcomes from all splits"""
    print("==> Combined flat.csv and labels.csv from all splits")
    
    # Load demographics data
    df_demo = load_combined_data(data_path, 'flat.csv', patient_ids)
    # Load outcomes data
    df_labels = load_combined_data(data_path, 'labels.csv', patient_ids)
    
    if df_demo is None and df_labels is None:
        print("No demographic or outcome data found")
        return
    
    if df_demo is not None:
        print("Demographics shape:", df_demo.shape)
        print(df_demo.describe(include='all').T)
        print("\nMissing value rates:")
        print(df_demo.isnull().mean().sort_values(ascending=False).head(10))

    # Process gender - assuming it's encoded as 0/1
    if df_demo is not None and 'gender' in df_demo.columns:
        df_gender = df_demo['gender'].dropna()
        gender_counts = df_gender.value_counts()
        gender_pct = gender_counts.div(len(df_gender)).mul(100).sort_values()
        gender_labels = {0: 'Female', 1: 'Male'}  # Adjust if encoding is different
        gender_pct.index = gender_pct.index.map(gender_labels)
    
    # Process ethnicity - one-hot encoded columns
    if df_demo is not None:
        ethnicity_cols = [col for col in df_demo.columns if col.startswith('ethnicity_')]
        if ethnicity_cols:
            eth_counts = {}
            for col in ethnicity_cols:
                count = df_demo[col].sum()
                if count > 0:
                    eth_name = col.replace('ethnicity_', '').replace('_', ' ')
                    eth_counts[eth_name] = count
            eth_pct = pd.Series(eth_counts).div(len(df_demo)).mul(100).head(8).sort_values()
    
    # Process age - prioritize original age data over processed data
    age_display_mode = "normalized"  # Default
    age = None
    age_xlabel = "Age (standardized)"
    age_title = "(a) Age Distribution (Standardized)"
    
    # Try to load original regression features first (includes age)
    original_regression_data, available_features = load_original_regression_features(data_path, patient_ids)
    
    if original_regression_data is not None and 'age' in available_features:
        # Use original age data (in years)
        age = original_regression_data['age'].dropna()
        age_display_mode = "years"
        age_xlabel = "Age (years)"
        age_title = "(a) Age Distribution"
        print("Using original age data for visualization")
    elif df_demo is not None and 'age' in df_demo.columns:
        # Fall back to processed age data
        age = df_demo['age'].dropna()
        # Check if this looks like normalized data (range roughly -2 to 2)
        if age.min() >= -3 and age.max() <= 3:
            age_display_mode = "normalized"
            age_xlabel = "Age (standardized)"
            age_title = "(a) Age Distribution (Standardized)"
        else:
            # This looks like actual age in years
            age_display_mode = "years"
            age_xlabel = "Age (years)"
            age_title = "(a) Age Distribution"
        print("Using processed age data for visualization")
    
    # Process LOS
    if df_labels is not None and 'actualiculos' in df_labels.columns:
        # Convert from days to hours by multiplying by 24
        df_labels['actualiculos_hours'] = df_labels['actualiculos'] * 24
        print("\nLoS summary (converted to hours):")
        print(df_labels['actualiculos_hours'].describe())
        los = df_labels['actualiculos_hours'].dropna()
    
    # Create combined demographics and outcomes figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    
    # (a) Age Distribution - Top Left
    if age is not None and len(age) > 0:
        ax = axes[0, 0]
        sns.histplot(age, bins=30, kde=True, ax=ax,
                     edgecolor='black', linewidth=0.2, alpha=0.6)
        m, md = age.mean(), age.median()
        
        if age_display_mode == "normalized":
            ax.axvline(m, color='C1', linewidth=1, label=f"Mean {m:.2f}")
            ax.axvline(md, color='C2', linestyle='--', linewidth=1, label=f"Median {md:.2f}")
        else:
            ax.axvline(m, color='C1', linewidth=1, label=f"Mean {m:.0f}y")
            ax.axvline(md, color='C2', linestyle='--', linewidth=1, label=f"Median {md:.0f}y")
            
        ax.set_title(age_title)
        ax.set_xlabel(age_xlabel)
        ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=9, loc='upper right')
        _format_axes(ax)

    # (b) Gender Distribution - Top Right
    if df_demo is not None and 'gender' in df_demo.columns and len(df_gender) > 0:
        ax = axes[0, 1]
        ax.barh(gender_pct.index, gender_pct.values,
                color=['C0','C3'], edgecolor='black', height=0.6, linewidth=0.3, alpha=0.8)
        ax.set_title("(b) Gender Distribution")
        ax.set_xlabel("% of Cohort")
        ax.set_xlim(0, gender_pct.max() * 1.1)
        for i, v in enumerate(gender_pct.values):
            ax.text(v + 1, i, f"{v:.1f}%", va='center', fontsize=9)
        _format_axes(ax)

    # (c) Ethnicity Distribution - Bottom Left
    if df_demo is not None and ethnicity_cols and len(eth_pct) > 0:
        def fold_label(label, maxlen=12):
            return '\n'.join([label[i:i+maxlen] for i in range(0, len(label), maxlen)])
        eth_pct.index = [fold_label(str(lbl), maxlen=12) for lbl in eth_pct.index]
    
        ax = axes[1, 0]
        ax.barh(eth_pct.index, eth_pct.values,
                color='C4', edgecolor='black', height=0.6, linewidth=0.3, alpha=0.8)
        ax.set_title("(c) Ethnicity Distribution")
        ax.set_xlabel("% of Cohort")
        ax.set_xlim(0, eth_pct.max() * 1.1)
        for i, v in enumerate(eth_pct.values):
            ax.text(v + 0.5, i, f"{v:.1f}%", va='center', fontsize=9)
        _format_axes(ax)
    
    # (d) ICU Length of Stay - Bottom Right
    if df_labels is not None and 'actualiculos' in df_labels.columns and len(los) > 0:
        ax = axes[1, 1]
        sns.histplot(los, bins=50, kde=True, ax=ax,
                     edgecolor='black', linewidth=0.3, alpha=0.6)
        m, md = los.mean(), los.median()
        ax.axvline(m, color='C1', linewidth=1, label=f"Mean {m:.1f}h")
        ax.axvline(md, color='C2', linestyle='--', linewidth=1, label=f"Median {md:.1f}h")
        ax.set_title("(d) ICU Length of Stay")
        ax.set_xlabel("Hours")
        ax.set_ylabel("Density / Count")
        ax.legend(frameon=False, fontsize=9, loc='upper right')
        _format_axes(ax)
    
    # Adjust layout and save
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"demographics_and_outcomes_combined.pdf"))
    plt.close(fig)

def inspect_labels_combined(data_path, patient_ids, out_dir):
    """Analyze combined labels from all splits (without mortality)"""
    print("\n==> Combined labels.csv from all splits")
    df = load_combined_data(data_path, 'labels.csv', patient_ids)
    
    if df is None:
        print("No labels.csv data found")
        return
    
    print("Shape:", df.shape)
    
    if 'actualiculos' in df.columns:
        # Convert from days to hours by multiplying by 24
        df['actualiculos_hours'] = df['actualiculos'] * 24
        
        print("LoS summary (converted to hours):")
        print(df['actualiculos_hours'].describe())
        
        los = df['actualiculos_hours'].dropna()
        if len(los) > 0:
            fig, ax = plt.subplots(figsize=(4, 3))
            # Add KDE and annotate mean/median
            sns.histplot(los, bins=50, kde=True, ax=ax,
                         edgecolor='black', linewidth=0.3, alpha=0.6)
            m, md = los.mean(), los.median()
            ax.axvline(m, color='C1', linewidth=1, label=f"Mean {m:.1f}h")
            ax.axvline(md, color='C2', linestyle='--', linewidth=1, label=f"Median {md:.1f}h")
            ax.set_title("ICU Length of Stay")
            ax.set_xlabel("Hours")
            ax.set_ylabel("Density / Count")
            ax.legend(frameon=False, fontsize=9, loc='upper right')
            _format_axes(ax)
            
            fig.savefig(os.path.join(out_dir, f"icu_los_distribution.pdf"))
            plt.close(fig)

def inspect_diagnoses_combined(data_path, patient_ids, out_dir):
    """Analyze combined diagnoses from all splits"""
    print("\n==> Combined diagnoses.csv from all splits")
    df = load_combined_data(data_path, 'diagnoses.csv', patient_ids)
    
    if df is None:
        print("No diagnoses.csv data found")
        return
    
    print("Total rows:", len(df))
    print("Unique patients:", df['patient'].nunique())
    
    # Count diagnosis categories (exclude patient column)
    diag_cols = [col for col in df.columns if col != 'patient']
    diag_counts = {}
    
    for col in diag_cols:
        if col in df.columns:
            count = df[col].sum()
            if count > 0:
                diag_counts[col] = count
    
    if diag_counts:
        # Get top 10 diagnoses
        sorted_diags = sorted(diag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top10_names = [item[0] for item in sorted_diags]
        top10_counts = [item[1] for item in sorted_diags]
        
        print(f"Top 10 diagnosis categories:")
        for name, count in sorted_diags:
            print(f"  {name}: {count}")
        
        fig, ax = plt.subplots(figsize=(4, 3))
        y_pos = range(len(top10_names))
        ax.barh(y_pos, top10_counts, 
                edgecolor='black', linewidth=0.3, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([name[:40] + '...' if len(name) > 40 else name for name in top10_names])
        ax.set_title("Top 10 Diagnosis Categories")
        ax.set_xlabel("Count")
        ax.set_ylabel("Diagnosis Category")
        _format_axes(ax)
        
        fig.savefig(os.path.join(out_dir, f"diagnosis_top10.pdf"))
        plt.close(fig)

def inspect_timeseries_combined(data_path, patient_ids, out_dir):
    """Analyze combined timeseries from all splits"""
    print("\n==> Combined timeseries.csv from all splits")
    ts = load_combined_data(data_path, 'timeseries.csv', patient_ids)

    if ts is None:
        print("No timeseries.csv data found")
        return

    print("Rows:", ts.shape[0])
    print("Unique patients:", ts['patient'].nunique())
    
    # Check available columns for vital signs
    vital_cols = [col for col in ts.columns if not col.endswith('_mask') 
                  and col not in ['patient', 'time', 'hour']]
    print("Available vital signs:", vital_cols)

    # Time coverage per patient - show actual distribution
    if 'time' in ts.columns:
        ts['time'] = pd.to_numeric(ts['time'], errors='coerce')
        tdur = ts.groupby('patient')['time'].agg(['min', 'max', 'count'])
        # Calculate hours from time points 
        tdur['hrs'] = (tdur['max'] - tdur['min']) + 1  # +1 to include both endpoints
        
        dur = tdur['hrs'].dropna()
        dur = dur[dur > 0]
        
        # Check if data is already standardized (all 48h)
        unique_durations = dur.unique()
        is_standardized = len(unique_durations) == 1 and unique_durations[0] == 48
        
        if is_standardized:
            print("✓ All patients have standardized 48-hour observation windows")
            print("  (Skipping time coverage plot - no variation to show)")
        else:
            # Show the distribution only if there's meaningful variation
            if len(dur) > 0 and dur.std() > 0:
                fig, ax = plt.subplots(figsize=(4, 3))
                
                # Use log scale if there's a wide range of values
                if dur.max() / dur.min() > 10:
                    sns.histplot(dur, bins=30, kde=True, log_scale=(True, False), ax=ax,
                                 edgecolor='black', linewidth=0.3, alpha=0.6)
                    xlabel = "Hours (log scale)"
                    title = "Time Coverage Distribution (Log Scale)"
                else:
                    sns.histplot(dur, bins=30, kde=True, ax=ax,
                                 edgecolor='black', linewidth=0.3, alpha=0.6)
                    xlabel = "Hours"
                    title = "Time Coverage Distribution"
                
                m, md = dur.mean(), dur.median()
                ax.axvline(m, color='C1', linewidth=1, label=f"Mean {m:.1f}h")
                ax.axvline(md, color='C2', linestyle='--', linewidth=1, label=f"Median {md:.1f}h")
                ax.set_title(title)
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Density / Count")
                ax.legend(frameon=False, fontsize=9, loc='upper right')
                _format_axes(ax)
                fig.savefig(os.path.join(out_dir, f"time_coverage.pdf"))
                plt.close(fig)
            else:
                print("  (No meaningful variation in time coverage to plot)")

def generate_regression_features_table(data_path, patient_ids, out_dir):
    """Generate comprehensive regression features table comparing original vs processed data"""
    print("\n==> Generating Regression Features Comparison Table")
    
    # Load original regression features
    original_data, available_original_features = load_original_regression_features(data_path, patient_ids)
    
    # Load processed features from flat.csv
    processed_data = load_combined_data(data_path, 'flat.csv', patient_ids)
    
    # Target regression features
    target_features = ['age', 'height', 'weight', 'hour', 'eyes', 'motor', 'verbal']
    
    # Create comparison table
    comparison_results = []
    
    for feature in target_features:
        result = {'Feature': feature}
        
        # Check original data
        if original_data is not None and feature in available_original_features:
            original_values = original_data[feature].dropna()
            if len(original_values) > 0:
                result['Original_Count'] = len(original_values)
                result['Original_Mean'] = f"{original_values.mean():.2f}"
                result['Original_Std'] = f"{original_values.std():.2f}"
                result['Original_Min'] = f"{original_values.min():.2f}"
                result['Original_Max'] = f"{original_values.max():.2f}"
                result['Original_Available'] = 'Yes'
            else:
                result['Original_Available'] = 'No Data'
        else:
            result['Original_Available'] = 'Not Found'
        
        # Check processed data
        if processed_data is not None and feature in processed_data.columns:
            processed_values = processed_data[feature].dropna()
            if len(processed_values) > 0:
                result['Processed_Count'] = len(processed_values)
                result['Processed_Mean'] = f"{processed_values.mean():.3f}"
                result['Processed_Std'] = f"{processed_values.std():.3f}"
                result['Processed_Min'] = f"{processed_values.min():.3f}"
                result['Processed_Max'] = f"{processed_values.max():.3f}"
                result['Processed_Available'] = 'Yes'
            else:
                result['Processed_Available'] = 'No Data'
        else:
            result['Processed_Available'] = 'Not Found'
        
        comparison_results.append(result)
    
    # Convert to DataFrame for easy display and saving
    comparison_df = pd.DataFrame(comparison_results)
    
    # Fill NaN values with appropriate placeholders
    comparison_df = comparison_df.fillna('N/A')
    
    print("\n" + "="*100)
    print("REGRESSION FEATURES COMPARISON TABLE")
    print("="*100)
    print(comparison_df.to_string(index=False))
    
    # Save detailed table to CSV
    table_path = os.path.join(out_dir, 'regression_features_comparison.csv')
    comparison_df.to_csv(table_path, index=False)
    print(f"\nDetailed comparison table saved to: {table_path}")
    
    # Create summary table for available original features
    if original_data is not None and available_original_features:
        print(f"\n{'='*60}")
        print("ORIGINAL FEATURES SUMMARY")
        print(f"{'='*60}")
        
        summary_data = []
        for feature in available_original_features:
            feature_values = original_data[feature].dropna()
            if len(feature_values) > 0:
                summary_data.append({
                    'Feature': feature,
                    'Count': len(feature_values),
                    'Mean': f"{feature_values.mean():.2f}",
                    'Median': f"{feature_values.median():.2f}",
                    'Std': f"{feature_values.std():.2f}",
                    'Min': f"{feature_values.min():.2f}",
                    'Max': f"{feature_values.max():.2f}",
                    'Missing_Rate': f"{(1 - len(feature_values)/len(original_data))*100:.1f}%"
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            print(summary_df.to_string(index=False))
            
            # Save original features summary
            summary_path = os.path.join(out_dir, 'original_regression_features_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"\nOriginal features summary saved to: {summary_path}")
            
            # Save the original data itself
            original_data_path = os.path.join(out_dir, 'original_regression_features_data.csv')
            original_data.to_csv(original_data_path, index=False)
            print(f"Original regression features data saved to: {original_data_path}")
    
    return comparison_df, original_data

def generate_summary_report(data_path, patient_ids):
    """Generate comprehensive summary report"""
    print(f"\n{'='*60}")
    print("COMPREHENSIVE SUMMARY REPORT")
    print(f"{'='*60}")
    
    # Overall statistics
    splits = ['train', 'val', 'test']
    split_counts = {}
    
    for split in splits:
        stays_path = os.path.join(data_path, split, 'stays.txt')
        if os.path.exists(stays_path):
            stays_df = pd.read_csv(stays_path)
            split_counts[split] = len(stays_df)
    
    print(f"Total unique patients: {len(patient_ids)}")
    print(f"Split distribution:")
    for split, count in split_counts.items():
        print(f"  {split}: {count} patients")
    
    # Demographics summary
    df_flat = load_combined_data(data_path, 'flat.csv', patient_ids)
    original_regression_data, available_features = load_original_regression_features(data_path, patient_ids)
    
    if df_flat is not None:
        print(f"\nDemographics Summary:")
        print(f"  Patients with demographic data: {len(df_flat)}")
        
        # Age information - show both original and processed if available
        if original_regression_data is not None and 'age' in available_features:
            original_age = original_regression_data['age'].dropna()
            print(f"  Age statistics (original, years):")
            print(f"    Mean: {original_age.mean():.1f}")
            print(f"    Median: {original_age.median():.1f}")
            print(f"    Range: {original_age.min():.0f} - {original_age.max():.0f}")
            
            if 'age' in df_flat.columns:
                processed_age = df_flat['age'].dropna()
                print(f"  Age statistics (processed, standardized):")
                print(f"    Mean: {processed_age.mean():.3f}")
                print(f"    Median: {processed_age.median():.3f}")
                print(f"    Range: {processed_age.min():.3f} - {processed_age.max():.3f}")
        else:
            if 'age' in df_flat.columns:
                age_stats = df_flat['age'].describe()
                print(f"  Age statistics (processed, standardized):")
                print(f"    Mean: {age_stats['mean']:.3f}")
                print(f"    Median: {age_stats['50%']:.3f}")
                print(f"    Range: {age_stats['min']:.3f} - {age_stats['max']:.3f}")
        
        # Show available original regression features
        if original_regression_data is not None and available_features:
            print(f"\n  Original regression features available: {len(available_features)}")
            for feature in available_features:
                feature_data = original_regression_data[feature].dropna()
                if len(feature_data) > 0:
                    print(f"    {feature}: mean={feature_data.mean():.2f}, range=[{feature_data.min():.2f}, {feature_data.max():.2f}]")
        
        if 'gender' in df_flat.columns:
            gender_counts = df_flat['gender'].value_counts()
            print(f"  Gender distribution:")
            for gender, count in gender_counts.items():
                label = 'Male' if gender == 1 else 'Female'
                pct = count / len(df_flat) * 100
                print(f"    {label}: {count} ({pct:.1f}%)")
    
    # Clinical outcomes summary
    df_labels = load_combined_data(data_path, 'labels.csv', patient_ids)
    if df_labels is not None:
        print(f"\nClinical Outcomes Summary:")
        print(f"  Patients with outcome data: {len(df_labels)}")
        
        if 'actualiculos' in df_labels.columns:
            # Convert to hours for summary
            los_hours = df_labels['actualiculos'] * 24
            los_stats = los_hours.describe()
            print(f"  ICU Length of Stay (hours):")
            print(f"    Mean: {los_stats['mean']:.1f}")
            print(f"    Median: {los_stats['50%']:.1f}")
            print(f"    Range: {los_stats['min']:.1f} - {los_stats['max']:.1f}")
    
    # Diagnosis summary
    df_diag = load_combined_data(data_path, 'diagnoses.csv', patient_ids)
    if df_diag is not None:
        print(f"\nDiagnosis Summary:")
        print(f"  Total diagnosis records: {len(df_diag)}")
        print(f"  Patients with diagnoses: {df_diag['patient'].nunique()}")
    
    # Timeseries summary
    df_ts = load_combined_data(data_path, 'timeseries.csv', patient_ids)
    if df_ts is not None:
        print(f"\nTimeseries Summary:")
        print(f"  Total timeseries records: {len(df_ts)}")
        print(f"  Patients with timeseries: {df_ts['patient'].nunique()}")
        print(f"  Standard observation window: 48 hours")
        
        vital_cols = [col for col in df_ts.columns if not col.endswith('_mask') 
                      and col not in ['patient', 'time', 'hour']]
        print(f"  Available vital signs: {len(vital_cols)}")
        for i, col in enumerate(vital_cols[:5]):  # Show first 5
            non_null = df_ts[col].notna().sum()
            pct = non_null / len(df_ts) * 100
            print(f"    {col}: {pct:.1f}% non-null")
        if len(vital_cols) > 5:
            print(f"    ... and {len(vital_cols)-5} more")

def main():
    """Main function to analyze combined data from all splits"""
    with open('paths.json') as f:
        data_path = json.load(f)["MIMIC_path"]
    
    out_dir = os.path.join(data_path, 'analysis_figures')
    os.makedirs(out_dir, exist_ok=True)
    
    print("=== MIMIC Combined Data Analysis ===")
    
    # Load all patient IDs from all splits
    patient_ids = load_all_stays_ids(data_path)
    
    if not patient_ids:
        print("No patient IDs found in any split")
        return
    
    print(f"\n{'='*50}")
    print("ANALYZING COMBINED DATA FROM ALL SPLITS")
    print(f"{'='*50}")
    
    # Analyze each data type with combined data
    inspect_demographics_and_outcomes_combined(data_path, patient_ids, out_dir)
    inspect_diagnoses_combined(data_path, patient_ids, out_dir)
    inspect_timeseries_combined(data_path, patient_ids, out_dir)
    
    # Generate regression features comparison table
    generate_regression_features_table(data_path, patient_ids, out_dir)
    
    # Generate comprehensive summary
    generate_summary_report(data_path, patient_ids)
    
    print(f"\nAll combined analysis figures and regression features tables saved to: {out_dir}")

if __name__ == '__main__':
    main()