import pandas as pd
import itertools  # Add this import for islice
import json
import os
import numpy as np
import gc 

def resample_and_mask(timeseries, MIMIC_path, header, mask_decay=True, decay_rate = 4/3, test=False,
                       verbose=False):
    if verbose:
        print('Resampling to 2 hour intervals...')
    # take the mean of any duplicate index entries for unstacking
    timeseries = timeseries.groupby(level=[0, 1]).mean()
    # put patient into columns so that we can round the timedeltas to the nearest hour and take the mean in the time interval
    unstacked = timeseries.unstack(level=0)
    del (timeseries)
    unstacked.index = unstacked.index.ceil(freq='H')
    resampled = unstacked.resample('H', closed='right', label='right').mean()
    del (unstacked)

    # store which values had to be imputed
    if mask_decay:
        if verbose:
            print('Calculating mask decay features...')
        mask_bool = resampled.notnull()
        mask = mask_bool.astype(int)
        mask.replace({0: np.nan}, inplace=True)  # so that forward fill works
        inv_mask_bool = ~mask_bool
        count_non_measurements = inv_mask_bool.cumsum() - \
                                 inv_mask_bool.cumsum().where(mask_bool).ffill().fillna(0)
        mask = mask.ffill().fillna(0) / (count_non_measurements * decay_rate).replace(0, 1)
        mask = mask.iloc[-48:]
        del (mask_bool, inv_mask_bool, count_non_measurements)
    else:
        if verbose:
            print('Calculating binary mask features...')
        mask = resampled.iloc[-48:].notnull()
        mask = mask.astype(int)

    if verbose:
        print('Filling missing data forwards...')
    # carry forward missing values (note they will still be 0 in the nulls table)
    resampled = resampled.fillna(method='ffill').iloc[-48:]

    # simplify the indexes of both tables
    resampled.index = list(range(1, 49))
    mask.index = list(range(1, 49))

    if verbose:
        print('Filling in remaining values with zeros...')
    resampled.fillna(0, inplace=True)

    if verbose:
        print('Reconfiguring and combining features with mask features...')
    # pivot the table around to give the final data
    resampled = resampled.stack(level=1).swaplevel(0, 1).sort_index(level=0)
    mask = mask.stack(level=1).swaplevel(0, 1).sort_index(level=0)

    # rename the columns in pandas for the mask so it doesn't complain
    mask.columns = [str(col) + '_mask' for col in mask.columns]

    # merge the mask with the features
    final = pd.concat([resampled, mask], axis=1)

    if verbose:
        print('Saving progress...')
    # save to csv
    if test is False:
        final.to_csv(MIMIC_path + 'preprocessed_timeseries.csv', mode='a', header=header)
    return
    

def add_time_of_day(processed_timeseries, flat_features):

    print('==> Adding time of day features...')
    processed_timeseries = processed_timeseries.join(flat_features[['hour']], how='inner', on='patient')
    processed_timeseries['hour'] = processed_timeseries['time'] + processed_timeseries['hour']
    hour_list = np.linspace(0, 1, 24)  # make sure it's still scaled well
    processed_timeseries['hour'] = processed_timeseries['hour'].apply(lambda x: hour_list[x%24 - 24])
    return processed_timeseries
    
    
def further_processing(MIMIC_path, test=False):

    processed_timeseries = pd.read_csv(MIMIC_path + 'preprocessed_timeseries.csv')
    processed_timeseries.rename(columns={
        'Unnamed: 0': 'patient',
        'Unnamed: 1': 'time'
    }, inplace=True)
    processed_timeseries.set_index('patient', inplace=True)
    flat_features = pd.read_csv(MIMIC_path + 'flat_features.csv')
    flat_features.rename(columns={'patientunitstayid': 'patient'}, inplace=True)
    flat_features.set_index('patient', inplace=True)

    processed_timeseries = add_time_of_day(processed_timeseries, flat_features)

    print('==> Getting rid of time series that don\'t vary across time for at least 30% patients '
          '- these will be added to the flat features instead of time series...')
    # we want to see at least 2 mask variables per patient on average, this would be 2/48 recordings
    mask_cols = [col for col in processed_timeseries.columns if 'mask' in col]
    # we say equals 1 in case mask decay is being used
    mean_masks = processed_timeseries[mask_cols].eq(1).groupby('patient').mean().mean()
    mask_to_flat = list(mean_masks[(mean_masks <= 2 / 48)].index)
    cols_to_flat = [x[:-5] for x in mask_to_flat] + mask_to_flat  # remove '_mask'

    # keep only the most recent measurement, and it's corresponding mask value
    flat_features = flat_features.join(
        processed_timeseries.loc[processed_timeseries['time'] == 48][cols_to_flat],
        how='inner', on='patient')
    processed_timeseries.drop(cols_to_flat, axis=1, inplace=True)

    # remove RR (patient) as it has hardly any entries and we already have respiration as a feature
    processed_timeseries.drop(columns=['RR (patient)', 'RR (patient)_mask'], inplace=True, errors='ignore')

    if test is False:
        print('==> Saving flat features with non-time varying features added...')
        flat_features.to_csv(MIMIC_path + 'preprocessed_flat.csv')

        print('==> Saving finalised preprocessed timeseries...')
        # this will replace old one that was updated earlier in the script
        processed_timeseries.to_csv(MIMIC_path + 'preprocessed_timeseries.csv')

    return
    
    
def reconfigure_timeseries_chunked(timeseries, offset_column, feature_column, test=False):
    """A memory-efficient version of reconfigure_timeseries that processes data in chunks"""
    # Ensure we have appropriate indexing
    if 'patientunitstayid' not in timeseries.columns:
        raise ValueError("DataFrame must contain patientunitstayid column")

    # Get unique patient IDs
    patient_ids = timeseries['patientunitstayid'].unique()

    # Process in smaller chunks of patients
    chunk_size = 1000  # Adjust based on memory constraints
    result_dfs = []

    for i in range(0, len(patient_ids), chunk_size):
        chunk_ids = patient_ids[i:i + chunk_size]
        chunk = timeseries[timeseries['patientunitstayid'].isin(chunk_ids)].copy()

        chunk[offset_column] = pd.to_timedelta(chunk[offset_column], unit='m')  # or 'T' for minutes

        # Set the index to patient ID and offset
        chunk.set_index(['patientunitstayid', offset_column], inplace=True)

        # Process this chunk
        try:
            # For smaller chunks, we can use the original pivot approach
            pivoted = chunk.pivot_table(columns=feature_column, index=chunk.index)
            result_dfs.append(pivoted)
        except ValueError as e:
            # If still too large, process patient by patient
            patient_results = []
            for pid in chunk_ids:
                try:
                    patient_data = chunk.loc[pid]
                    if isinstance(patient_data, pd.Series):
                        # Handle case with only one row
                        continue
                    patient_pivoted = patient_data.pivot_table(columns=feature_column, index=patient_data.index)
                    patient_results.append(patient_pivoted)
                except Exception:
                    # Skip problematic patients
                    continue

            if patient_results:
                result_dfs.extend(patient_results)

    # Combine all chunks
    if result_dfs:
        return pd.concat(result_dfs, sort=False)
    else:
        return pd.DataFrame()


# Fixed implementation of gen_patient_chunk to maintain MultiIndex
def custom_gen_patient_chunk(patients, merged, size=500):
    """Generate chunks of patient data from the merged DataFrame.

    Args:
        patients: Array-like of patient IDs
        merged: DataFrame with MultiIndex (patient_id, offset)
        size: Number of patients per chunk

    Yields:
        DataFrame chunks with MultiIndex preserved
    """
    # Process patients in chunks
    for i in range(0, len(patients), size):
        chunk_patients = patients[i:i + size]
        # Select rows for these patients while preserving the MultiIndex
        chunk = merged.loc[chunk_patients]
        yield chunk


def gen_timeseries_file(MIMIC_path, test=False):
    print('==> Loading data from timeseries files...')
    if test:
        timeseries_lab = pd.read_csv(MIMIC_path + 'timeserieslab.csv', nrows=10000)
        timeseries = pd.read_csv(MIMIC_path + 'timeseries.csv', nrows=10000)
    else:
        timeseries_lab = pd.read_csv(MIMIC_path + 'timeserieslab.csv')
        timeseries = pd.read_csv(MIMIC_path + 'timeseries.csv')

    print('==> Reconfiguring lab timeseries...')
    timeseries_lab = reconfigure_timeseries_chunked(timeseries_lab,
                                                    offset_column='labresultoffset',
                                                    feature_column='labname',
                                                    test=test)
    timeseries_lab.columns = timeseries_lab.columns.droplevel()

    print('==> Reconfiguring other timeseries...')
    timeseries = reconfigure_timeseries_chunked(timeseries,
                                                offset_column='chartoffset',
                                                feature_column='chartvaluelabel',
                                                test=test)
    timeseries.columns = timeseries.columns.droplevel()

    # note that in MIMIC the timeseries are a lot messier so there are a lot of variables present that are not useful
    # drop duplicate columns which appear in chartevents
    print('==> Dropping the following columns because they have duplicates in labevents:')
    cols = []
    for col in timeseries.columns:
        if col in timeseries_lab.columns or col in timeseries_lab.columns + ' (serum)':
            cols.append(col)
    # plus some others which don't quite match up based on strings
    cols += ['WBC', 'HCO3 (serum)', 'Lactic Acid', 'PH (Arterial)', 'Arterial O2 pressure', 'Arterial CO2 Pressure',
             'Arterial Base Excess', 'TCO2 (calc) Arterial', 'Ionized Calcium', 'BUN', 'Calcium non-ionized',
             'Anion gap']
    for col in cols:
        print('\t' + col)
    timeseries.drop(columns=cols, inplace=True)

    # just take a single Braden score, the individual variables will be deleted
    timeseries['Braden Score'] = timeseries[['Braden Activity', 'Braden Friction/Shear', 'Braden Mobility',
                                             'Braden Moisture', 'Braden Nutrition', 'Braden Sensory Perception']].sum(
        axis=1)
    timeseries['Braden Score'].replace(0, np.nan, inplace=True)  # this is where it hasn't been measured

    # finally remove some binary and less useful variables from the original set
    print('==> Also removing some binary and less useful variables:')
    other = ['18 Gauge Dressing Occlusive', '18 Gauge placed in outside facility', '18 Gauge placed in the field',
             '20 Gauge Dressing Occlusive', '20 Gauge placed in outside facility', '20 Gauge placed in the field',
             'Alarms On', 'Ambulatory aid', 'CAM-ICU MS Change', 'Eye Care', 'High risk (>51) interventions',
             'History of falling (within 3 mnths)', 'IV/Saline lock', 'Mental status', 'Parameters Checked',
             'ST Segment Monitoring On', 'Secondary diagnosis', 'Acuity Workload Question 1',
             'Acuity Workload Question 2', 'Arterial Line Zero/Calibrate',
             'Arterial Line placed in outside facility', 'Cuff Pressure',
             'Gait/Transferring', 'Glucose (whole blood)', 'Goal Richmond-RAS Scale',
             'Inspiratory Time', 'Braden Activity', 'Braden Friction/Shear', 'Braden Mobility',
             'Braden Moisture', 'Braden Nutrition', 'Braden Sensory Perception',
             'Multi Lumen placed in outside facility',
             'O2 Saturation Pulseoxymetry Alarm - High', 'Orientation', 'Orientation to Person',
             'Orientation to Place', 'Orientation to Time', 'Potassium (whole blood)',
             'SpO2 Desat Limit', 'Subglottal Suctioning', 'Ventilator Tank #1', 'Ventilator Tank #2']
    for col in other:
        print('\t' + col)
    timeseries.drop(columns=other, inplace=True)

    # Get unique patients from the first level of the MultiIndex
    patients = timeseries.index.get_level_values(0).unique()

    # Set chunk size
    size = 1000

    # Create merged DataFrame before generating chunks
    merged = pd.concat([timeseries_lab, timeseries], axis=1, sort=False)

    # Verify we have a proper MultiIndex before proceeding
    if not isinstance(merged.index, pd.MultiIndex):
        print("Warning: merged DataFrame does not have a MultiIndex. This might cause issues.")

    # Calculate quantiles on the full merged dataset
    quantiles = merged.quantile([0.05, 0.95])

    # Apply normalization
    merged = 2 * (merged - quantiles.loc[0.05]) / (quantiles.loc[0.95] - quantiles.loc[0.05]) - 1

    # Clip outliers
    merged.clip(lower=-4, upper=4, inplace=True)

    # Use our custom implementation to ensure MultiIndex is preserved
    gen_chunks = custom_gen_patient_chunk(patients, merged, size=size)

    i = size
    header = True  # for the first chunk include the header in the csv file

    print('==> Starting main processing loop...')

    for patient_chunk in gen_chunks:
        # Verify the chunk has a proper MultiIndex
        if not isinstance(patient_chunk.index, pd.MultiIndex):
            print("Error: chunk does not have a MultiIndex!")
            # Try to fix it by resetting and setting the index again
            temp_df = patient_chunk.reset_index()
            if 'level_0' in temp_df.columns and 'level_1' in temp_df.columns:
                patient_chunk = temp_df.set_index(['level_0', 'level_1'])
            else:
                print("Cannot fix MultiIndex, skipping chunk")
                continue

        try:
            resample_and_mask(patient_chunk, MIMIC_path, header, mask_decay=True, decay_rate=4 / 3, test=test,
                              verbose=False)
            print('==> Processed ' + str(i) + ' patients...')
        except ValueError as e:
            print(f"Error processing chunk: {str(e)}")
            # Print more information to help debug
            print(f"Chunk index type: {type(patient_chunk.index)}")
            print(f"Chunk columns: {patient_chunk.columns}")
            # Try a different approach - process each patient separately
            if isinstance(patient_chunk.index, pd.MultiIndex):
                chunk_patients = patient_chunk.index.get_level_values(0).unique()
                for pid in chunk_patients:
                    try:
                        # Extract single patient data
                        single_patient = patient_chunk.loc[pid]
                        # Process just this patient
                        resample_and_mask(single_patient, MIMIC_path, header, mask_decay=True, decay_rate=4 / 3,
                                          test=test, verbose=False)
                        print(f"Processed patient {pid}")
                    except Exception as e2:
                        print(f"Failed to process patient {pid}: {str(e2)}")

        i += size
        header = False

    return


def timeseries_main(MIMIC_path, test=False):
    # make sure the preprocessed_timeseries.csv file is not there because the first section of this script appends to it
    if test is False:
        print('==> Removing the preprocessed_timeseries.csv file if it exists...')
        try:
            os.remove(MIMIC_path + 'preprocessed_timeseries.csv')
        except FileNotFoundError:
            pass
    gen_timeseries_file(MIMIC_path, test)
    further_processing(MIMIC_path)
    return


if __name__ == '__main__':
    with open('paths.json', 'r') as f:
        MIMIC_path = json.load(f)["MIMIC_path"]
    test = False
    timeseries_main(MIMIC_path, test)