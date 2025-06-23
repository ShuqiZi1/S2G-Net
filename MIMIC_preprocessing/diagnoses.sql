-- Ensure correct schema
SET search_path TO mimiciv, public;

-- Drop the old materialized view if it exists
DROP MATERIALIZED VIEW IF EXISTS mimic_iv_diagnoses CASCADE;

-- Create a new materialized view
CREATE MATERIALIZED VIEW mimic_iv_diagnoses AS

-- Step 1: Extract hospital admission diagnoses (ICD codes)
SELECT di.subject_id, di.hadm_id, di.icd_code, di.icd_version
FROM mimiciv.diagnoses_icd di
INNER JOIN mimiciv.admissions adm ON di.hadm_id = adm.hadm_id
INNER JOIN mimiciv.icustays icu ON adm.hadm_id = icu.hadm_id
WHERE adm.admittime <= icu.intime + INTERVAL '24 hours'

UNION

-- Step 2: Extract ICU diagnoses (ICD codes recorded within first 24 hours)
SELECT di.subject_id, di.hadm_id, di.icd_code, di.icd_version
FROM mimiciv.diagnoses_icd di
INNER JOIN mimiciv.admissions adm ON di.hadm_id = adm.hadm_id
INNER JOIN mimiciv.icustays icu ON adm.hadm_id = icu.hadm_id
WHERE di.seq_num = 1  -- Primary ICU diagnosis
AND adm.admittime <= icu.intime + INTERVAL '24 hours';
