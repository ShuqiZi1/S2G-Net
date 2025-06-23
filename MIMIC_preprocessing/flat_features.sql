/* Useful explorative query for finding variables to include in chartevents
select count(distinct ch.stay_id), d.label, avg(valuenum) as avg_value
  from chartevents as ch
    inner join icustays as i
      on ch.stay_id = i.stay_id
    inner join d_items as d
      on d.itemid = ch.itemid
    where lower(d.label) like '%height%' -- <-- enter phrase of interest
      and date_part('hour', ch.charttime) - date_part('hour', i.intime) < 5
    group by d.label;
*/

-- requires tablefunc extension, can be obtained with 'CREATE EXTENSION tablefunc;'
drop materialized view if exists extra_vars cascade;
create materialized view extra_vars as
  select * from crosstab(
    'SELECT ch.stay_id, d.label, AVG(valuenum) AS value
        FROM mimiciv.chartevents AS ch
        INNER JOIN mimiciv.icustays AS i
            ON ch.stay_id = i.stay_id
        INNER JOIN mimiciv.d_items AS d
            ON d.itemid = ch.itemid
        WHERE ch.valuenum IS NOT NULL
          AND d.label IN (''Admission Weight (Kg)'', ''GCS - Eye Opening'', ''GCS - Motor Response'', ''GCS - Verbal Response'', ''Height (cm)'')
          AND ch.valuenum != 0
          AND DATE_PART(''hour'', ch.charttime) - DATE_PART(''hour'', i.intime) BETWEEN -24 AND 5
        GROUP BY ch.stay_id, d.label;'
       ) as ct(stay_id integer, weight double precision, eyes double precision, motor double precision, verbal double precision, height double precision);


DROP MATERIALIZED VIEW IF EXISTS ld_flat CASCADE;
CREATE MATERIALIZED VIEW ld_flat AS
  SELECT DISTINCT
    i.stay_id AS patientunitstayid,
    p.gender,
    (EXTRACT(YEAR FROM i.intime) - p.anchor_year + p.anchor_age) AS age,
    adm.race AS ethnicity,  -- Changed ethnicity to race
    i.first_careunit,
    adm.admission_location,
    adm.insurance,
    ev.height,
    ev.weight,
    EXTRACT(HOUR FROM i.intime) AS hour,
    ev.eyes,
    ev.motor,
    ev.verbal
  FROM ld_labels AS la
  INNER JOIN patients AS p ON p.subject_id = la.subject_id
  INNER JOIN icustays AS i ON i.stay_id = la.stay_id
  INNER JOIN admissions AS adm ON adm.hadm_id = la.hadm_id
  LEFT JOIN extra_vars AS ev ON ev.stay_id = la.stay_id;
