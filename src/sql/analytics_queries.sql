-- ====================================================================
-- PREDICTIVE MAINTENANCE ANALYTICS — SQL QUERY LIBRARY (50+ Queries)
-- ====================================================================
-- Target DB: PostgreSQL / DuckDB (ANSI SQL where possible)
-- Dataset: Microsoft Azure Predictive Maintenance (100 machines, 2015)
-- Tables:
--   pdm_telemetry   — hourly sensor readings (876,099 rows)
--   pdm_errors      — error events during operation
--   pdm_maint       — maintenance records (planned + reactive)
--   pdm_failures    — component failure/replacement events
--   pdm_machines    — machine metadata (model, age)
-- ====================================================================


-- ====================================================================
-- SECTION 1: DATA EXPLORATION & QUALITY (Queries 1-8)
-- ====================================================================

-- Query 1: Basic dataset statistics — row counts per table
SELECT 'telemetry' AS table_name, COUNT(*) AS row_count FROM pdm_telemetry
UNION ALL
SELECT 'errors', COUNT(*) FROM pdm_errors
UNION ALL
SELECT 'maintenance', COUNT(*) FROM pdm_maint
UNION ALL
SELECT 'failures', COUNT(*) FROM pdm_failures
UNION ALL
SELECT 'machines', COUNT(*) FROM pdm_machines
ORDER BY row_count DESC;

-- Query 2: Date range and coverage of telemetry data
SELECT
    MIN(datetime) AS first_reading,
    MAX(datetime) AS last_reading,
    COUNT(DISTINCT DATE_TRUNC('day', datetime)) AS days_covered,
    COUNT(DISTINCT machineID) AS machines_with_data,
    COUNT(*) AS total_readings,
    COUNT(*) / COUNT(DISTINCT machineID) AS avg_readings_per_machine
FROM pdm_telemetry;

-- Query 3: Machines with missing telemetry hours (data quality check)
WITH expected_hours AS (
    SELECT machineID, COUNT(*) AS expected_readings
    FROM (
        SELECT DISTINCT machineID,
               GENERATE_SERIES(MIN(datetime)::date, MAX(datetime)::date, '1 hour'::interval) AS hour_slot
        FROM pdm_telemetry
        GROUP BY machineID
    ) sub
    GROUP BY machineID
),
actual_hours AS (
    SELECT machineID, COUNT(*) AS actual_readings
    FROM pdm_telemetry
    GROUP BY machineID
)
SELECT
    e.machineID,
    e.expected_readings,
    a.actual_readings,
    e.expected_readings - a.actual_readings AS missing_readings,
    ROUND(100.0 * (e.expected_readings - a.actual_readings) / e.expected_readings, 2) AS pct_missing
FROM expected_hours e
JOIN actual_hours a ON e.machineID = a.machineID
WHERE e.expected_readings != a.actual_readings
ORDER BY missing_readings DESC
LIMIT 20;

-- Query 4: Distribution of machine ages
SELECT
    CASE
        WHEN age = 0 THEN 'New (0 years)'
        WHEN age BETWEEN 1 AND 5 THEN '1-5 years'
        WHEN age BETWEEN 6 AND 10 THEN '6-10 years'
        WHEN age BETWEEN 11 AND 15 THEN '11-15 years'
        ELSE '15+ years'
    END AS age_group,
    COUNT(*) AS machine_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
FROM pdm_machines
GROUP BY age_group
ORDER BY MIN(age);

-- Query 5: Sensor value ranges and basic statistics
SELECT
    'volt' AS sensor,
    MIN(volt) AS min_val, MAX(volt) AS max_val,
    AVG(volt) AS avg_val, STDDEV(volt) AS std_val
FROM pdm_telemetry
UNION ALL
SELECT 'rotate', MIN(rotate), MAX(rotate), AVG(rotate), STDDEV(rotate) FROM pdm_telemetry
UNION ALL
SELECT 'pressure', MIN(pressure), MAX(pressure), AVG(pressure), STDDEV(pressure) FROM pdm_telemetry
UNION ALL
SELECT 'vibration', MIN(vibration), MAX(vibration), AVG(vibration), STDDEV(vibration) FROM pdm_telemetry;

-- Query 6: Machine model distribution
SELECT
    model,
    COUNT(*) AS machine_count,
    ROUND(AVG(age), 1) AS avg_age_years
FROM pdm_machines
GROUP BY model
ORDER BY machine_count DESC;

-- Query 7: Count unique machines per model
SELECT
    model,
    COUNT(DISTINCT machineID) AS machine_count
FROM pdm_machines
GROUP BY model
ORDER BY machine_count DESC;

-- Query 8: Sensor readings with outlier detection (3-sigma rule)
WITH stats AS (
    SELECT
        AVG(volt) AS volt_mean, STDDEV(volt) AS volt_std,
        AVG(rotate) AS rot_mean, STDDEV(rotate) AS rot_std,
        AVG(pressure) AS pres_mean, STDDEV(pressure) AS pres_std,
        AVG(vibration) AS vib_mean, STDDEV(vibration) AS vib_std
    FROM pdm_telemetry
)
SELECT
    machineID,
    datetime,
    CASE WHEN ABS(volt - s.volt_mean) > 3 * s.volt_std THEN 'VOLTAGE_OUTLIER' END AS volt_flag,
    CASE WHEN ABS(rotate - s.rot_mean) > 3 * s.rot_std THEN 'ROTATION_OUTLIER' END AS rot_flag,
    CASE WHEN ABS(pressure - s.pres_mean) > 3 * s.pres_std THEN 'PRESSURE_OUTLIER' END AS pres_flag,
    CASE WHEN ABS(vibration - s.vib_mean) > 3 * s.vib_std THEN 'VIBRATION_OUTLIER' END AS vib_flag
FROM pdm_telemetry, stats s
WHERE ABS(volt - s.volt_mean) > 3 * s.volt_std
   OR ABS(rotate - s.rot_mean) > 3 * s.rot_std
   OR ABS(pressure - s.pres_mean) > 3 * s.pres_std
   OR ABS(vibration - s.vib_mean) > 3 * s.vib_std
ORDER BY datetime DESC
LIMIT 100;


-- ====================================================================
-- SECTION 2: ASSET HEALTH MONITORING (Queries 9-20)
-- ====================================================================

-- Query 9: Current health score per machine (latest readings + z-score)
WITH machine_stats AS (
    SELECT
        machineID,
        AVG(volt) AS avg_volt, STDDEV(volt) AS std_volt,
        AVG(vibration) AS avg_vib, STDDEV(vibration) AS std_vib
    FROM pdm_telemetry
    WHERE datetime >= NOW() - INTERVAL '30 days'
    GROUP BY machineID
),
latest AS (
    SELECT DISTINCT ON (machineID)
        machineID, datetime, volt, vibration
    FROM pdm_telemetry
    WHERE datetime >= NOW() - INTERVAL '7 days'
    ORDER BY machineID, datetime DESC
)
SELECT
    l.machineID,
    l.datetime AS last_reading,
    ROUND(l.volt, 2) AS current_voltage,
    ROUND(l.vibration, 2) AS current_vibration,
    ROUND(ABS(l.volt - s.avg_volt) / NULLIF(s.std_volt, 0), 2) AS volt_zscore,
    ROUND(ABS(l.vibration - s.avg_vib) / NULLIF(s.std_vib, 0), 2) AS vib_zscore,
    CASE
        WHEN ABS(l.volt - s.avg_volt) / NULLIF(s.std_volt, 0) > 3
          OR ABS(l.vibration - s.avg_vib) / NULLIF(s.std_vib, 0) > 3
        THEN 'CRITICAL'
        WHEN ABS(l.volt - s.avg_volt) / NULLIF(s.std_volt, 0) > 2
          OR ABS(l.vibration - s.avg_vib) / NULLIF(s.std_vib, 0) > 2
        THEN 'WARNING'
        ELSE 'HEALTHY'
    END AS health_status
FROM latest l
JOIN machine_stats s ON l.machineID = s.machineID
ORDER BY
    GREATEST(
        ABS(l.volt - s.avg_volt) / NULLIF(s.std_volt, 0),
        ABS(l.vibration - s.avg_vib) / NULLIF(s.std_vib, 0)
    ) DESC;

-- Query 10: Machines with most severe vibration drift (degradation indicator)
WITH baseline AS (
    SELECT machineID, AVG(vibration) AS baseline_vib
    FROM pdm_telemetry
    WHERE datetime BETWEEN '2015-01-01' AND '2015-01-31'
    GROUP BY machineID
),
recent AS (
    SELECT machineID, AVG(vibration) AS recent_vib
    FROM pdm_telemetry
    WHERE datetime >= '2015-11-01'
    GROUP BY machineID
)
SELECT
    b.machineID,
    ROUND(b.baseline_vib, 2) AS baseline_vibration,
    ROUND(r.recent_vib, 2) AS recent_vibration,
    ROUND(r.recent_vib - b.baseline_vib, 2) AS vibration_drift,
    ROUND(100.0 * (r.recent_vib - b.baseline_vib) / NULLIF(b.baseline_vib, 0), 1) AS pct_increase
FROM baseline b
JOIN recent r ON b.machineID = r.machineID
WHERE r.recent_vib > b.baseline_vib
ORDER BY vibration_drift DESC
LIMIT 15;

-- Query 11: Hourly degradation rate per machine (slope of vibration over time)
SELECT
    machineID,
    COUNT(*) AS data_points,
    ROUND(REGR_SLOPE(vibration, EXTRACT(EPOCH FROM datetime) / 3600.0), 6) AS vib_degradation_per_hour,
    ROUND(REGR_SLOPE(vibration, EXTRACT(EPOCH FROM datetime) / 3600.0) * 8760, 3) AS vib_degradation_per_year
FROM pdm_telemetry
GROUP BY machineID
HAVING COUNT(*) > 100
ORDER BY vib_degradation_per_hour DESC;

-- Query 12: Machines with increasing anomaly count over time
WITH daily_stats AS (
    SELECT
        machineID,
        DATE_TRUNC('day', datetime) AS day,
        AVG(vibration) AS avg_vib,
        STDDEV(vibration) AS std_vib
    FROM pdm_telemetry
    GROUP BY machineID, DATE_TRUNC('day', datetime)
),
anomalies AS (
    SELECT
        t.machineID,
        t.datetime,
        CASE WHEN ABS(t.vibration - ds.avg_vib) > 3 * NULLIF(ds.std_vib, 0)
             THEN 1 ELSE 0 END AS is_anomaly
    FROM pdm_telemetry t
    JOIN daily_stats ds ON t.machineID = ds.machineID
        AND DATE_TRUNC('day', t.datetime) = ds.day
)
SELECT
    machineID,
    DATE_TRUNC('week', datetime) AS week,
    SUM(is_anomaly) AS anomaly_count
FROM anomalies
GROUP BY machineID, DATE_TRUNC('week', datetime)
ORDER BY machineID, week;

-- Query 13: Machines that have never failed (for reliability analysis)
SELECT
    m.machineID,
    m.model,
    m.age,
    COUNT(t.machineID) AS telemetry_records
FROM pdm_machines m
LEFT JOIN pdm_failures f ON m.machineID = f.machineID
JOIN pdm_telemetry t ON m.machineID = t.machineID
WHERE f.machineID IS NULL
GROUP BY m.machineID, m.model, m.age
ORDER BY m.age DESC;

-- Query 14: Top 10 machines by total anomaly count
WITH daily_baseline AS (
    SELECT
        machineID,
        AVG(vibration) AS baseline_mean,
        STDDEV(vibration) AS baseline_std
    FROM pdm_telemetry
    WHERE datetime BETWEEN '2015-01-01' AND '2015-01-31'
    GROUP BY machineID
)
SELECT
    t.machineID,
    COUNT(*) AS anomaly_count,
    ROUND(100.0 * COUNT(*) / (
        SELECT COUNT(*) FROM pdm_telemetry t2 WHERE t2.machineID = t.machineID
    ), 2) AS anomaly_rate_pct
FROM pdm_telemetry t
JOIN daily_baseline b ON t.machineID = b.machineID
WHERE ABS(t.vibration - b.baseline_mean) > 3 * NULLIF(b.baseline_std, 0)
GROUP BY t.machineID
ORDER BY anomaly_count DESC
LIMIT 10;

-- Query 15: Rolling 7-day vibration trend per machine
SELECT
    machineID,
    DATE_TRUNC('day', datetime) AS day,
    AVG(vibration) AS daily_avg_vibration,
    AVG(AVG(vibration)) OVER (
        PARTITION BY machineID
        ORDER BY DATE_TRUNC('day', datetime)
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_7day_avg
FROM pdm_telemetry
GROUP BY machineID, DATE_TRUNC('day', datetime)
ORDER BY machineID, day;

-- Query 16: Sensor correlation per machine (identify coupled degradation)
SELECT
    machineID,
    CORR(volt, vibration) AS volt_vib_correlation,
    CORR(rotate, vibration) AS rotate_vib_correlation,
    CORR(pressure, vibration) AS pressure_vib_correlation,
    CORR(volt, pressure) AS volt_pressure_correlation
FROM pdm_telemetry
GROUP BY machineID
HAVING CORR(volt, vibration) IS NOT NULL
ORDER BY ABS(CORR(volt, vibration)) DESC;

-- Query 17: Machines that exceed vibration threshold in last 24 hours
WITH thresholds AS (
    SELECT
        machineID,
        AVG(vibration) + 3 * STDDEV(vibration) AS vib_upper_limit
    FROM pdm_telemetry
    WHERE datetime BETWEEN '2015-01-01' AND '2015-06-30'
    GROUP BY machineID
)
SELECT
    t.machineID,
    COUNT(*) AS exceedances,
    ROUND(MAX(t.vibration), 2) AS max_vibration,
    ROUND(th.vib_upper_limit, 2) AS threshold
FROM pdm_telemetry t
JOIN thresholds th ON t.machineID = th.machineID
WHERE t.datetime >= NOW() - INTERVAL '24 hours'
  AND t.vibration > th.vib_upper_limit
GROUP BY t.machineID, th.vib_upper_limit
ORDER BY exceedances DESC;

-- Query 18: Health score aggregate — percentage of machines at each health level
WITH health AS (
    SELECT
        t.machineID,
        CASE
            WHEN t.vibration > b.avg_vib + 3 * b.std_vib THEN 'Critical'
            WHEN t.vibration > b.avg_vib + 2 * b.std_vib THEN 'Warning'
            ELSE 'Healthy'
        END AS health_status
    FROM pdm_telemetry t
    JOIN (
        SELECT machineID, AVG(vibration) AS avg_vib, STDDEV(vibration) AS std_vib
        FROM pdm_telemetry WHERE datetime < '2015-06-01'
        GROUP BY machineID
    ) b ON t.machineID = b.machineID
    WHERE t.datetime = (
        SELECT MAX(datetime) FROM pdm_telemetry WHERE machineID = t.machineID
    )
)
SELECT
    health_status,
    COUNT(*) AS machine_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
FROM health
GROUP BY health_status
ORDER BY
    CASE health_status
        WHEN 'Critical' THEN 1
        WHEN 'Warning' THEN 2
        ELSE 3
    END;

-- Query 19: Machines with sudden spike detection (vibration > 2x running avg)
SELECT
    t1.machineID,
    t1.datetime,
    ROUND(t1.vibration, 2) AS current_vib,
    ROUND(t2.avg_vib, 2) AS running_avg,
    ROUND(t1.vibration / NULLIF(t2.avg_vib, 0), 2) AS spike_ratio
FROM pdm_telemetry t1
JOIN (
    SELECT
        machineID,
        AVG(vibration) AS avg_vib
    FROM pdm_telemetry
    WHERE datetime >= NOW() - INTERVAL '30 days'
    GROUP BY machineID
) t2 ON t1.machineID = t2.machineID
WHERE t1.datetime >= NOW() - INTERVAL '24 hours'
  AND t1.vibration > 2 * t2.avg_vib
ORDER BY spike_ratio DESC;

-- Query 20: Cumulative operating hours per machine
SELECT
    machineID,
    COUNT(*) AS total_hours_operated,
    MIN(datetime) AS first_recorded,
    MAX(datetime) AS last_recorded,
    EXTRACT(DAY FROM MAX(datetime) - MIN(datetime)) AS days_in_service
FROM pdm_telemetry
GROUP BY machineID
ORDER BY total_hours_operated DESC;


-- ====================================================================
-- SECTION 3: FAILURE ANALYSIS (Queries 21-30)
-- ====================================================================

-- Query 21: Total failures by component type
SELECT
    failure,
    COUNT(*) AS failure_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_total
FROM pdm_failures
GROUP BY failure
ORDER BY failure_count DESC;

-- Query 22: Monthly failure trend over time
SELECT
    DATE_TRUNC('month', datetime) AS month,
    failure,
    COUNT(*) AS failure_count
FROM pdm_failures
GROUP BY DATE_TRUNC('month', datetime), failure
ORDER BY month, failure;

-- Query 23: Mean Time Between Failures (MTBF) per machine
WITH failure_intervals AS (
    SELECT
        machineID,
        datetime,
        LAG(datetime) OVER (PARTITION BY machineID ORDER BY datetime) AS prev_failure,
        EXTRACT(EPOCH FROM datetime - LAG(datetime) OVER (
            PARTITION BY machineID ORDER BY datetime
        )) / 3600.0 AS hours_between_failures
    FROM pdm_failures
)
SELECT
    machineID,
    COUNT(*) AS total_failures,
    ROUND(AVG(hours_between_failures), 1) AS mtbf_hours,
    ROUND(AVG(hours_between_failures) / 24, 1) AS mtbf_days,
    ROUND(MIN(hours_between_failures), 1) AS min_time_between,
    ROUND(MAX(hours_between_failures), 1) AS max_time_between
FROM failure_intervals
WHERE prev_failure IS NOT NULL
GROUP BY machineID
ORDER BY mtbf_hours ASC
LIMIT 15;

-- Query 24: Weibull analysis parameters (shape and scale for reliability engineering)
-- Approximation using order statistics
WITH failure_times AS (
    SELECT
        machineID,
        ROW_NUMBER() OVER (PARTITION BY machineID ORDER BY datetime) AS failure_rank,
        EXTRACT(EPOCH FROM datetime - MIN(datetime) OVER (PARTITION BY machineID))
            / 3600.0 AS hours_since_first
    FROM pdm_failures
),
ranked AS (
    SELECT
        failure_rank,
        hours_since_first,
        LN(hours_since_first) AS ln_t,
        LN(LN(1.0 / (1.0 - failure_rank / (COUNT(*) OVER () + 1.0)))) AS ln_ln_inv_survival
    FROM failure_times
    WHERE hours_since_first > 0 AND failure_rank > 0
)
SELECT
    ROUND(REGR_SLOPE(ln_ln_inv_survival, ln_t), 3) AS weibull_shape_beta,
    ROUND(EXP(-REGR_INTERCEPT(ln_ln_inv_survival, ln_t)
              / NULLIF(REGR_SLOPE(ln_ln_inv_survival, ln_t), 0)), 1) AS weibull_scale_eta_hours
FROM ranked;

-- Query 25: Failure rate by machine model
SELECT
    m.model,
    COUNT(DISTINCT m.machineID) AS machine_count,
    COUNT(f.machineID) AS total_failures,
    ROUND(COUNT(f.machineID)::numeric / COUNT(DISTINCT m.machineID), 2) AS failures_per_machine,
    ROUND(COUNT(f.machineID)::numeric / COUNT(DISTINCT m.machineID) / 12, 2) AS failures_per_machine_per_month
FROM pdm_machines m
LEFT JOIN pdm_failures f ON m.machineID = f.machineID
GROUP BY m.model
ORDER BY failures_per_machine DESC;

-- Query 26: Time between failures analysis — identifying patterns
WITH ordered_failures AS (
    SELECT
        machineID,
        datetime,
        failure,
        ROW_NUMBER() OVER (PARTITION BY machineID ORDER BY datetime) AS seq
    FROM pdm_failures
)
SELECT
    a.machineID,
    a.datetime AS failure_time,
    a.failure AS current_failure,
    b.failure AS next_failure,
    EXTRACT(EPOCH FROM b.datetime - a.datetime) / 3600.0 AS hours_until_next
FROM ordered_failures a
LEFT JOIN ordered_failures b ON a.machineID = b.machineID AND a.seq + 1 = b.seq
WHERE b.machineID IS NOT NULL
ORDER BY hours_until_next ASC
LIMIT 20;

-- Query 27: Machines with highest failure frequency (top "problem machines")
SELECT
    f.machineID,
    m.model,
    m.age,
    COUNT(*) AS failure_count,
    MIN(f.datetime) AS first_failure,
    MAX(f.datetime) AS last_failure,
    EXTRACT(DAY FROM MAX(f.datetime) - MIN(f.datetime)) AS days_between_first_last
FROM pdm_failures f
JOIN pdm_machines m ON f.machineID = m.machineID
GROUP BY f.machineID, m.model, m.age
ORDER BY failure_count DESC
LIMIT 10;

-- Query 28: Component failure cascade analysis (which components fail together)
SELECT
    a.machineID,
    a.datetime AS failure_time,
    STRING_AGG(DISTINCT a.failure, ', ' ORDER BY a.failure) AS failed_components,
    COUNT(DISTINCT a.failure) AS components_affected
FROM pdm_failures a
JOIN pdm_failures b ON a.machineID = b.machineID
    AND ABS(EXTRACT(EPOCH FROM a.datetime - b.datetime)) / 3600.0 <= 24
    AND a.failure != b.failure
GROUP BY a.machineID, a.datetime
HAVING COUNT(DISTINCT a.failure) > 1
ORDER BY components_affected DESC;

-- Query 29: Failure probability by machine age bucket
SELECT
    CASE
        WHEN m.age = 0 THEN '0 years'
        WHEN m.age BETWEEN 1 AND 5 THEN '1-5 years'
        WHEN m.age BETWEEN 6 AND 10 THEN '6-10 years'
        WHEN m.age BETWEEN 11 AND 15 THEN '11-15 years'
        ELSE '15+ years'
    END AS age_group,
    COUNT(DISTINCT m.machineID) AS total_machines,
    COUNT(DISTINCT f.machineID) AS machines_with_failures,
    COUNT(*) AS total_failure_events,
    ROUND(COUNT(*)::numeric / COUNT(DISTINCT m.machineID), 2) AS failures_per_machine
FROM pdm_machines m
LEFT JOIN pdm_failures f ON m.machineID = f.machineID
GROUP BY
    CASE
        WHEN m.age = 0 THEN '0 years'
        WHEN m.age BETWEEN 1 AND 5 THEN '1-5 years'
        WHEN m.age BETWEEN 6 AND 10 THEN '6-10 years'
        WHEN m.age BETWEEN 11 AND 15 THEN '11-15 years'
        ELSE '15+ years'
    END
ORDER BY MIN(m.age);

-- Query 30: Failure count by day of week (operational pattern analysis)
SELECT
    EXTRACT(DOW FROM datetime) AS day_of_week,
    CASE EXTRACT(DOW FROM datetime)
        WHEN 0 THEN 'Sunday'
        WHEN 1 THEN 'Monday'
        WHEN 2 THEN 'Tuesday'
        WHEN 3 THEN 'Wednesday'
        WHEN 4 THEN 'Thursday'
        WHEN 5 THEN 'Friday'
        WHEN 6 THEN 'Saturday'
    END AS day_name,
    COUNT(*) AS failure_count
FROM pdm_failures
GROUP BY EXTRACT(DOW FROM datetime)
ORDER BY day_of_week;


-- ====================================================================
-- SECTION 4: MAINTENANCE OPTIMIZATION (Queries 31-38)
-- ====================================================================

-- Query 31: Reactive vs. planned maintenance ratio
SELECT
    CASE
        WHEN comp IN ('comp1', 'comp2', 'comp3', 'comp4') THEN 'Reactive (Unscheduled)'
        ELSE 'Planned (Scheduled)'
    END AS maintenance_type,
    COUNT(*) AS event_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
FROM pdm_maint
GROUP BY
    CASE
        WHEN comp IN ('comp1', 'comp2', 'comp3', 'comp4') THEN 'Reactive (Unscheduled)'
        ELSE 'Planned (Scheduled)'
    END;

-- Query 32: Optimal maintenance window analysis (hours between error and failure)
SELECT
    e.machineID,
    e.datetime AS error_time,
    f.datetime AS failure_time,
    EXTRACT(EPOCH FROM f.datetime - e.datetime) / 3600.0 AS hours_error_to_failure,
    e.errorID,
    f.failure
FROM pdm_errors e
JOIN pdm_failures f ON e.machineID = f.machineID
    AND f.datetime > e.datetime
    AND f.datetime <= e.datetime + INTERVAL '168 hours'
ORDER BY hours_error_to_failure ASC;

-- Query 33: Maintenance effectiveness — failures within 30 days of maintenance
SELECT
    maint.machineID,
    maint.datetime AS maintenance_date,
    maint.comp AS maintained_component,
    COUNT(fail.machineID) AS failures_within_30d,
    MIN(fail.datetime) AS first_subsequent_failure,
    EXTRACT(EPOCH FROM MIN(fail.datetime) - maint.datetime) / 3600.0 AS hours_to_failure
FROM pdm_maint maint
LEFT JOIN pdm_failures fail ON maint.machineID = fail.machineID
    AND fail.datetime > maint.datetime
    AND fail.datetime <= maint.datetime + INTERVAL '30 days'
GROUP BY maint.machineID, maint.datetime, maint.comp
HAVING COUNT(fail.machineID) > 0
ORDER BY hours_to_failure ASC;

-- Query 34: Cost comparison: early replacement vs. run-to-failure
-- Assumptions: planned cost = $2,500, emergency cost = $15,000 + 12hr downtime
WITH replacement_costs AS (
    SELECT
        comp,
        CASE
            WHEN comp IN ('comp1', 'comp2', 'comp3', 'comp4') THEN 15000 + 12 * 8000
            ELSE 2500 + 4 * 8000
        END AS estimated_cost
    FROM pdm_maint
)
SELECT
    comp,
    COUNT(*) AS replacements,
    SUM(estimated_cost) AS total_cost,
    ROUND(AVG(estimated_cost), 0) AS avg_cost_per_replacement
FROM replacement_costs
GROUP BY comp
ORDER BY total_cost DESC;

-- Query 35: Maintenance schedule optimization — time between consecutive maintenance events
WITH maintenance_ordered AS (
    SELECT
        machineID,
        datetime,
        LAG(datetime) OVER (PARTITION BY machineID ORDER BY datetime) AS prev_maintenance,
        comp
    FROM pdm_maint
)
SELECT
    machineID,
    comp,
    COUNT(*) AS maintenance_count,
    ROUND(AVG(EXTRACT(EPOCH FROM datetime - prev_maintenance) / 3600.0), 1) AS avg_hours_between,
    ROUND(MIN(EXTRACT(EPOCH FROM datetime - prev_maintenance) / 3600.0), 1) AS min_hours_between,
    ROUND(MAX(EXTRACT(EPOCH FROM datetime - prev_maintenance) / 3600.0), 1) AS max_hours_between
FROM maintenance_ordered
WHERE prev_maintenance IS NOT NULL
GROUP BY machineID, comp
ORDER BY avg_hours_between ASC
LIMIT 20;

-- Query 36: Component replacement frequency analysis
SELECT
    comp AS component,
    COUNT(*) AS replacement_count,
    COUNT(DISTINCT machineID) AS machines_affected,
    ROUND(AVG(AGE(MAX(datetime), MIN(datetime))), 0) AS avg_days_between_replacements
FROM pdm_maint
GROUP BY comp
ORDER BY replacement_count DESC;

-- Query 37: Identify machines overdue for maintenance (no maintenance in last 90 days)
SELECT
    m.machineID,
    MAX(mt.datetime) AS last_maintenance,
    EXTRACT(DAY FROM NOW() - MAX(mt.datetime)) AS days_since_last,
    m.model,
    m.age
FROM pdm_machines m
LEFT JOIN pdm_maint mt ON m.machineID = mt.machineID
GROUP BY m.machineID, m.model, m.age
HAVING MAX(mt.datetime) < NOW() - INTERVAL '90 days'
   OR MAX(mt.datetime) IS NULL
ORDER BY days_since_last DESC NULLS FIRST;

-- Query 38: Maintenance efficiency — hours of downtime per maintenance event
SELECT
    DATE_TRUNC('month', datetime) AS month,
    COUNT(*) AS maintenance_events,
    COUNT(*) * 4 AS planned_downtime_hours,  -- Assumption: 4 hours per planned
    SUM(CASE WHEN comp IN ('comp1', 'comp2', 'comp3', 'comp4') THEN 12 ELSE 4 END) AS estimated_downtime_hours
FROM pdm_maint
GROUP BY DATE_TRUNC('month', datetime)
ORDER BY month;


-- ====================================================================
-- SECTION 5: COST ANALYTICS (Queries 39-45)
-- ====================================================================

-- Query 39: Total cost of ownership estimate (annualized)
-- Assumptions: planned=$2,500, emergency=$15,000 ($8,000/hr × 12hr downtime + repair)
WITH cost_model AS (
    SELECT
        DATE_TRUNC('month', mt.datetime) AS month,
        COUNT(*) FILTER (WHERE mt.comp IN ('comp1', 'comp2', 'comp3', 'comp4')) AS emergency_repairs,
        COUNT(*) FILTER (WHERE mt.comp NOT IN ('comp1', 'comp2', 'comp3', 'comp4')) AS planned_repairs,
        COUNT(DISTINCT f.machineID) AS unique_failures
    FROM pdm_maint mt
    LEFT JOIN pdm_failures f ON mt.machineID = f.machineID
        AND DATE_TRUNC('month', mt.datetime) = DATE_TRUNC('month', f.datetime)
    GROUP BY DATE_TRUNC('month', mt.datetime)
)
SELECT
    TO_CHAR(month, 'YYYY-MM') AS month,
    emergency_repairs,
    planned_repairs,
    emergency_repairs * 15000 + emergency_repairs * 12 * 8000 AS emergency_cost,
    planned_repairs * 2500 + planned_repairs * 4 * 8000 AS planned_cost,
    emergency_repairs * 15000 + emergency_repairs * 12 * 8000
    + planned_repairs * 2500 + planned_repairs * 4 * 8000 AS total_monthly_cost
FROM cost_model
ORDER BY month;

-- Query 40: Cost per failure type
SELECT
    f.failure,
    COUNT(*) AS occurrences,
    COUNT(*) * 15000 AS emergency_repair_cost,
    COUNT(*) * 12 * 8000 AS downtime_cost,
    COUNT(*) * 15000 + COUNT(*) * 12 * 8000 AS total_cost,
    ROUND(100.0 * (COUNT(*) * 15000 + COUNT(*) * 12 * 8000)
          / SUM(COUNT(*) * 15000 + COUNT(*) * 12 * 8000) OVER (), 1) AS pct_of_total_cost
FROM pdm_failures f
GROUP BY f.failure
ORDER BY total_cost DESC;

-- Query 41: ROI calculation comparing reactive to predictive maintenance
-- Reactive: all failures result in emergency repairs
-- Predictive: 80% of failures caught early → planned repairs
WITH failure_stats AS (
    SELECT COUNT(*) AS total_failures FROM pdm_failures
)
SELECT
    'Reactive (Current)' AS strategy,
    total_failures AS failure_events,
    total_failures * (15000 + 12 * 8000) AS annual_cost,
    0 AS savings
FROM failure_stats
UNION ALL
SELECT
    'Predictive (ML-Driven)' AS strategy,
    ROUND(total_failures * 0.2) AS failure_events,
    ROUND(total_failures * 0.8 * (2500 + 4 * 8000) + total_failures * 0.2 * (15000 + 12 * 8000)) AS annual_cost,
    ROUND(total_failures * (15000 + 12 * 8000)
          - (total_failures * 0.8 * (2500 + 4 * 8000) + total_failures * 0.2 * (15000 + 12 * 8000))) AS savings
FROM failure_stats;

-- Query 42: Cost-benefit analysis of installing IoT sensors
-- Per-machine sensor cost: $20,000, annual platform: $50,000
WITH baseline AS (
    SELECT COUNT(*) AS n_failures FROM pdm_failures
),
cost_breakdown AS (
    SELECT
        100 AS n_machines,
        100 * 20000 AS total_hardware_cost,
        50000 AS annual_software_cost,
        120000 AS annual_personnel_cost,
        (SELECT n_failures FROM baseline) AS annual_failures
)
SELECT
    n_machines,
    total_hardware_cost,
    annual_software_cost,
    annual_personnel_cost,
    total_hardware_cost / 3 + annual_software_cost + annual_personnel_cost AS annualized_cost,
    annual_failures * 0.8 * (15000 + 12 * 8000 - 2500 - 4 * 8000) AS gross_savings,
    annual_failures * 0.8 * (15000 + 12 * 8000 - 2500 - 4 * 8000)
    - (total_hardware_cost / 3 + annual_software_cost + annual_personnel_cost) AS net_savings_year1
FROM cost_breakdown;

-- Query 43: Cumulative cost of unplanned downtime
SELECT
    DATE_TRUNC('month', datetime) AS month,
    COUNT(*) AS failures,
    COUNT(*) * 12 AS downtime_hours,
    COUNT(*) * 12 * 8000 AS downtime_cost,
    SUM(COUNT(*) * 12 * 8000) OVER (ORDER BY DATE_TRUNC('month', datetime)) AS cumulative_cost
FROM pdm_failures
GROUP BY DATE_TRUNC('month', datetime)
ORDER BY month;

-- Query 44: Cost per machine per year
SELECT
    f.machineID,
    COUNT(*) AS failures,
    COUNT(*) * (15000 + 12 * 8000) AS annual_failure_cost,
    COUNT(mt.machineID) AS maintenance_events,
    COUNT(mt.machineID) * 2500 AS annual_planned_cost,
    COUNT(*) * (15000 + 12 * 8000) + COUNT(mt.machineID) * 2500 AS total_annual_cost
FROM pdm_failures f
LEFT JOIN pdm_maint mt ON f.machineID = mt.machineID
GROUP BY f.machineID
ORDER BY total_annual_cost DESC
LIMIT 15;

-- Query 45: Savings from converting emergency repairs to planned (at 80% detection rate)
SELECT
    'comp1' AS component,
    (SELECT COUNT(*) FROM pdm_failures WHERE failure = 'comp1') AS failures,
    (SELECT COUNT(*) FROM pdm_failures WHERE failure = 'comp1') * 0.8 * (111000 - 34500) AS potential_savings
UNION ALL
SELECT 'comp2', (SELECT COUNT(*) FROM pdm_failures WHERE failure = 'comp2'),
    (SELECT COUNT(*) FROM pdm_failures WHERE failure = 'comp2') * 0.8 * (111000 - 34500)
UNION ALL
SELECT 'comp3', (SELECT COUNT(*) FROM pdm_failures WHERE failure = 'comp3'),
    (SELECT COUNT(*) FROM pdm_failures WHERE failure = 'comp3') * 0.8 * (111000 - 34500)
UNION ALL
SELECT 'comp4', (SELECT COUNT(*) FROM pdm_failures WHERE failure = 'comp4'),
    (SELECT COUNT(*) FROM pdm_failures WHERE failure = 'comp4') * 0.8 * (111000 - 34500);


-- ====================================================================
-- SECTION 6: FLEET MANAGEMENT (Queries 46-52)
-- ====================================================================

-- Query 46: Fleet-wide risk score ranking
WITH risk_factors AS (
    SELECT
        t.machineID,
        COUNT(f.machineID) AS historical_failures,
        AVG(t.vibration) AS avg_vibration,
        MAX(mt.datetime) AS last_maintenance,
        CASE WHEN MAX(mt.datetime) < NOW() - INTERVAL '90 days' THEN 1 ELSE 0 END AS overdue_maintenance
    FROM pdm_telemetry t
    LEFT JOIN pdm_failures f ON t.machineID = f.machineID
    LEFT JOIN pdm_maint mt ON t.machineID = mt.machineID
    WHERE t.datetime >= NOW() - INTERVAL '30 days'
    GROUP BY t.machineID
)
SELECT
    machineID,
    historical_failures,
    ROUND(avg_vibration, 2) AS avg_vibration,
    overdue_maintenance,
    ROUND(
        historical_failures * 0.4
        + (avg_vibration / (SELECT AVG(vibration) FROM pdm_telemetry)) * 0.3
        + overdue_maintenance * 0.3,
        2
    ) AS risk_score
FROM risk_factors
ORDER BY risk_score DESC
LIMIT 20;

-- Query 47: Cross-machine correlation — identify machines with similar failure patterns
SELECT
    a.machineID AS machine_a,
    b.machineID AS machine_b,
    m1.model AS model_a,
    m2.model AS model_b,
    CORR(
        (SELECT AVG(vibration) FROM pdm_telemetry WHERE machineID = a.machineID),
        (SELECT AVG(vibration) FROM pdm_telemetry WHERE machineID = b.machineID)
    ) AS vibration_correlation
FROM pdm_failures a
CROSS JOIN pdm_failures b
JOIN pdm_machines m1 ON a.machineID = m1.machineID
JOIN pdm_machines m2 ON b.machineID = m2.machineID
WHERE a.machineID < b.machineID
  AND a.failure = b.failure
GROUP BY a.machineID, b.machineID, m1.model, m2.model
ORDER BY vibration_correlation DESC;

-- Query 48: Fleet utilization analysis (operating vs idle patterns)
SELECT
    DATE_TRUNC('hour', datetime) AS hour,
    COUNT(DISTINCT machineID) AS machines_reporting,
    ROUND(100.0 * COUNT(DISTINCT machineID) / 100, 1) AS pct_fleet_reporting
FROM pdm_telemetry
GROUP BY DATE_TRUNC('hour', datetime)
ORDER BY hour;

-- Query 49: Monthly fleet health summary
SELECT
    DATE_TRUNC('month', t.datetime) AS month,
    COUNT(DISTINCT t.machineID) AS active_machines,
    ROUND(AVG(t.vibration), 2) AS avg_vibration,
    ROUND(STDDEV(t.vibration), 2) AS std_vibration,
    COUNT(DISTINCT f.machineID) AS machines_with_failures,
    COUNT(DISTINCT e.machineID) AS machines_with_errors,
    ROUND(COUNT(DISTINCT f.machineID)::numeric / 100, 3) AS failure_rate
FROM pdm_telemetry t
LEFT JOIN pdm_failures f ON DATE_TRUNC('month', t.datetime) = DATE_TRUNC('month', f.datetime)
LEFT JOIN pdm_errors e ON DATE_TRUNC('month', t.datetime) = DATE_TRUNC('month', e.datetime)
GROUP BY DATE_TRUNC('month', t.datetime)
ORDER BY month;

-- Query 50: Machines that should be prioritized for replacement/retirement
WITH scores AS (
    SELECT
        m.machineID,
        m.model,
        m.age,
        COUNT(DISTINCT f.machineID) AS total_failures,
        AVG(t.vibration) AS avg_vibration,
        MAX(f.datetime) AS last_failure
    FROM pdm_machines m
    JOIN pdm_telemetry t ON m.machineID = t.machineID
    LEFT JOIN pdm_failures f ON m.machineID = f.machineID
    WHERE t.datetime >= NOW() - INTERVAL '60 days'
    GROUP BY m.machineID, m.model, m.age
)
SELECT
    machineID,
    model,
    age,
    total_failures,
    ROUND(avg_vibration, 2) AS recent_avg_vibration,
    last_failure,
    ROUND(
        total_failures * 0.5 + age * 0.3 + (avg_vibration / 50.0) * 0.2,
        2
    ) AS replacement_priority_score
FROM scores
ORDER BY replacement_priority_score DESC
LIMIT 10;

-- Query 51: Fleet-wide KPI dashboard (single-row summary)
SELECT
    COUNT(DISTINCT t.machineID) AS total_machines,
    ROUND(AVG(t.volt), 2) AS fleet_avg_voltage,
    ROUND(AVG(t.vibration), 2) AS fleet_avg_vibration,
    COUNT(DISTINCT f.machineID) AS machines_with_failures_30d,
    COUNT(DISTINCT e.machineID) AS machines_with_errors_30d,
    ROUND(100.0 * COUNT(DISTINCT f.machineID) / COUNT(DISTINCT t.machineID), 2) AS pct_failed_30d
FROM pdm_telemetry t
LEFT JOIN pdm_failures f ON t.machineID = f.machineID
    AND f.datetime >= NOW() - INTERVAL '30 days'
LEFT JOIN pdm_errors e ON t.machineID = e.machineID
    AND e.datetime >= NOW() - INTERVAL '30 days'
WHERE t.datetime >= NOW() - INTERVAL '7 days';

-- Query 52: Peer group comparison — rank similar machines by health
WITH machine_profiles AS (
    SELECT
        m.machineID,
        m.model,
        m.age,
        AVG(t.vibration) AS avg_vib,
        AVG(t.volt) AS avg_volt,
        COUNT(f.machineID) AS failure_count
    FROM pdm_machines m
    JOIN pdm_telemetry t ON m.machineID = t.machineID
    LEFT JOIN pdm_failures f ON m.machineID = f.machineID
    WHERE t.datetime >= NOW() - INTERVAL '30 days'
    GROUP BY m.machineID, m.model, m.age
)
SELECT
    model,
    machineID,
    age,
    ROUND(avg_vib, 2) AS avg_vibration,
    failure_count,
    RANK() OVER (PARTITION BY model ORDER BY avg_vib DESC) AS vibration_rank_in_model,
    RANK() OVER (PARTITION BY model ORDER BY failure_count DESC) AS failure_rank_in_model
FROM machine_profiles
ORDER BY model, vibration_rank_in_model;


-- ====================================================================
-- SECTION 7: ADVANCED ANALYTICS (Queries 53-57)
-- ====================================================================

-- Query 53: Survival analysis — percentage of machines surviving without failure
WITH failure_times AS (
    SELECT
        machineID,
        MIN(datetime) AS first_failure_time,
        EXTRACT(EPOCH FROM MIN(datetime) - '2015-01-01'::timestamp) / 3600.0 AS hours_to_failure
    FROM pdm_failures
    GROUP BY machineID
),
all_machines AS (
    SELECT machineID, 8760 AS total_hours FROM pdm_machines  -- 365 * 24
),
survival AS (
    SELECT
        a.machineID,
        COALESCE(f.hours_to_failure, 8760) AS survival_hours,
        CASE WHEN f.machineID IS NULL THEN 1 ELSE 0 END AS censored
    FROM all_machines a
    LEFT JOIN failure_times f ON a.machineID = f.machineID
)
SELECT
    ROUND(100.0 * SUM(CASE WHEN survival_hours > 2000 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_survive_2000hr,
    ROUND(100.0 * SUM(CASE WHEN survival_hours > 4000 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_survive_4000hr,
    ROUND(100.0 * SUM(CASE WHEN survival_hours > 6000 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_survive_6000hr,
    ROUND(100.0 * SUM(CASE WHEN survival_hours > 8000 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_survive_8000hr,
    ROUND(AVG(survival_hours), 0) AS mean_survival_hours,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY survival_hours), 0) AS median_survival_hours
FROM survival;

-- Query 54: Lead-time forecasting — predict next maintenance based on historical patterns
WITH maint_intervals AS (
    SELECT
        machineID,
        comp,
        datetime,
        LAG(datetime) OVER (PARTITION BY machineID, comp ORDER BY datetime) AS prev_maint,
        EXTRACT(EPOCH FROM datetime - LAG(datetime) OVER (
            PARTITION BY machineID, comp ORDER BY datetime
        )) / 3600.0 AS hours_between
    FROM pdm_maint
)
SELECT
    comp AS component,
    COUNT(*) AS sample_size,
    ROUND(AVG(hours_between), 0) AS avg_interval_hours,
    ROUND(STDDEV(hours_between), 0) AS std_interval_hours,
    ROUND(AVG(hours_between) - 1.96 * STDDEV(hours_between), 0) AS lower_95ci,
    ROUND(AVG(hours_between) + 1.96 * STDDEV(hours_between), 0) AS upper_95ci
FROM maint_intervals
WHERE prev_maint IS NOT NULL
GROUP BY comp
ORDER BY avg_interval_hours;

-- Query 55: Anomaly detection — identify unusual sensor patterns preceding failures
WITH pre_failure_data AS (
    SELECT
        t.machineID,
        t.datetime,
        t.volt,
        t.vibration,
        t.rotate,
        t.pressure,
        f.datetime AS failure_time,
        f.failure,
        ROW_NUMBER() OVER (PARTITION BY t.machineID, f.datetime ORDER BY t.datetime DESC) AS hours_before
    FROM pdm_telemetry t
    JOIN pdm_failures f ON t.machineID = f.machineID
        AND t.datetime BETWEEN f.datetime - INTERVAL '72 hours' AND f.datetime
)
SELECT
    failure,
    COUNT(*) AS sample_count,
    ROUND(AVG(volt), 2) AS avg_prefailure_voltage,
    ROUND(AVG(vibration), 2) AS avg_prefailure_vibration,
    ROUND(AVG(rotate), 2) AS avg_prefailure_rotation,
    ROUND(AVG(pressure), 2) AS avg_prefailure_pressure,
    ROUND(STDDEV(vibration), 4) AS std_prefailure_vibration
FROM pre_failure_data
WHERE hours_before <= 72
GROUP BY failure
ORDER BY sample_count DESC;

-- Query 56: Prognostic horizon estimation — how early can we detect each failure type?
WITH pre_failure_zscore AS (
    SELECT
        f.failure,
        t.machineID,
        t.datetime,
        EXTRACT(EPOCH FROM f.datetime - t.datetime) / 3600.0 AS hours_before,
        (t.vibration - b.avg_vib) / NULLIF(b.std_vib, 0) AS vib_zscore
    FROM pdm_failures f
    JOIN pdm_telemetry t ON f.machineID = t.machineID
        AND t.datetime BETWEEN f.datetime - INTERVAL '168 hours' AND f.datetime - INTERVAL '1 hour'
    JOIN (
        SELECT machineID, AVG(vibration) AS avg_vib, STDDEV(vibration) AS std_vib
        FROM pdm_telemetry WHERE datetime < '2015-06-01'
        GROUP BY machineID
    ) b ON t.machineID = b.machineID
)
SELECT
    failure,
    ROUND(AVG(hours_before) FILTER (WHERE ABS(vib_zscore) > 2), 1) AS avg_detection_horizon_hours,
    ROUND(MAX(hours_before) FILTER (WHERE ABS(vib_zscore) > 2), 1) AS max_detection_horizon_hours,
    COUNT(*) FILTER (WHERE ABS(vib_zscore) > 2) AS detectable_events
FROM pre_failure_zscore
GROUP BY failure
ORDER BY avg_detection_horizon_hours DESC;

-- Query 57: Feature importance proxy — which sensor correlates most with failure?
WITH failure_markers AS (
    SELECT
        t.machineID,
        t.volt,
        t.rotate,
        t.pressure,
        t.vibration,
        CASE WHEN f.machineID IS NOT NULL THEN 1 ELSE 0 END AS will_fail_24h
    FROM pdm_telemetry t
    LEFT JOIN pdm_failures f ON t.machineID = f.machineID
        AND f.datetime > t.datetime
        AND f.datetime <= t.datetime + INTERVAL '24 hours'
)
SELECT
    'voltage' AS sensor,
    ROUND(CORR(volt, will_fail_24h)::numeric, 4) AS correlation_with_failure
FROM failure_markers
UNION ALL
SELECT 'rotation', ROUND(CORR(rotate, will_fail_24h)::numeric, 4) FROM failure_markers
UNION ALL
SELECT 'pressure', ROUND(CORR(pressure, will_fail_24h)::numeric, 4) FROM failure_markers
UNION ALL
SELECT 'vibration', ROUND(CORR(vibration, will_fail_24h)::numeric, 4) FROM failure_markers
ORDER BY ABS(correlation_with_failure) DESC;
