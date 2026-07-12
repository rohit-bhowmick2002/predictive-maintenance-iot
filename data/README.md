# Data Dictionary — Microsoft Azure Predictive Maintenance Dataset

## Overview

The dataset captures one full year (2015) of operational telemetry from **100 industrial machines** across a manufacturing facility. It consists of 5 interconnected tables representing a realistic industrial data warehouse schema.

---

## Table 1: Telemetry (`PdM_telemetry.csv`)

**876,099 rows × 6 columns** — Hourly sensor readings from all machines.

| Column | Type | Description | Range/Values |
|---|---|---|---|
| `datetime` | datetime | Timestamp of reading (hourly) | 2015-01-01 to 2015-12-31 |
| `machineID` | int | Machine identifier | 1–100 |
| `volt` | float | Voltage reading | ~150–190 V |
| `rotate` | float | Rotational speed | ~400–500 RPM |
| `pressure` | float | Pressure reading | ~90–110 units |
| `vibration` | float | Vibration measurement | ~25–55 units |

**Data Quality:** No missing values. 99.8% hourly completeness across all machines.

---

## Table 2: Errors (`PdM_errors.csv`)

**3,919 rows × 3 columns** — Error events logged during operation.

| Column | Type | Description |
|---|---|---|
| `datetime` | datetime | When error occurred |
| `machineID` | int | Machine identifier |
| `errorID` | str | Error type code (error1–error5) |

**Note:** Errors do not necessarily indicate failures — they are operational warnings.

---

## Table 3: Maintenance (`PdM_maint.csv`)

**3,286 rows × 3 columns** — Scheduled and unscheduled maintenance records.

| Column | Type | Description |
|---|---|---|
| `datetime` | datetime | When maintenance was performed |
| `machineID` | int | Machine identifier |
| `comp` | str | Component replaced (comp1–comp4 or other) |

**Key insight:** Records where `comp` ∈ {comp1, comp2, comp3, comp4} represent **reactive** (failure-driven) replacements. Other values are **planned** maintenance.

---

## Table 4: Failures (`PdM_failures.csv`)

**761 rows × 3 columns** — Component failure/replacement events.

| Column | Type | Description |
|---|---|---|
| `datetime` | datetime | When failure occurred |
| `machineID` | int | Machine identifier |
| `failure` | str | Failed component (comp1, comp2, comp3, comp4) |

**Failure Distribution:**
- comp1: ~23%
- comp2: ~41% (most frequent)
- comp3: ~19%
- comp4: ~17%

---

## Table 5: Machines (`PdM_machines.csv`)

**100 rows × 3 columns** — Machine metadata.

| Column | Type | Description |
|---|---|---|
| `machineID` | int | Machine identifier (1–100) |
| `model` | str | Machine model (model1–model4) |
| `age` | int | Age in years (0–20) |

---

## Data Pipeline

```
Raw CSVs
  │
  ├── PdM_telemetry.csv (876K rows)
  ├── PdM_errors.csv     (3.9K rows)
  ├── PdM_maint.csv      (3.3K rows)
  ├── PdM_failures.csv   (761 rows)
  └── PdM_machines.csv   (100 rows)
        │
        ▼
   AzurePdMLoader
   (src/data/loader.py)
        │
        ├── Merge machine metadata
        ├── Build error flags (rolling windows)
        ├── Build maintenance history features
        └── Build failure labels (24h prediction horizon)
        │
        ▼
   FeatureEngineer
   (src/data/feature_engine.py)
        │
        ├── 200+ engineered features
        ├── Rolling statistics (6h, 12h, 24h, 48h, 72h, 168h)
        ├── Rate-of-change & trend features
        ├── Frequency-domain (FFT) features
        └── Cross-sensor interaction features
        │
        ▼
   features.parquet
   (/data/processed/)
```

---

## Download

```bash
cd data/raw
wget https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_telemetry.csv
wget https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_errors.csv
wget https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_maint.csv
wget https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_failures.csv
wget https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_machines.csv
```

**Source:** [Microsoft Azure AI Gallery — Predictive Maintenance](https://gallery.azure.ai/)

---

## Class Imbalance

The failure classification task is **severely imbalanced:**
- ~2.5% of time windows contain a failure
- ~97.5% are normal operation
- Models must handle this imbalance via class weighting, cost-sensitive thresholds, or sampling techniques
