"""
Multi-table data ingestion for the Microsoft Azure Predictive Maintenance dataset.

Loads and joins 5 tables:
  1. PdM_telemetry.csv  — 876,099 rows: hourly sensor readings from 100 machines
  2. PdM_errors.csv     — Error events logged during operation
  3. PdM_maint.csv      — Maintenance records (both planned & reactive)
  4. PdM_failures.csv   — Component failure/replacement events
  5. PdM_machines.csv   — Machine metadata (model, age)

Schema after join: time-series telemetry enriched with error flags,
maintenance flags, failure labels, and machine metadata.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from ..utils.config import config
from ..utils.logger import logger


@dataclass
class DatasetStats:
    """Summary statistics for the loaded dataset."""
    n_rows: int
    n_machines: int
    n_features: int
    date_range: Tuple[str, str]
    n_failures: int
    failure_rate_pct: float
    n_errors: int
    n_maintenance_events: int
    sensor_columns: list


class AzurePdMLoader:
    """
    Loads, validates, and joins the Azure Predictive Maintenance dataset.

    Usage:
        loader = AzurePdMLoader()
        df = loader.load_all()
        stats = loader.get_statistics()
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or config.data.raw_dir
        self.cfg = config.data
        self._telemetry: Optional[pd.DataFrame] = None
        self._errors: Optional[pd.DataFrame] = None
        self._maint: Optional[pd.DataFrame] = None
        self._failures: Optional[pd.DataFrame] = None
        self._machines: Optional[pd.DataFrame] = None
        self._merged: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Individual table loaders
    # ------------------------------------------------------------------

    def load_telemetry(self) -> pd.DataFrame:
        """Load hourly sensor telemetry from 100 machines (876,099 rows)."""
        logger.info("Loading telemetry data...")
        path = self.data_dir / self.cfg.telemetry_file

        dtypes = {
            "machineID": "int32",
            "volt": "float32",
            "rotate": "float32",
            "pressure": "float32",
            "vibration": "float32",
        }
        df = pd.read_csv(
            path,
            parse_dates=["datetime"],
            dtype=dtypes,
            low_memory=False,
        )
        # Map column names
        df = df.rename(columns=self.cfg.column_mapping)
        # Sort by machine and time
        df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
        self._telemetry = df
        logger.info(f"  → Loaded {len(df):,} telemetry records from {df['machine_id'].nunique()} machines")
        return df

    def load_errors(self) -> pd.DataFrame:
        """Load machine error events."""
        logger.info("Loading error data...")
        path = self.data_dir / self.cfg.errors_file
        df = pd.read_csv(path, parse_dates=["datetime", "errorID"])
        df = df.rename(columns=self.cfg.column_mapping)
        self._errors = df
        logger.info(f"  → Loaded {len(df):,} error records")
        return df

    def load_maintenance(self) -> pd.DataFrame:
        """Load maintenance records (planned and reactive)."""
        logger.info("Loading maintenance data...")
        path = self.data_dir / self.cfg.maint_file
        df = pd.read_csv(path, parse_dates=["datetime"])
        df = df.rename(columns=self.cfg.column_mapping)
        self._maint = df
        logger.info(f"  → Loaded {len(df):,} maintenance records")
        return df

    def load_failures(self) -> pd.DataFrame:
        """Load component failure/replacement events."""
        logger.info("Loading failure data...")
        path = self.data_dir / self.cfg.failures_file
        df = pd.read_csv(path, parse_dates=["datetime"])
        df = df.rename(columns=self.cfg.column_mapping)
        self._failures = df
        logger.info(f"  → Loaded {len(df):,} failure records")
        return df

    def load_machines(self) -> pd.DataFrame:
        """Load machine metadata (model type, age)."""
        logger.info("Loading machine metadata...")
        path = self.data_dir / self.cfg.machines_file
        df = pd.read_csv(path)
        df = df.rename(columns=self.cfg.column_mapping)
        self._machines = df
        logger.info(f"  → Loaded {len(df):,} machine metadata records")
        return df

    # ------------------------------------------------------------------
    # Full load & join
    # ------------------------------------------------------------------

    def load_all(self) -> pd.DataFrame:
        """
        Load all 5 tables and perform the master join.

        Returns a single DataFrame with telemetry enriched by:
          - Error flag (was there an error in the last N hours?)
          - Maintenance flag (was maintenance done recently?)
          - Failure label (did this component fail in the next M hours?)
          - Machine metadata (model, age)
        """
        logger.info("=" * 60)
        logger.info("LOADING AZURE PREDICTIVE MAINTENANCE DATASET")
        logger.info("=" * 60)

        # Load all tables
        telemetry = self.load_telemetry()
        errors = self.load_errors()
        maint = self.load_maintenance()
        failures = self.load_failures()
        machines = self.load_machines()

        # --- Join machine metadata ---
        df = telemetry.merge(machines, on="machine_id", how="left")
        logger.info(f"After machine metadata join: {len(df):,} rows")

        # --- Create error flags ---
        # For each telemetry row, flag if any error occurred in preceding 24h window
        error_flags = self._build_error_features(df, errors)
        df = df.merge(error_flags, on=["machine_id", "timestamp"], how="left")

        # --- Create maintenance flags ---
        maint_flags = self._build_maintenance_features(df, maint)
        df = df.merge(maint_flags, on=["machine_id", "timestamp"], how="left")

        # --- Create failure labels ---
        failure_labels = self._build_failure_labels(df, failures)
        df = df.merge(failure_labels, on=["machine_id", "timestamp"], how="left")

        # Fill NaN flags
        flag_cols = [c for c in df.columns if c.startswith(("error_", "maint_", "failure_"))]
        df[flag_cols] = df[flag_cols].fillna(0).astype("int8")

        logger.info(f"✅ Final merged dataset: {len(df):,} rows × {len(df.columns)} columns")
        self._merged = df
        return df

    # ------------------------------------------------------------------
    # Feature construction helpers
    # ------------------------------------------------------------------

    def _build_error_features(
        self, telemetry: pd.DataFrame, errors: pd.DataFrame
    ) -> pd.DataFrame:
        """Build error-related features: error in last 24h, error count, error type."""
        if errors.empty:
            return pd.DataFrame()

        # Aggregate errors by machine and hour
        errors["timestamp_hour"] = errors["timestamp"].dt.floor("H")
        error_agg = (
            errors.groupby(["machine_id", "timestamp_hour"])
            .agg(
                error_count=("error_id", "count"),
                error_types=("error_id", lambda x: list(x.unique())),
            )
            .reset_index()
        )
        error_agg.columns = ["machine_id", "timestamp", "error_count_1h", "error_types_1h"]

        # Join to telemetry at hour level
        telemetry_hour = telemetry.copy()
        telemetry_hour["timestamp"] = telemetry_hour["timestamp"].dt.floor("H")

        result = telemetry_hour[["machine_id", "timestamp"]].drop_duplicates()
        result = result.merge(error_agg, on=["machine_id", "timestamp"], how="left")

        return result

    def _build_maintenance_features(
        self, telemetry: pd.DataFrame, maint: pd.DataFrame
    ) -> pd.DataFrame:
        """Build maintenance-related features."""
        if maint.empty:
            return pd.DataFrame()

        maint_sorted = maint.sort_values(["machine_id", "timestamp"])

        result_rows = []
        for (machine_id, ts), group in telemetry.groupby(["machine_id", pd.Grouper(key="timestamp", freq="H")]):
            machine_maint = maint_sorted[maint_sorted["machine_id"] == machine_id]
            if not machine_maint.empty:
                # Count maintenance in last 7 days
                recent = machine_maint[
                    (machine_maint["timestamp"] <= ts)
                    & (machine_maint["timestamp"] > ts - pd.Timedelta(days=7))
                ]
                result_rows.append({
                    "machine_id": machine_id,
                    "timestamp": ts,
                    "maint_count_7d": len(recent),
                    "maint_components_replaced": ",".join(recent["component"].astype(str).unique())
                    if not recent.empty else "",
                })

        if result_rows:
            return pd.DataFrame(result_rows)
        return pd.DataFrame(columns=["machine_id", "timestamp", "maint_count_7d", "maint_components_replaced"])

    def _build_failure_labels(
        self, telemetry: pd.DataFrame, failures: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build failure classification labels.

        For each telemetry timestamp, label = 1 if ANY component fails
        within the prediction horizon (default: 24 hours).
        """
        if failures.empty:
            return pd.DataFrame()

        pred_horizon = config.features.prediction_horizon_hours
        result_rows = []

        for machine_id in telemetry["machine_id"].unique():
            machine_tele = telemetry[telemetry["machine_id"] == machine_id]
            machine_fails = failures[failures["machine_id"] == machine_id]

            for _, row in machine_tele.iterrows():
                ts = row["timestamp"]
                horizon_end = ts + pd.Timedelta(hours=pred_horizon)

                # Check if any failure occurs in the prediction window
                failures_in_window = machine_fails[
                    (machine_fails["timestamp"] > ts)
                    & (machine_fails["timestamp"] <= horizon_end)
                ]

                label = 1 if len(failures_in_window) > 0 else 0

                # Also capture which component(s) will fail
                failing_comps = (
                    ",".join(failures_in_window["failure_type"].astype(str).unique())
                    if label == 1 else ""
                )
                comp1 = 1 if label == 1 and any("comp1" in str(c) for c in failures_in_window["failure_type"]) else 0
                comp2 = 1 if label == 1 and any("comp2" in str(c) for c in failures_in_window["failure_type"]) else 0
                comp3 = 1 if label == 1 and any("comp3" in str(c) for c in failures_in_window["failure_type"]) else 0
                comp4 = 1 if label == 1 and any("comp4" in str(c) for c in failures_in_window["failure_type"]) else 0

                # Time to next failure (for regression tasks)
                future_fails = machine_fails[machine_fails["timestamp"] > ts]
                ttf_hours = (
                    (future_fails["timestamp"].min() - ts).total_seconds() / 3600
                    if not future_fails.empty else 9999
                )

                result_rows.append({
                    "machine_id": machine_id,
                    "timestamp": ts,
                    "failure_label": label,
                    "failure_components": failing_comps,
                    "failure_comp1": comp1,
                    "failure_comp2": comp2,
                    "failure_comp3": comp3,
                    "failure_comp4": comp4,
                    "time_to_failure_hours": ttf_hours,
                })

        return pd.DataFrame(result_rows)

    # ------------------------------------------------------------------
    # Statistics & validation
    # ------------------------------------------------------------------

    def get_statistics(self) -> DatasetStats:
        """Compute summary statistics for the loaded dataset."""
        df = self._merged
        if df is None:
            raise ValueError("Data not loaded. Call load_all() first.")

        return DatasetStats(
            n_rows=len(df),
            n_machines=df["machine_id"].nunique(),
            n_features=len(df.columns),
            date_range=(str(df["timestamp"].min()), str(df["timestamp"].max())),
            n_failures=int(df["failure_label"].sum()) if "failure_label" in df.columns else 0,
            failure_rate_pct=round(df["failure_label"].mean() * 100, 3) if "failure_label" in df.columns else 0.0,
            n_errors=int(df.get("error_count_1h", pd.Series([0])).sum()),
            n_maintenance_events=int(df.get("maint_count_7d", pd.Series([0])).sum()),
            sensor_columns=self.cfg.sensor_cols,
        )

    def validate_data(self) -> Dict[str, bool]:
        """Run data quality checks."""
        df = self._merged
        if df is None:
            raise ValueError("Data not loaded. Call load_all() first.")

        checks = {
            "has_telemetry": len(df) > 0,
            "has_timestamps": df["timestamp"].notna().all(),
            "has_machine_ids": df["machine_id"].notna().all(),
            "no_duplicate_rows": not df.duplicated(subset=["machine_id", "timestamp"]).any(),
            "timestamps_monotonic_per_machine": all(
                df.groupby("machine_id")["timestamp"].is_monotonic_increasing
            ),
            "sensors_no_null": all(df[self.cfg.sensor_cols].notna().all()),
            "sensors_positive": all((df[self.cfg.sensor_cols] >= 0).all()),
            "machine_count_correct": df["machine_id"].nunique() == 100,
        }

        logger.info("Data validation results:")
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            logger.info(f"  {status} {check}")

        return checks


if __name__ == "__main__":
    loader = AzurePdMLoader()
    df = loader.load_all()
    stats = loader.get_statistics()
    print(f"\n{stats}")
    loader.validate_data()
