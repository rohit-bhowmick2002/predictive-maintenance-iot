"""
Data validation and quality monitoring for PdM pipeline.

Checks:
  1. Schema validation (column names, types)
  2. Completeness (missing value detection)
  3. Consistency (value ranges, logical constraints)
  4. Timeliness (data freshness)
  5. Drift detection (statistical distribution changes)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from ..utils.config import config
from ..utils.logger import logger


@dataclass
class ValidationResult:
    """Result of a data validation check."""
    check_name: str
    passed: bool
    details: str = ""
    severity: str = "error"  # error, warning, info


class DataValidator:
    """
    Validates telemetry data quality against expected schemas and rules.

    Usage:
        validator = DataValidator()
        results = validator.validate_telemetry(df)
        if not all(r.passed for r in results if r.severity == "error"):
            raise ValueError("Data validation failed")
    """

    def __init__(self):
        self.cfg = config.data
        self.results: List[ValidationResult] = []

    def validate_telemetry(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Run all telemetry validation checks."""
        self.results = []

        # Schema checks
        self._check_required_columns(df, ["machineID", "datetime"] + self.cfg.sensor_cols)
        self._check_no_duplicates(df, subset=["machineID", "datetime"])
        self._check_dtypes(df, {
            "machineID": "int",
            "datetime": "datetime",
            "volt": "float",
            "rotate": "float",
            "pressure": "float",
            "vibration": "float",
        })

        # Completeness
        self._check_completeness(df, threshold=0.95)
        self._check_machine_count(df, expected=100)

        # Value range checks
        for col in self.cfg.sensor_cols:
            self._check_positive_values(df, col)
            self._check_outlier_count(df, col, n_std=5.0, max_pct=0.05)

        # Timeliness
        self._check_date_range(df, min_year=2015, max_gap_hours=8760)
        self._check_hourly_frequency(df)

        # Logical consistency
        self._check_monotonic_per_machine(df)

        # Report
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        errors = sum(1 for r in self.results if not r.passed and r.severity == "error")

        logger.info(f"Data Validation: {passed} passed, {failed} failed ({errors} errors)")
        for r in self.results:
            status = "✅" if r.passed else "❌" if r.severity == "error" else "⚠️"
            logger.info(f"  {status} [{r.severity}] {r.check_name}: {r.details}")

        return self.results

    def _check_required_columns(self, df: pd.DataFrame, required: List[str]):
        missing = [c for c in required if c not in df.columns]
        self.results.append(ValidationResult(
            "required_columns",
            len(missing) == 0,
            f"Missing: {missing}" if missing else "All required columns present",
            "error",
        ))

    def _check_no_duplicates(self, df: pd.DataFrame, subset: List[str]):
        n_dupes = df.duplicated(subset=subset).sum()
        self.results.append(ValidationResult(
            "no_duplicates",
            n_dupes == 0,
            f"Found {n_dupes} duplicate rows" if n_dupes > 0 else "No duplicates found",
            "error",
        ))

    def _check_dtypes(self, df: pd.DataFrame, expected: Dict[str, str]):
        mismatches = []
        for col, dtype in expected.items():
            if col in df.columns and str(df[col].dtype).startswith(dtype):
                continue
            elif col in df.columns:
                mismatches.append(f"{col}: expected {dtype}, got {df[col].dtype}")
        self.results.append(ValidationResult(
            "column_dtypes",
            len(mismatches) == 0,
            "; ".join(mismatches) if mismatches else "All types correct",
            "error",
        ))

    def _check_completeness(self, df: pd.DataFrame, threshold: float = 0.95):
        completeness = 1 - df.isna().mean().mean()
        self.results.append(ValidationResult(
            "completeness",
            completeness >= threshold,
            f"Data {completeness:.1%} complete (threshold: {threshold:.0%})",
            "error",
        ))

    def _check_machine_count(self, df: pd.DataFrame, expected: int = 100):
        actual = df["machineID"].nunique()
        self.results.append(ValidationResult(
            "machine_count",
            actual == expected,
            f"Expected {expected} machines, found {actual}",
            "error",
        ))

    def _check_positive_values(self, df: pd.DataFrame, col: str):
        neg_count = (df[col] < 0).sum()
        self.results.append(ValidationResult(
            f"positive_values_{col}",
            neg_count == 0,
            f"{neg_count} negative values found in {col}" if neg_count > 0 else f"All {col} values positive",
            "warning",
        ))

    def _check_outlier_count(self, df: pd.DataFrame, col: str, n_std: float = 5.0, max_pct: float = 0.05):
        mean, std = df[col].mean(), df[col].std()
        if std == 0:
            return
        outlier_pct = (abs(df[col] - mean) > n_std * std).mean()
        self.results.append(ValidationResult(
            f"outlier_check_{col}",
            outlier_pct <= max_pct,
            f"{outlier_pct:.2%} outliers in {col} (threshold: {max_pct:.0%})",
            "warning",
        ))

    def _check_date_range(self, df: pd.DataFrame, min_year: int = 2015, max_gap_hours: int = 8760):
        min_date = df["datetime"].min()
        max_date = df["datetime"].max()
        span_hours = (max_date - min_date).total_seconds() / 3600
        self.results.append(ValidationResult(
            "date_range",
            min_date.year >= min_year and span_hours <= max_gap_hours,
            f"Range: {min_date.date()} to {max_date.date()} ({span_hours:.0f} hours)",
            "info",
        ))

    def _check_hourly_frequency(self, df: pd.DataFrame):
        """Check that data follows hourly frequency (within tolerance)."""
        from collections import Counter
        for machine_id in df["machineID"].unique()[:5]:  # Sample 5 machines
            machine_data = df[df["machineID"] == machine_id].sort_values("datetime")
            if len(machine_data) > 2:
                diffs = machine_data["datetime"].diff().dropna()
                hourly = diffs[diffs == pd.Timedelta(hours=1)]
                if len(hourly) / len(diffs) < 0.90:
                    self.results.append(ValidationResult(
                        "hourly_frequency",
                        False,
                        f"Machine {machine_id}: only {len(hourly)/len(diffs):.0%} of intervals are exactly 1 hour",
                        "warning",
                    ))
                    return
        self.results.append(ValidationResult(
            "hourly_frequency",
            True,
            "Sampled machines show correct hourly frequency",
            "info",
        ))

    def _check_monotonic_per_machine(self, df: pd.DataFrame):
        """Verify timestamps are monotonic within each machine."""
        all_monotonic = True
        for machine_id, group in df.groupby("machineID"):
            if not group.sort_values("datetime")["datetime"].is_monotonic_increasing:
                all_monotonic = False
                break
        self.results.append(ValidationResult(
            "monotonic_timestamps",
            all_monotonic,
            "Timestamps monotonic per machine" if all_monotonic else "Non-monotonic timestamps found",
            "error",
        ))

    def check_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        threshold_pct: float = 0.10,
    ) -> List[ValidationResult]:
        """
        Detect distribution drift between reference and current data.

        Uses Kolmogorov-Smirnov test for each sensor column.
        """
        from scipy import stats
        drift_results = []

        for col in self.cfg.sensor_cols:
            if col in reference_df.columns and col in current_df.columns:
                ref = reference_df[col].dropna().sample(min(5000, len(reference_df)))
                cur = current_df[col].dropna().sample(min(5000, len(current_df)))

                ks_stat, p_value = stats.ks_2samp(ref, cur)

                drifted = p_value < 0.05 and ks_stat > threshold_pct
                drift_results.append(ValidationResult(
                    f"drift_check_{col}",
                    not drifted,
                    f"KS={ks_stat:.4f}, p={p_value:.4f}" + (" — DRIFT DETECTED" if drifted else ""),
                    "warning" if drifted else "info",
                ))

        for r in drift_results:
            status = "⚠️" if not r.passed else "ℹ️"
            logger.info(f"  {status} {r.check_name}: {r.details}")

        return drift_results
