"""
Feature engineering module — generates 200+ features from raw sensor telemetry.

Feature categories:
  1. Statistical aggregates (rolling windows: 6h, 12h, 24h, 48h, 72h, 168h)
  2. Trend features (linear regression slope over windows)
  3. Rate-of-change features (first derivative approximations)
  4. Frequency-domain features (FFT, spectral energy)
  5. Interaction features (cross-sensor ratios and products)
  6. Cumulative degradation indicators
  7. Error & maintenance history features
"""

import pandas as pd
import numpy as np
from scipy import stats, signal
from pathlib import Path
from typing import List, Optional
from ..utils.config import config
from ..utils.logger import logger


class FeatureEngineer:
    """
    Generates a rich feature set from raw telemetry data.

    Usage:
        engineer = FeatureEngineer()
        df_features = engineer.fit_transform(df)
    """

    def __init__(self):
        self.cfg = config.features
        self.sensor_cols = config.data.sensor_cols
        self._feature_names: List[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features from raw telemetry.

        Args:
            df: Merged DataFrame from AzurePdMLoader

        Returns:
            DataFrame with original + engineered features
        """
        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING — Generating 200+ features")
        logger.info("=" * 60)

        df = df.sort_values(["machine_id", "timestamp"]).copy()

        # 1. Statistical rolling window features
        df = self._add_rolling_statistics(df)

        # 2. Rate-of-change features
        df = self._add_rate_of_change(df)

        # 3. Trend features
        df = self._add_trend_features(df)

        # 4. Frequency-domain features
        df = self._add_frequency_features(df)

        # 5. Cross-sensor interaction features
        df = self._add_interaction_features(df)

        # 6. Cumulative degradation indicators
        df = self._add_degradation_indicators(df)

        # 7. Time-based features
        df = self._add_temporal_features(df)

        # 8. Error and maintenance history features
        df = self._add_history_features(df)

        # Drop rows with NaN from rolling windows
        n_before = len(df)
        df = df.dropna()
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.info(f"  Dropped {n_dropped:,} rows with NaN values ({n_dropped/n_before*100:.1f}%)")

        self._feature_names = [c for c in df.columns if c not in
            ["machine_id", "timestamp", "failure_label", "failure_components",
             "failure_comp1", "failure_comp2", "failure_comp3", "failure_comp4",
             "time_to_failure_hours"] + self.sensor_cols]

        logger.info(f"✅ Generated {len(self._feature_names)} engineered features")
        logger.info(f"   Final shape: {len(df):,} rows × {len(df.columns)} columns")

        # Save processed dataset
        output_path = config.data.processed_dir / "features.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"   Saved to {output_path}")

        return df

    # ------------------------------------------------------------------
    # 1. Rolling Statistical Features (~84 features)
    # ------------------------------------------------------------------

    def _add_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window statistical aggregates per machine."""
        logger.info("  [1/8] Rolling statistical features...")
        stats_funcs = self.cfg.statistical_features
        windows = self.cfg.window_sizes

        new_cols = []
        for machine_id, group in df.groupby("machine_id"):
            group = group.set_index("timestamp")
            for col in self.sensor_cols:
                for w in windows:
                    rolling = group[col].rolling(window=f"{w}H", min_periods=max(2, w // 6))

                    for stat in stats_funcs:
                        col_name = f"{col}_{stat}_{w}h"

                        if stat == "mean":
                            group[col_name] = rolling.mean().values
                        elif stat == "std":
                            group[col_name] = rolling.std().values
                        elif stat == "min":
                            group[col_name] = rolling.min().values
                        elif stat == "max":
                            group[col_name] = rolling.max().values
                        elif stat == "skew":
                            group[col_name] = rolling.skew().values
                        elif stat == "kurtosis":
                            group[col_name] = rolling.kurt().values
                        elif stat == "q25":
                            group[col_name] = rolling.quantile(0.25).values
                        elif stat == "q50":
                            group[col_name] = rolling.quantile(0.50).values
                        elif stat == "q75":
                            group[col_name] = rolling.quantile(0.75).values
                        elif stat == "range":
                            group[col_name] = (rolling.max() - rolling.min()).values
                        elif stat == "iqr":
                            q75 = rolling.quantile(0.75)
                            q25 = rolling.quantile(0.25)
                            group[col_name] = (q75 - q25).values

                        new_cols.append(col_name)

            group = group.reset_index()
            df.loc[group.index, group.columns] = group

        logger.info(f"      Added {len(set(new_cols))} rolling statistical features")
        return df

    # ------------------------------------------------------------------
    # 2. Rate-of-Change Features (~24 features)
    # ------------------------------------------------------------------

    def _add_rate_of_change(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add first and second derivatives of sensor readings."""
        logger.info("  [2/8] Rate-of-change features...")

        for col in self.sensor_cols:
            df[f"{col}_delta_1h"] = df.groupby("machine_id")[col].diff(1)
            df[f"{col}_delta_6h"] = df.groupby("machine_id")[col].diff(6)
            df[f"{col}_delta_24h"] = df.groupby("machine_id")[col].diff(24)

            # Acceleration (2nd derivative)
            df[f"{col}_accel_1h"] = df.groupby("machine_id")[f"{col}_delta_1h"].diff(1)

            # Relative change
            df[f"{col}_pct_change_24h"] = df.groupby("machine_id")[col].pct_change(24).fillna(0)
            df[f"{col}_pct_change_24h"] = df[f"{col}_pct_change_24h"].replace([np.inf, -np.inf], 0)

            # Rolling rate of change
            df[f"{col}_roc_24h"] = (
                df.groupby("machine_id")[col].transform(lambda x: x.rolling(24, min_periods=6).apply(
                    lambda y: (y.iloc[-1] - y.iloc[0]) / (len(y) - 1) if len(y) > 1 else 0
                ))
            )

        return df

    # ------------------------------------------------------------------
    # 3. Trend Features (~12 features)
    # ------------------------------------------------------------------

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add linear trend features over rolling windows."""
        logger.info("  [3/8] Trend features...")

        for col in self.sensor_cols:
            for w in [24, 72, 168]:
                def _compute_trend(series, window=w):
                    if len(series) < window // 2:
                        return np.nan
                    x = np.arange(len(series))
                    slope, _, r, _, _ = stats.linregress(x, series)
                    return slope

                df[f"{col}_slope_{w}h"] = (
                    df.groupby("machine_id")[col]
                    .transform(lambda x: x.rolling(w, min_periods=max(6, w // 4))
                              .apply(_compute_trend, raw=False))
                )

        return df

    # ------------------------------------------------------------------
    # 4. Frequency-Domain Features (~12 features)
    # ------------------------------------------------------------------

    def _add_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add FFT-based frequency domain features."""
        logger.info("  [4/8] Frequency-domain features...")

        window = 64  # Power of 2 for FFT

        for col in self.sensor_cols:
            df[f"{col}_spectral_energy"] = np.nan
            df[f"{col}_spectral_centroid"] = np.nan
            df[f"{col}_dominant_freq"] = np.nan

            for machine_id in df["machine_id"].unique():
                mask = df["machine_id"] == machine_id
                machine_idx = df[mask].index

                for i in range(window, len(machine_idx)):
                    segment = df.loc[machine_idx[i - window:i], col].values
                    if len(segment) < window // 2:
                        continue

                    fft = np.abs(np.fft.rfft(segment - np.mean(segment)))
                    freqs = np.fft.rfftfreq(window)

                    if np.sum(fft) > 0:
                        df.loc[machine_idx[i], f"{col}_spectral_energy"] = np.sum(fft ** 2)
                        df.loc[machine_idx[i], f"{col}_spectral_centroid"] = (
                            np.sum(freqs * fft) / np.sum(fft) if np.sum(fft) > 0 else 0
                        )
                        df.loc[machine_idx[i], f"{col}_dominant_freq"] = freqs[np.argmax(fft)]

        return df

    # ------------------------------------------------------------------
    # 5. Interaction Features (~24 features)
    # ------------------------------------------------------------------

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-sensor interaction features."""
        logger.info("  [5/8] Interaction features...")

        pairs = [
            ("voltage", "rotation_speed"),
            ("voltage", "pressure"),
            ("voltage", "vibration"),
            ("rotation_speed", "pressure"),
            ("rotation_speed", "vibration"),
            ("pressure", "vibration"),
        ]

        for a, b in pairs:
            if a in df.columns and b in df.columns:
                df[f"{a}_x_{b}"] = df[a] * df[b]
                df[f"{a}_div_{b}"] = df[a] / (df[b].replace(0, np.nan) + 1e-8)
                df[f"{a}_minus_{b}"] = df[a] - df[b]

            # Rolling correlation between sensors
            df[f"{a}_{b}_corr_24h"] = (
                df.groupby("machine_id").apply(
                    lambda g: g[a].rolling(24, min_periods=12).corr(g[b])
                ).reset_index(level=0, drop=True)
            )

        return df

    # ------------------------------------------------------------------
    # 6. Degradation Indicators (~16 features)
    # ------------------------------------------------------------------

    def _add_degradation_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cumulative degradation indicators."""
        logger.info("  [6/8] Degradation indicators...")

        for col in self.sensor_cols:
            # Exponentially weighted moving average (EWMA) — emphasizes recent behavior
            df[f"{col}_ewma_alpha01"] = df.groupby("machine_id")[col].transform(
                lambda x: x.ewm(alpha=0.1, adjust=False).mean()
            )
            df[f"{col}_ewma_alpha005"] = df.groupby("machine_id")[col].transform(
                lambda x: x.ewm(alpha=0.05, adjust=False).mean()
            )

            # Cumulative deviation from mean
            df[f"{col}_cumul_deviation"] = df.groupby("machine_id")[col].transform(
                lambda x: (x - x.expanding().mean()).cumsum()
            )

            # Running z-score relative to first 168 hours
            df[f"{col}_running_zscore"] = df.groupby("machine_id")[col].transform(
                lambda x: (x - x.iloc[:168].mean()) / (x.iloc[:168].std() + 1e-8)
            )

        return df

    # ------------------------------------------------------------------
    # 7. Temporal Features (~8 features)
    # ------------------------------------------------------------------

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        logger.info("  [7/8] Temporal features...")

        df["hour_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["day_of_month"] = df["timestamp"].dt.day
        df["month"] = df["timestamp"].dt.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_night_shift"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] < 6)).astype(int)

        # Cyclical encoding for hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

        return df

    # ------------------------------------------------------------------
    # 8. History Features (~12 features)
    # ------------------------------------------------------------------

    def _add_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add error and maintenance history features."""
        logger.info("  [8/8] History features...")

        # Error history: rolling sum of errors over various windows
        if "error_count_1h" in df.columns:
            for w in [24, 72, 168]:
                df[f"error_count_{w}h"] = (
                    df.groupby("machine_id")["error_count_1h"]
                    .transform(lambda x: x.rolling(w, min_periods=1).sum())
                )

        # Maintenance history
        if "maint_count_7d" in df.columns:
            for w in [7, 30]:
                df[f"days_since_last_maint"] = np.nan
                for machine_id in df["machine_id"].unique():
                    mask = df["machine_id"] == machine_id
                    machine_df = df[mask].copy()
                    maint_mask = machine_df["maint_count_7d"] > 0
                    if maint_mask.any():
                        last_maint = machine_df.loc[maint_mask, "timestamp"].max()
                        hours_since = (machine_df["timestamp"] - last_maint).dt.total_seconds() / 3600
                        df.loc[mask, "days_since_last_maint"] = hours_since / 24

        return df

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    # For standalone testing
    print("Feature engineer ready. Run from main pipeline.")
    engineer = FeatureEngineer()
    print(f"Configured with window sizes: {engine.cfg.window_sizes}")
