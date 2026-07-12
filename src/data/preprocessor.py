"""
Preprocessing pipeline for the PdM telemetry dataset.

Key operations:
  1. Label engineering: convert component failures into classification targets
  2. Sliding window construction for sequence models
  3. Train/validation/test split respecting temporal ordering
  4. Feature scaling (StandardScaler, robust to sensor drift)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
from ..utils.config import config
from ..utils.logger import logger


class PdMPreprocessor:
    """
    Preprocesses telemetry data for failure classification.

    Usage:
        preprocessor = PdMPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    """

    def __init__(self):
        self.cfg = config
        self.scaler: Optional[RobustScaler] = None
        self.feature_names: list = []
        self._fitted = False

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = "failure_label",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit scaler on training data and transform all splits.

        Args:
            df: Merged dataframe from AzurePdMLoader
            target_col: Name of the binary target column

        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("Starting preprocessing pipeline...")

        # --- Identify feature columns ---
        sensor_cols = self.cfg.data.sensor_cols
        static_cols = ["machine_age_years"]
        flag_cols = [c for c in df.columns if c.startswith(("error_", "maint_"))]

        feature_cols = sensor_cols + [c for c in static_cols if c in df.columns]
        feature_cols += [c for c in flag_cols if c in df.columns and not df[c].dtype == object]

        # Remove object-type columns
        feature_cols = [c for c in feature_cols if df[c].dtype != object]

        self.feature_names = feature_cols
        logger.info(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

        # --- Temporal train/test split ---
        # Sort by time, use last 20% chronologically as test
        df_sorted = df.sort_values("timestamp")
        split_idx = int(len(df_sorted) * (1 - self.cfg.model.test_size))

        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:]

        logger.info(f"  Train: {len(train_df):,} rows ({train_df['timestamp'].min()} → {train_df['timestamp'].max()})")
        logger.info(f"  Test:  {len(test_df):,} rows ({test_df['timestamp'].min()} → {test_df['timestamp'].max()})")

        # --- Check class balance ---
        train_pos_rate = train_df[target_col].mean()
        test_pos_rate = test_df[target_col].mean()
        logger.info(f"  Train positive rate: {train_pos_rate:.4f} ({train_pos_rate*100:.2f}%)")
        logger.info(f"  Test positive rate:  {test_pos_rate:.4f} ({test_pos_rate*100:.2f}%)")

        # --- Fit scaler on training data ---
        self.scaler = RobustScaler()  # Robust to outliers common in sensor data
        self.scaler.fit(train_df[feature_cols])

        # --- Transform ---
        X_train = self.scaler.transform(train_df[feature_cols])
        X_test = self.scaler.transform(test_df[feature_cols])
        y_train = train_df[target_col].values.astype(np.int8)
        y_test = test_df[target_col].values.astype(np.int8)

        self._fitted = True
        logger.info(f"✅ Preprocessing complete. X_train: {X_train.shape}, X_test: {X_test.shape}")

        # Save scaler
        joblib.dump(self.scaler, self.cfg.data.models_dir / "scaler.joblib")

        return X_train, X_test, y_train, y_test

    def build_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 48,
        step: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build sequences for LSTM/Transformer models.

        Each sample is a [sequence_length, n_features] window.

        Args:
            df: DataFrame with sensor data (sorted by machine, time)
            sequence_length: Number of time steps per sequence
            step: Stride between sequences

        Returns:
            X: (n_sequences, sequence_length, n_features)
            y: (n_sequences,) target labels
            meta: (n_sequences, 3) — [machine_id, start_time, end_time]
        """
        logger.info(f"Building sequences (len={sequence_length}, step={step})...")

        sensor_cols = self.cfg.data.sensor_cols
        feature_cols = [c for c in sensor_cols if c in df.columns and df[c].dtype != object]

        X, y, meta = [], [], []

        for machine_id in df["machine_id"].unique():
            machine_df = df[df["machine_id"] == machine_id].sort_values("timestamp")
            values = machine_df[feature_cols].values
            labels = machine_df["failure_label"].values if "failure_label" in machine_df else np.zeros(len(machine_df))
            timestamps = machine_df["timestamp"].values

            for i in range(0, len(values) - sequence_length, step):
                X.append(values[i : i + sequence_length])
                y.append(labels[i + sequence_length - 1])
                meta.append([machine_id, timestamps[i], timestamps[i + sequence_length - 1]])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int8)
        meta = np.array(meta)

        logger.info(f"  Built {len(X):,} sequences of shape {X.shape[1:]}")

        # Save scaler for sequences
        self.seq_scaler = RobustScaler()
        orig_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = self.seq_scaler.fit_transform(X_flat)
        X = X_scaled.reshape(orig_shape)

        joblib.dump(self.seq_scaler, self.cfg.data.models_dir / "seq_scaler.joblib")

        return X, y, meta

    def split_sequences(
        self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split sequences preserving temporal order."""
        split_idx = int(len(X) * (1 - test_size))
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

    @staticmethod
    def get_class_weights(y: np.ndarray) -> dict:
        """Compute balanced class weights for imbalance handling."""
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        return dict(zip(classes, weights))


if __name__ == "__main__":
    from ..data.loader import AzurePdMLoader

    loader = AzurePdMLoader()
    df = loader.load_all()

    preprocessor = PdMPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Class weights: {PdMPreprocessor.get_class_weights(y_train)}")
