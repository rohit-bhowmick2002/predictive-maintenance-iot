"""
Unit tests for the Azure PdM data loader.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAzurePdMLoader:
    """Tests for AzurePdMLoader class."""

    def test_import(self):
        """Verify module imports correctly."""
        from src.data.loader import AzurePdMLoader
        loader = AzurePdMLoader()
        assert loader is not None

    def test_column_mapping(self):
        """Verify column name mapping is correct."""
        from src.utils.config import config
        assert "volt" in config.data.column_mapping
        assert config.data.column_mapping["volt"] == "voltage"
        assert config.data.column_mapping["machineID"] == "machine_id"

    def test_sensor_columns(self):
        """Verify sensor columns are correctly configured."""
        from src.utils.config import config
        expected = ["volt", "rotate", "pressure", "vibration"]
        assert config.data.sensor_cols == expected

    def test_dataset_stats_dataclass(self):
        """Verify DatasetStats dataclass works."""
        from src.data.loader import DatasetStats
        stats = DatasetStats(
            n_rows=1000, n_machines=10, n_features=20,
            date_range=("2020-01-01", "2020-12-31"),
            n_failures=50, failure_rate_pct=5.0,
            n_errors=30, n_maintenance_events=20,
            sensor_columns=["v1", "v2"],
        )
        assert stats.n_rows == 1000
        assert stats.failure_rate_pct == 5.0


class TestPreprocessor:
    """Tests for PdMPreprocessor."""

    def test_import(self):
        from src.data.preprocessor import PdMPreprocessor
        preprocessor = PdMPreprocessor()
        assert preprocessor is not None
        assert preprocessor._fitted is False

    def test_class_weights(self):
        """Verify class weight calculation for imbalanced data."""
        from src.data.preprocessor import PdMPreprocessor
        y = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        weights = PdMPreprocessor.get_class_weights(y)
        assert 0 in weights
        assert 1 in weights
        assert weights[0] < weights[1]  # Minority class gets higher weight

    def test_config_values(self):
        """Verify configuration values are reasonable."""
        from src.utils.config import config
        assert 0 < config.model.test_size < 1
        assert config.model.random_seed == 42
        assert len(config.features.window_sizes) > 0


class TestFeatureEngineer:
    """Tests for FeatureEngineer."""

    def test_import(self):
        from src.data.feature_engine import FeatureEngineer
        engineer = FeatureEngineer()
        assert engineer is not None

    def test_window_sizes(self):
        from src.utils.config import config
        assert 6 in config.features.window_sizes
        assert 168 in config.features.window_sizes

    def test_statistical_features(self):
        from src.utils.config import config
        stats = config.features.statistical_features
        assert "mean" in stats
        assert "std" in stats
        assert len(stats) >= 6


@pytest.mark.smoke
def test_config_loading():
    """Smoke test: verify config loads without error."""
    from src.utils.config import Config
    cfg = Config()
    assert cfg.data is not None
    assert cfg.model is not None
    assert cfg.features is not None
    assert cfg.evaluation is not None
