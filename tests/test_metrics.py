"""
Unit tests for evaluation metrics.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def binary_data():
    """Generate test prediction data."""
    np.random.seed(42)
    y_true = np.array([0] * 450 + [1] * 50)  # 500 samples, 10% positive
    np.random.shuffle(y_true)
    y_pred = y_true.copy()
    y_pred[np.random.choice(500, 15, replace=False)] = 1 - y_pred[np.random.choice(500, 15, replace=False)]  # ~3% errors
    y_proba = np.clip(y_true.astype(float) + np.random.normal(0, 0.15, 500), 0, 1)
    return y_true, y_pred, y_proba


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_compute_all_metrics(self, binary_data):
        from src.evaluation.metrics import compute_all_metrics
        y_true, y_pred, y_proba = binary_data
        metrics = compute_all_metrics(y_true, y_pred, y_proba, "Test")

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics
        assert "cost_total" in metrics
        assert 0 <= metrics["f1_score"] <= 1

    def test_phm_score_perfect(self):
        from src.evaluation.metrics import _compute_phm_score
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.1, 0.9, 0.9])
        score = _compute_phm_score(y_true, y_proba, y_pred)
        assert score > 0.9  # Near perfect

    def test_phm_score_bad(self):
        from src.evaluation.metrics import _compute_phm_score
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        y_proba = np.array([0.9, 0.9, 0.1, 0.1])
        score = _compute_phm_score(y_true, y_proba, y_pred)
        assert score < 0.5  # Very poor

    def test_compare_models(self, binary_data):
        from src.evaluation.metrics import compute_all_metrics, compare_models
        y_true, y_pred, y_proba = binary_data
        m1 = compute_all_metrics(y_true, y_pred, y_proba, "Model_A")
        m2 = compute_all_metrics(y_true, y_pred, y_proba, "Model_B")
        df = compare_models([m1, m2])
        assert len(df) == 2

    def test_threshold_analysis(self, binary_data):
        from src.evaluation.metrics import compute_threshold_analysis
        y_true, y_pred, y_proba = binary_data
        df = compute_threshold_analysis(y_true, y_proba)
        assert len(df) > 0
        assert "threshold" in df.columns
        assert "cost_total" in df.columns


class TestCostAnalysis:
    """Tests for cost analysis module."""

    def test_simulate_maintenance_costs(self):
        from src.evaluation.cost_analysis import simulate_maintenance_costs
        results = simulate_maintenance_costs(n_machines=50, n_years=3)

        assert "reactive" in results
        assert "preventive" in results
        assert "predictive" in results
        assert "roi" in results

        # Predictive should be cheaper than reactive
        assert results["predictive"]["annual_cost"] < results["reactive"]["annual_cost"]

        # ROI should be positive
        assert results["roi"]["roi_percentage"] > 0


class TestVisualization:
    """Tests for visualization module."""

    def test_import(self):
        from src.evaluation.visualization import plot_confusion_matrix
        assert callable(plot_confusion_matrix)

    @pytest.mark.smoke
    def test_generate_plots(self, binary_data, tmp_path):
        y_true, y_pred, y_proba = binary_data
        from src.evaluation.visualization import plot_confusion_matrix
        import matplotlib
        matplotlib.use("Agg")

        output = tmp_path / "test_cm.png"
        plot_confusion_matrix(y_true, y_pred, output_path=output)
        assert output.exists()

    @pytest.mark.smoke
    def test_threshold_optimization_plot(self, tmp_path):
        from src.evaluation.visualization import plot_threshold_optimization
        import matplotlib
        matplotlib.use("Agg")

        thresholds = np.arange(0.1, 1.0, 0.1)
        precision = np.array([0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.35])
        recall = np.array([0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95])
        costs = np.array([5000, 4000, 3000, 2500, 2200, 2500, 3000, 4000, 5000])

        output = tmp_path / "threshold_opt.png"
        plot_threshold_optimization(thresholds, precision, recall, costs, output_path=output)
        assert output.exists()
