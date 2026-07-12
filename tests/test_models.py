"""
Unit tests for ML models.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_data():
    """Generate synthetic data for model testing."""
    np.random.seed(42)
    X = np.random.randn(500, 10)
    y = np.zeros(500, dtype=int)
    y[np.random.choice(500, 50, replace=False)] = 1  # 10% positive
    return X, y


@pytest.fixture
def sample_sequences():
    """Generate synthetic sequence data."""
    np.random.seed(42)
    X = np.random.randn(200, 48, 4).astype(np.float32)
    y = np.zeros(200, dtype=np.int8)
    y[np.random.choice(200, 20, replace=False)] = 1
    return X, y


class TestXGBoostModel:
    """Tests for XGBoostFailureClassifier."""

    def test_import(self):
        from src.models.baseline import XGBoostFailureClassifier
        model = XGBoostFailureClassifier()
        assert model is not None
        assert model._fitted is False

    def test_fit_predict(self, sample_data):
        X, y = sample_data
        from src.models.baseline import XGBoostFailureClassifier
        model = XGBoostFailureClassifier()
        model.fit(X, y)
        assert model._fitted
        preds = model.predict(X[:10])
        assert len(preds) == 10
        assert set(preds).issubset({0, 1})

    def test_predict_proba(self, sample_data):
        X, y = sample_data
        from src.models.baseline import XGBoostFailureClassifier
        model = XGBoostFailureClassifier()
        model.fit(X, y)
        proba = model.predict_proba(X[:10])
        assert len(proba) == 10
        assert (proba >= 0).all() and (proba <= 1).all()


class TestRandomForestModel:
    """Tests for RandomForestFailureClassifier."""

    def test_import(self):
        from src.models.baseline import RandomForestFailureClassifier
        model = RandomForestFailureClassifier()
        assert model is not None

    def test_fit_predict(self, sample_data):
        X, y = sample_data
        from src.models.baseline import RandomForestFailureClassifier
        model = RandomForestFailureClassifier()
        model.fit(X, y)
        preds = model.predict(X[:5])
        assert len(preds) == 5


class TestLSTMModel:
    """Tests for LSTM model."""

    def test_import(self):
        import torch
        from src.models.lstm_model import LSTMFailureClassifier
        model = LSTMFailureClassifier(input_size=4, hidden_size=32, num_layers=1)
        assert model is not None
        assert model.input_size == 4

    def test_forward_pass(self):
        import torch
        from src.models.lstm_model import LSTMFailureClassifier
        model = LSTMFailureClassifier(input_size=4, hidden_size=32, num_layers=1)
        x = torch.randn(8, 48, 4)  # batch=8, seq=48, features=4
        out = model(x)
        assert out.shape == (8, 1)

    def test_lstm_trainer(self, sample_sequences):
        X, y = sample_sequences
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from src.models.lstm_model import LSTMFailureClassifier, LSTMTrainer

        model = LSTMFailureClassifier(input_size=4, hidden_size=16, num_layers=1, bidirectional=False)

        dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        loader = DataLoader(dataset, batch_size=16)

        trainer = LSTMTrainer(model, device="cpu", pos_weight=5.0, learning_rate=1e-3)
        history = trainer.fit(loader, loader, epochs=2, verbose=False)

        assert "best_epoch" in history
        assert len(trainer.train_losses) == 2


@pytest.mark.smoke
def test_evaluate_model(sample_data):
    """Smoke test: evaluate_model function."""
    X, y = sample_data
    from src.models.baseline import XGBoostFailureClassifier, evaluate_model

    model = XGBoostFailureClassifier()
    model.fit(X[:400], y[:400], X_val=X[400:], y_val=y[400:])

    metrics = evaluate_model(model, X[400:], y[400:], "Test")
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert 0 <= metrics["f1_score"] <= 1
