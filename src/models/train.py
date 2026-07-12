"""
Unified training pipeline for all predictive maintenance models.

Usage:
    python -m src.models.train --model all        # Train all models
    python -m src.models.train --model xgboost    # XGBoost only
    python -m src.models.train --model lstm       # LSTM only
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import mlflow
from ..utils.config import config
from ..utils.logger import logger, setup_logger
from ..data.loader import AzurePdMLoader
from ..data.preprocessor import PdMPreprocessor
from ..data.feature_engineer import FeatureEngineer
from ..models.baseline import (
    XGBoostFailureClassifier,
    RandomForestFailureClassifier,
    evaluate_model,
)
from ..models.lstm_model import (
    LSTMFailureClassifier,
    LSTMTrainer,
    create_dataloaders,
)
from ..evaluation.metrics import compute_all_metrics, compare_models
from ..evaluation.cost_analysis import simulate_maintenance_costs, generate_cost_comparison_chart
from ..evaluation.visualization import generate_all_plots


def load_and_prepare_data():
    """Load data and prepare features for all models."""
    logger.info("=" * 60)
    logger.info("DATA PIPELINE")
    logger.info("=" * 60)

    # Load
    loader = AzurePdMLoader()
    df = loader.load_all()
    stats = loader.get_statistics()
    logger.info(f"Dataset stats: {stats}")

    # Feature engineering
    engineer = FeatureEngineer()
    df_features = engineer.fit_transform(df)

    # Identify feature columns (exclude identifiers and targets)
    exclude_cols = [
        "machine_id", "timestamp", "failure_label", "failure_components",
        "failure_comp1", "failure_comp2", "failure_comp3", "failure_comp4",
        "time_to_failure_hours", "machine_model",
    ]
    feature_cols = [c for c in df_features.columns
                    if c not in exclude_cols
                    and df_features[c].dtype != object
                    and not df_features[c].isna().all()]

    # Prepare X, y
    preprocessor = PdMPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(
        df_features, target_col="failure_label"
    )

    # Ensure feature alignment
    feature_cols = [c for c in feature_cols if c in df_features.columns]
    X_train_named = df_features.iloc[:len(X_train)][feature_cols].values
    X_test_named = df_features.iloc[len(X_train):][feature_cols].values

    if X_train_named.shape[1] != X_train.shape[1]:
        logger.warning("Feature mismatch — using direct split from preprocessor")

    return X_train, X_test, y_train, y_test, feature_cols, df_features


def train_xgboost(X_train, X_test, y_train, y_test, feature_names):
    """Train and evaluate XGBoost model."""
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING: XGBOOST")
    logger.info("=" * 60)

    with mlflow.start_run(run_name="xgboost_pdm"):
        # Log parameters
        mlflow.log_params(config.model.xgboost_params)

        # Train
        model = XGBoostFailureClassifier()
        model.fit(X_train, y_train, X_val=X_test, y_val=y_test, feature_names=feature_names)

        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = compute_all_metrics(y_test, y_pred, y_proba, "XGBoost")

        # Log metrics
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

        # Save
        model.save()
        mlflow.sklearn.log_model(model.model, "xgboost_model")

    return model, metrics, y_pred, y_proba


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train and evaluate Random Forest model."""
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING: RANDOM FOREST")
    logger.info("=" * 60)

    with mlflow.start_run(run_name="random_forest_pdm"):
        mlflow.log_params(config.model.random_forest_params)

        model = RandomForestFailureClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = compute_all_metrics(y_test, y_pred, y_proba, "RandomForest")

        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        model.save()
        mlflow.sklearn.log_model(model.model, "rf_model")

    return model, metrics, y_pred, y_proba


def train_lstm(X_train, X_test, y_train, y_test, feature_names):
    """Train and evaluate LSTM model."""
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING: LSTM")
    logger.info("=" * 60)

    lstm_cfg = config.model.lstm_params
    n_features = X_train.shape[1]

    # Build sequences
    preprocessor = PdMPreprocessor()
    X_seq, y_seq, meta = preprocessor.build_sequences(
        pd.concat([
            pd.DataFrame(X_train, columns=feature_names[:n_features]),
            pd.DataFrame(X_test, columns=feature_names[:n_features]),
        ]),
        sequence_length=lstm_cfg["sequence_length"],
    )

    split_idx = int(len(X_seq) * 0.8)
    X_train_seq, X_test_seq = X_seq[:split_idx], X_seq[split_idx:]
    y_train_seq, y_test_seq = y_seq[:split_idx], y_seq[split_idx:]

    train_loader, val_loader, test_loader = create_dataloaders(
        X_train_seq, y_train_seq, X_test_seq, y_test_seq,
        batch_size=lstm_cfg["batch_size"],
    )

    with mlflow.start_run(run_name="lstm_pdm"):
        mlflow.log_params(lstm_cfg)

        model = LSTMFailureClassifier(
            input_size=n_features,
            hidden_size=lstm_cfg["hidden_size"],
            num_layers=lstm_cfg["num_layers"],
            dropout=lstm_cfg["dropout"],
            bidirectional=lstm_cfg["bidirectional"],
            attention_heads=lstm_cfg["attention_heads"],
        )

        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        pos_weight = len(y_train_seq[y_train_seq == 0]) / max(len(y_train_seq[y_train_seq == 1]), 1)

        trainer = LSTMTrainer(
            model, device=device, pos_weight=pos_weight,
            learning_rate=lstm_cfg["learning_rate"],
            weight_decay=lstm_cfg["weight_decay"],
            patience=lstm_cfg["patience"],
        )

        history = trainer.fit(train_loader, val_loader, epochs=lstm_cfg["epochs"])

        # Evaluate
        y_proba = trainer.predict(X_test_seq)
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = compute_all_metrics(y_test_seq, y_pred, y_proba, "LSTM")

        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        trainer.save()

    return trainer, metrics, y_pred, y_proba


def main():
    parser = argparse.ArgumentParser(description="Train PdM models")
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["all", "xgboost", "random_forest", "rf", "lstm"],
        help="Which model to train",
    )
    parser.add_argument(
        "--skip-lstm", action="store_true",
        help="Skip LSTM training (faster for quick iterations)",
    )
    args = parser.parse_args()

    setup_logger()

    # Set MLflow tracking
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("predictive_maintenance_pdm")

    # Load data
    X_train, X_test, y_train, y_test, feature_names, df = load_and_prepare_data()

    all_metrics = []

    # XGBoost
    if args.model in ["all", "xgboost"]:
        xgb_model, xgb_metrics, y_pred_xgb, y_proba_xgb = train_xgboost(
            X_train, X_test, y_train, y_test, feature_names
        )
        all_metrics.append(xgb_metrics)

    # Random Forest
    if args.model in ["all", "random_forest", "rf"]:
        rf_model, rf_metrics, y_pred_rf, y_proba_rf = train_random_forest(
            X_train, X_test, y_train, y_test
        )
        all_metrics.append(rf_metrics)

    # LSTM
    if args.model in ["all", "lstm"] and not args.skip_lstm:
        lstm_trainer, lstm_metrics, y_pred_lstm, y_proba_lstm = train_lstm(
            X_train, X_test, y_train, y_test, feature_names
        )
        all_metrics.append(lstm_metrics)

    # Model comparison
    if len(all_metrics) > 1:
        compare_models(all_metrics, output_path=Path("reports/model_comparison.csv"))

    # Generate visualizations for the best model
    best_metrics = max(all_metrics, key=lambda m: m["f1_score"])
    logger.info(f"\n🏆 Best model: {best_metrics['model']} (F1: {best_metrics['f1_score']:.4f})")

    # Cost analysis
    logger.info("\n" + "=" * 60)
    logger.info("COST ANALYSIS")
    logger.info("=" * 60)
    cost_results = simulate_maintenance_costs(
        n_machines=100,
        model_recall=best_metrics["recall"],
        model_precision=best_metrics["precision"],
    )
    generate_cost_comparison_chart(cost_results)

    logger.info("\n✅ Training complete! Summary:")
    logger.info(f"  Models trained: {[m['model'] for m in all_metrics]}")
    logger.info(f"  Best F1: {best_metrics['f1_score']:.4f} ({best_metrics['model']})")
    logger.info(f"  ROC-AUC: {best_metrics['roc_auc']:.4f}")
    logger.info(f"  Expected annual savings: ${cost_results['roi']['savings_vs_reactive_3yr']/3:,.0f}")


if __name__ == "__main__":
    main()
