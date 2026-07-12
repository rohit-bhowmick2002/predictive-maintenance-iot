"""
Tree-based baseline models for failure classification.

Models:
  1. XGBoost — Gradient boosting with Optuna hyperparameter optimization
  2. Random Forest — Ensemble of decision trees with balanced class weighting

Features:
  - SHAP-based feature importance
  - Threshold optimization for cost-sensitive decision making
  - MLflow experiment tracking
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import shap
from ..utils.config import config
from ..utils.logger import logger


class XGBoostFailureClassifier:
    """
    XGBoost classifier optimized for imbalanced failure prediction.

    Uses asymmetric cost-sensitive threshold tuning.
    """

    def __init__(self, params: Optional[Dict] = None):
        self.params = params or config.model.xgboost_params
        self.model: Optional[xgb.XGBClassifier] = None
        self.optimal_threshold: float = 0.5
        self.feature_importance: Optional[pd.DataFrame] = None
        self._fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None,
    ) -> "XGBoostFailureClassifier":
        """Train XGBoost with early stopping."""
        logger.info("Training XGBoost classifier...")

        pos_weight = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1
        self.params["scale_pos_weight"] = pos_weight

        self.model = xgb.XGBClassifier(**self.params, use_label_encoder=False)

        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train)

        # Tune threshold for cost-sensitive optimization
        self._tune_threshold(X_val if X_val is not None else X_train,
                             y_val if y_val is not None else y_train)

        # SHAP feature importance
        if feature_names:
            self._compute_shap_importance(X_train[:min(5000, len(X_train))], feature_names)

        self._fitted = True
        logger.info(f"✅ XGBoost trained. Optimal threshold: {self.optimal_threshold:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with cost-optimized threshold."""
        proba = self.predict_proba(X)
        return (proba >= self.optimal_threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return failure probabilities."""
        return self.model.predict_proba(X)[:, 1]

    def _tune_threshold(self, X: np.ndarray, y: np.ndarray) -> None:
        """Find optimal threshold minimizing expected cost."""
        proba = self.model.predict_proba(X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, proba)

        # Cost function: cost_FP * FP_rate + cost_FN * FN_rate
        best_cost = float("inf")
        best_thresh = 0.5

        for i, thresh in enumerate(thresholds):
            if i >= len(precisions):
                break
            # Estimate confusion matrix at this threshold
            preds = (proba >= thresh).astype(int)
            fp = ((preds == 1) & (y == 0)).sum()
            fn = ((preds == 0) & (y == 1)).sum()

            total_cost = (
                fp * config.evaluation.cost_false_positive
                + fn * config.evaluation.cost_false_negative
            )
            if total_cost < best_cost:
                best_cost = total_cost
                best_thresh = thresh

        self.optimal_threshold = best_thresh
        logger.info(f"  Cost-optimal threshold: {best_thresh:.4f} (cost: {best_cost:,.0f})")

    def _compute_shap_importance(self, X: np.ndarray, feature_names: list) -> None:
        """Compute SHAP feature importance values."""
        logger.info("  Computing SHAP values...")
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X[:min(2000, len(X))])

            self.feature_importance = pd.DataFrame({
                "feature": feature_names,
                "importance": np.abs(shap_values).mean(axis=0),
            }).sort_values("importance", ascending=False)

            # Save SHAP summary
            shap.summary_plot(
                shap_values, X[:min(2000, len(X))],
                feature_names=feature_names,
                show=False,
                max_display=20,
            )
            import matplotlib.pyplot as plt
            plt.tight_layout()
            plt.savefig(config.data.processed_dir.parent / "reports/figures/shap_summary_xgboost.png",
                       dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("  SHAP summary saved to reports/figures/")
        except Exception as e:
            logger.warning(f"  SHAP computation skipped: {e}")

    def save(self, path: Optional[Path] = None) -> None:
        path = path or config.data.models_dir / "xgboost_model.joblib"
        joblib.dump({
            "model": self.model,
            "threshold": self.optimal_threshold,
            "feature_importance": self.feature_importance,
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: Optional[Path] = None) -> "XGBoostFailureClassifier":
        path = path or config.data.models_dir / "xgboost_model.joblib"
        data = joblib.load(path)
        self.model = data["model"]
        self.optimal_threshold = data["threshold"]
        self.feature_importance = data.get("feature_importance")
        self._fitted = True
        return self


class RandomForestFailureClassifier:
    """
    Random Forest with balanced class weighting for failure classification.
    """

    def __init__(self, params: Optional[Dict] = None):
        self.params = params or config.model.random_forest_params
        self.model: Optional[RandomForestClassifier] = None
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RandomForestFailureClassifier":
        logger.info("Training Random Forest classifier...")
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X_train, y_train)
        self._fitted = True
        logger.info("✅ Random Forest trained")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Optional[Path] = None) -> None:
        path = path or config.data.models_dir / "random_forest_model.joblib"
        joblib.dump(self.model, path)

    def load(self, path: Optional[Path] = None) -> "RandomForestFailureClassifier":
        path = path or config.data.models_dir / "random_forest_model.joblib"
        self.model = joblib.load(path)
        self._fitted = True
        return self


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Comprehensive evaluation with business-relevant metrics.

    Returns dict of metrics for reporting.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = {
        "model": model_name,
        "accuracy": (y_pred == y_test).mean(),
        "precision": (y_pred[y_pred == 1] == y_test[y_pred == 1]).mean()
        if y_pred.sum() > 0 else 0,
        "recall": (y_pred[y_test == 1] == 1).mean()
        if y_test.sum() > 0 else 0,
        "f1_score": f1_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5,
        "avg_precision": average_precision_score(y_test, y_proba),
    }

    # Compute expected cost
    fp = ((y_pred == 1) & (y_test == 0)).sum()
    fn = ((y_pred == 0) & (y_test == 1)).sum()
    tp = ((y_pred == 1) & (y_test == 1)).sum()
    tn = ((y_pred == 0) & (y_test == 0)).sum()

    metrics["expected_cost"] = (
        fp * config.evaluation.cost_false_positive
        + fn * config.evaluation.cost_false_negative
        + tp * config.evaluation.cost_true_positive
    )
    metrics["cost_per_sample"] = metrics["expected_cost"] / len(y_test)
    metrics["fp_count"] = fp
    metrics["fn_count"] = fn
    metrics["tp_count"] = tp
    metrics["tn_count"] = tn

    # Log
    logger.info(f"\n{'='*50}")
    logger.info(f"  {model_name} Evaluation")
    logger.info(f"{'='*50}")
    logger.info(f"  Accuracy:         {metrics['accuracy']:.4f}")
    logger.info(f"  Precision:        {metrics['precision']:.4f}")
    logger.info(f"  Recall:           {metrics['recall']:.4f}")
    logger.info(f"  F1-Score:         {metrics['f1_score']:.4f}")
    logger.info(f"  MCC:              {metrics['mcc']:.4f}")
    logger.info(f"  ROC-AUC:          {metrics['roc_auc']:.4f}")
    logger.info(f"  Avg Precision:    {metrics['avg_precision']:.4f}")
    logger.info(f"  False Positives:  {fp}")
    logger.info(f"  False Negatives:  {fn}")
    logger.info(f"  Expected Cost:    ${metrics['expected_cost']:,.2f}")
    logger.info(f"  Cost/Sample:      ${metrics['cost_per_sample']:,.2f}")

    return metrics
