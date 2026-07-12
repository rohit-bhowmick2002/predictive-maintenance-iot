"""
Comprehensive evaluation metrics for predictive maintenance models.

Includes:
  - Standard classification metrics
  - Cost-sensitive (asymmetric) scoring
  - PHM-specific metrics (prognostic horizon, early prediction score)
  - Model comparison framework
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    brier_score_loss,
)
from ..utils.config import config
from ..utils.logger import logger


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Compute a comprehensive set of evaluation metrics.

    Returns a flat dictionary suitable for MLflow logging.
    """
    metrics = {}

    # Basic metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics["n_samples"] = len(y_true)
    metrics["n_positive"] = int(y_true.sum())
    metrics["n_negative"] = int((1 - y_true).sum())

    # Classification metrics
    metrics["accuracy"] = (tp + tn) / (tp + tn + fp + fn)
    metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics["f1_score"] = f1_score(y_true, y_pred)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    metrics["fnr"] = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Probability-based metrics
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["avg_precision"] = average_precision_score(y_true, y_proba)
        metrics["brier_score"] = brier_score_loss(y_true, y_proba)
    else:
        metrics["roc_auc"] = 0.5
        metrics["avg_precision"] = 0.0
        metrics["brier_score"] = 1.0

    # Cost-sensitive metrics
    eval_cfg = config.evaluation
    metrics["cost_total"] = (
        fp * eval_cfg.cost_false_positive
        + fn * eval_cfg.cost_false_negative
        + tp * eval_cfg.cost_true_positive
        + tn * eval_cfg.cost_true_negative
    )
    metrics["cost_per_sample"] = metrics["cost_total"] / len(y_true)

    # PHM scoring function (custom)
    metrics["phm_score"] = _compute_phm_score(y_true, y_proba, y_pred)

    # Model name for tracking
    metrics["model"] = model_name

    return metrics


def _compute_phm_score(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    early_penalty: float = 0.1,
    late_penalty: float = 1.0,
) -> float:
    """
    Prognostics and Health Management (PHM) scoring function.

    Penalizes late predictions (missed failures) more heavily than early predictions
    (false alarms). A custom scoring function widely used in the PHM community.

    Score = 1 - (early_penalty * FP_rate + late_penalty * FN_rate)
    Higher is better (max = 1.0 for perfect predictions).
    """
    fp_rate = ((y_pred == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)
    fn_rate = ((y_pred == 0) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1)

    score = 1 - (early_penalty * fp_rate + late_penalty * fn_rate) / (early_penalty + late_penalty)
    return max(0.0, score)


def compute_prognostic_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    time_to_failure: Optional[np.ndarray] = None,
    prediction_horizon: int = 24,
) -> Dict[str, float]:
    """
    Compute prognostic-specific metrics.

    Args:
        y_true: Ground truth labels
        y_proba: Predicted failure probabilities
        time_to_failure: Hours until actual failure (optional)
        prediction_horizon: Hours ahead we're predicting
    """
    metrics = {}

    # Early prediction score: how far in advance can we detect?
    if time_to_failure is not None:
        detected = y_proba >= 0.5
        true_failures = y_true == 1

        if true_failures.sum() > 0:
            detection_times = time_to_failure[detected & true_failures]
            if len(detection_times) > 0:
                metrics["mean_detection_lead_time"] = detection_times.mean()
                metrics["median_detection_lead_time"] = np.median(detection_times)
                metrics["detection_rate"] = detected[true_failures].mean()

    # Prognostic horizon: distance from first reliable detection to failure
    if time_to_failure is not None and len(y_proba) > 1:
        # Simulate: find first timestep where probability exceeds threshold
        threshold = 0.5
        for t in range(len(y_proba) - 1, -1, -1):
            if y_true[t] == 1 and y_proba[t] >= threshold:
                metrics["prognostic_horizon_hours"] = time_to_failure[t]
                break

    return metrics


def compare_models(
    model_results: List[Dict[str, float]],
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compare multiple models side by side.

    Args:
        model_results: List of metric dictionaries from compute_all_metrics()
        output_path: Optional path to save CSV

    Returns:
        Comparison DataFrame
    """
    df = pd.DataFrame(model_results)

    # Select key columns for display
    display_cols = [
        "model", "accuracy", "precision", "recall", "f1_score",
        "mcc", "roc_auc", "avg_precision", "cost_total", "cost_per_sample",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    comparison = df[display_cols].set_index("model").sort_values("f1_score", ascending=False)

    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 80)
    logger.info("\n" + comparison.to_string())

    if output_path:
        comparison.to_csv(output_path)
        logger.info(f"\nComparison saved to {output_path}")

    return comparison


def compute_threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Analyze model performance across different probability thresholds.

    Useful for business decision: what threshold balances false alarms vs. missed failures?
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)

    rows = []
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        metrics = compute_all_metrics(y_true, y_pred, y_proba)
        metrics["threshold"] = thresh
        rows.append(metrics)

    df = pd.DataFrame(rows)
    return df[["threshold", "precision", "recall", "f1_score", "cost_total", "fpr", "fnr"]]


def plot_evaluation_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_dir: Optional[Path] = None,
    prefix: str = "",
):
    """Generate and save ROC curve, PR curve, and calibration plot."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir = output_dir or Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{prefix}_" if prefix else ""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    axes[0].plot(fpr, tpr, "b-", linewidth=2, label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].fill_between(fpr, tpr, alpha=0.1, color="blue")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    axes[1].plot(recall, precision, "g-", linewidth=2, label=f"AP = {ap:.3f}")
    axes[1].fill_between(recall, precision, alpha=0.1, color="green")
    baseline = y_true.mean()
    axes[1].axhline(y=baseline, color="r", linestyle="--", alpha=0.5, label=f"Baseline = {baseline:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    # Calibration Plot
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    axes[2].plot(prob_pred, prob_true, "o-", linewidth=2, color="purple")
    axes[2].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[2].set_xlabel("Mean Predicted Probability")
    axes[2].set_ylabel("Fraction of Positives")
    axes[2].set_title("Calibration Plot (Reliability Diagram)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = output_dir / f"{prefix}evaluation_curves.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Evaluation curves saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    y_true = np.random.randint(0, 2, 1000)
    y_proba = np.clip(y_true + np.random.normal(0, 0.2, 1000), 0, 1)
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = compute_all_metrics(y_true, y_pred, y_proba, "Example")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
