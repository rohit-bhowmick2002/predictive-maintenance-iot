"""
Visualization module for predictive maintenance results.

Generates:
  - Confusion matrices with cost annotations
  - Model comparison charts
  - Sensor degradation plots
  - Fleet health heatmaps
  - Precision-Recall vs threshold plots
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from ..utils.config import config
from ..utils.logger import logger

# Style
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})
COLORS = sns.color_palette("husl", 8)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    output_path: Optional[Path] = None,
    cost_matrix: Optional[Dict] = None,
):
    """Plot confusion matrix with optional cost annotations."""
    from sklearn.metrics import confusion_matrix as cm_func

    cm = cm_func(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Healthy (0)", "Failure (1)"],
                yticklabels=["Healthy (0)", "Failure (1)"],
                ax=ax, annot_kws={"fontsize": 16, "fontweight": "bold"})

    # Add cost labels
    eval_cfg = config.evaluation
    ax.text(0.5, 0.5, f"TN\n${eval_cfg.cost_true_negative:,.0f}",
            ha="center", va="center", fontsize=10, color="green")
    ax.text(1.5, 0.5, f"FP\n${eval_cfg.cost_false_positive:,.0f}",
            ha="center", va="center", fontsize=10, color="orange")
    ax.text(0.5, 1.5, f"FN\n${eval_cfg.cost_false_negative:,.0f}",
            ha="center", va="center", fontsize=10, color="red")
    ax.text(1.5, 1.5, f"TP\n${eval_cfg.cost_true_positive:,.0f}",
            ha="center", va="center", fontsize=10, color="green")

    total_cost = (fp * eval_cfg.cost_false_positive + fn * eval_cfg.cost_false_negative
                  + tp * eval_cfg.cost_true_positive + tn * eval_cfg.cost_true_negative)
    ax.set_title(f"{title}\nTotal Cost: ${total_cost:,.0f}")

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.savefig(Path("reports/figures/confusion_matrix.png"), dpi=150, bbox_inches="tight")
        plt.close()


def plot_model_comparison(
    model_metrics: List[Dict],
    output_path: Optional[Path] = None,
):
    """Side-by-side bar chart comparing multiple models."""
    output_path = output_path or Path("reports/figures/model_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(model_metrics)
    models = df["model"].tolist()
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

    x = np.arange(len(models))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        values = [m.get(metric, 0) for m in model_metrics]
        bars = ax.bar(x + i * width, values, width, label=label, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                   f"{val:.3f}", ha="center", fontsize=8, rotation=90)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Classification Metrics")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Model comparison chart saved to {output_path}")


def plot_degradation_curves(
    df: pd.DataFrame,
    machine_id: int,
    failure_timestamp: pd.Timestamp,
    hours_before: int = 168,
    output_path: Optional[Path] = None,
):
    """
    Plot sensor degradation leading up to a failure.

    Shows all 4 sensor channels with trend lines and anomaly markers.
    """
    machine_data = df[df["machine_id"] == machine_id].copy()
    window_start = failure_timestamp - pd.Timedelta(hours=hours_before)
    window_data = machine_data[
        (machine_data["timestamp"] >= window_start)
        & (machine_data["timestamp"] <= failure_timestamp)
    ]

    sensor_cols = config.data.sensor_cols
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    for i, col in enumerate(sensor_cols):
        if col in window_data.columns:
            ax = axes[i]
            ax.plot(window_data["timestamp"], window_data[col], color=COLORS[i], linewidth=1.5, alpha=0.8)

            # Add trend line
            if len(window_data) > 10:
                z = np.polyfit(range(len(window_data)), window_data[col], 1)
                p = np.poly1d(z)
                ax.plot(window_data["timestamp"], p(range(len(window_data))),
                       "--", color="red", linewidth=1.5, alpha=0.6, label="Trend")

            ax.set_ylabel(col.replace("_", " ").title())
            ax.legend(loc="upper left", fontsize=9)
            ax.grid(True, alpha=0.3)

            # Mark failure point
            ax.axvline(x=failure_timestamp, color="red", linestyle=":", linewidth=2, alpha=0.7)

    axes[-1].set_xlabel("Timestamp")
    axes[0].set_title(f"Sensor Degradation — Machine {machine_id} | {hours_before}h Before Failure")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))

    plt.tight_layout()
    out = output_path or Path(f"reports/figures/degradation_machine_{machine_id}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Degradation plot saved to {out}")


def plot_fleet_health_heatmap(
    health_scores: pd.DataFrame,
    output_path: Optional[Path] = None,
):
    """
    Fleet health heatmap: machines × time showing health scores.

    Args:
        health_scores: DataFrame with columns [machine_id, timestamp, health_score]
    """
    output_path = output_path or Path("reports/figures/fleet_health_heatmap.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pivot = health_scores.pivot_table(
        index="machine_id",
        columns=health_scores["timestamp"].dt.date,
        values="health_score",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(
        pivot, cmap="RdYlGn_r", center=0.5, vmin=0, vmax=1,
        cbar_kws={"label": "Failure Risk"},
        ax=ax, linewidths=0.5, linecolor="white",
    )
    ax.set_title("Fleet Health Heatmap — 100 Machines", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Machine ID")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Fleet health heatmap saved to {output_path}")


def plot_threshold_optimization(
    thresholds: np.ndarray,
    precision: np.ndarray,
    recall: np.ndarray,
    costs: np.ndarray,
    output_path: Optional[Path] = None,
):
    """Plot precision, recall, and cost vs threshold."""
    output_path = output_path or Path("reports/figures/threshold_optimization.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(thresholds, precision, "b-", linewidth=2, label="Precision")
    ax1.plot(thresholds, recall, "g-", linewidth=2, label="Recall")
    ax1.set_xlabel("Probability Threshold")
    ax1.set_ylabel("Score")
    ax1.legend(loc="center left")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, costs, "r--", linewidth=2, label="Expected Cost ($)")
    ax2.set_ylabel("Expected Cost ($)")
    ax2.legend(loc="center right")

    # Mark optimal threshold
    opt_idx = np.argmin(costs)
    ax1.axvline(thresholds[opt_idx], color="red", linestyle=":", alpha=0.7)
    ax1.annotate(f"Optimal: {thresholds[opt_idx]:.2f}",
                 xy=(thresholds[opt_idx], costs[opt_idx]),
                 fontsize=11, fontweight="bold",
                 bbox=dict(facecolor="white", edgecolor="red", alpha=0.8))

    ax1.set_title("Threshold Optimization: Precision, Recall & Cost Trade-off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Threshold optimization plot saved to {output_path}")


def generate_all_plots(
    model_results: Optional[List[Dict]] = None,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    y_proba: Optional[np.ndarray] = None,
):
    """Generate the full suite of visualization plots."""
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating visualization suite...")

    # Plot 1: Confusion matrix
    if y_true is not None and y_pred is not None:
        plot_confusion_matrix(y_true, y_pred, output_path=output_dir / "confusion_matrix.png")

    # Plot 2: Model comparison
    if model_results:
        plot_model_comparison(model_results, output_path=output_dir / "model_comparison.png")

    # Plot 3: Threshold optimization
    if y_true is not None and y_proba is not None:
        thresholds = np.arange(0.05, 1.0, 0.05)
        precision_list, recall_list, cost_list = [], [], []

        for t in thresholds:
            preds = (y_proba >= t).astype(int)
            tp = ((preds == 1) & (y_true == 1)).sum()
            fp = ((preds == 1) & (y_true == 0)).sum()
            fn = ((preds == 0) & (y_true == 1)).sum()

            precision_list.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
            recall_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            cost_list.append(fp * 500 + fn * 12000 + tp * 500)

        plot_threshold_optimization(
            thresholds, np.array(precision_list), np.array(recall_list),
            np.array(cost_list), output_path=output_dir / "threshold_optimization.png",
        )

    logger.info(f"✅ All plots saved to {output_dir}/")
