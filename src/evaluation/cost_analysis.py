"""
Business cost analysis & ROI simulation for Predictive Maintenance.

Simulates maintenance cost outcomes under three strategies:
  1. Reactive (run-to-failure)
  2. Preventive (time-based schedules)
  3. Predictive (ML-driven, this project's approach)

Generates ROI analysis charts and business case data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from ..utils.config import config
from ..utils.logger import logger


@dataclass
class MaintenanceCosts:
    """Cost parameters for maintenance strategies."""
    planned_maintenance_cost: float = 2500.0      # Scheduled repair/replace
    emergency_repair_cost: float = 15000.0         # Unscheduled emergency repair
    hourly_downtime_cost: float = 8000.0           # Lost production per hour
    planned_downtime_hours: float = 4.0            # Planned repair duration
    emergency_downtime_hours: float = 12.0          # Emergency repair duration
    inspection_cost: float = 500.0                  # Proactive inspection
    sensor_hardware_cost: float = 20000.0           # Per-machine IoT kit
    software_platform_cost: float = 50000.0         # Annual platform fee
    data_scientist_cost: float = 120000.0           # Annual personnel


def simulate_maintenance_costs(
    n_machines: int = 100,
    n_years: int = 3,
    failure_rate_per_year: float = 0.15,
    false_alarm_rate: float = 0.05,
    model_recall: float = 0.90,
    model_precision: float = 0.85,
) -> Dict[str, Dict]:
    """
    Simulate total cost of ownership under three maintenance strategies.

    Returns cost breakdown for each strategy.
    """
    costs = MaintenanceCosts()

    # --- Strategy 1: Reactive Maintenance ---
    # All failures result in emergency repairs
    n_failures_per_year = int(n_machines * failure_rate_per_year)
    reactive_annual = (
        n_failures_per_year * costs.emergency_repair_cost
        + n_failures_per_year * costs.emergency_downtime_hours * costs.hourly_downtime_cost
    )
    reactive_total = reactive_annual * n_years

    reactive = {
        "strategy": "Reactive (Run-to-Failure)",
        "annual_cost": reactive_annual,
        "total_cost_3yr": reactive_total,
        "n_emergency_repairs_per_year": n_failures_per_year,
        "downtime_hours_per_year": n_failures_per_year * costs.emergency_downtime_hours,
    }

    # --- Strategy 2: Preventive (Time-Based) ---
    # Schedule maintenance every N hours for ALL machines
    maintenance_interval_hours = 720  # Every 30 days
    inspections_per_year = int(8760 / maintenance_interval_hours) * n_machines
    preventive_annual = (
        inspections_per_year * costs.planned_maintenance_cost
        + inspections_per_year * costs.planned_downtime_hours * costs.hourly_downtime_cost
        # Still some unexpected failures
        + n_failures_per_year * 0.3 * costs.emergency_repair_cost  # 70% reduction
        + n_failures_per_year * 0.3 * costs.emergency_downtime_hours * costs.hourly_downtime_cost
    )
    preventive_total = preventive_annual * n_years

    preventive = {
        "strategy": "Preventive (Time-Based)",
        "annual_cost": preventive_annual,
        "total_cost_3yr": preventive_total,
        "inspections_per_year": inspections_per_year,
        "n_emergency_repairs_per_year": int(n_failures_per_year * 0.3),
        "downtime_hours_per_year": (
            inspections_per_year * costs.planned_downtime_hours
            + n_failures_per_year * 0.3 * costs.emergency_downtime_hours
        ),
    }

    # --- Strategy 3: Predictive (ML-Driven) ---
    # Only act when model predicts failure
    true_positives = n_failures_per_year * model_recall
    false_positives = n_failures_per_year * (1 - model_precision) / model_precision * model_recall

    predictive_annual = (
        # Planned repairs for correctly predicted failures
        true_positives * costs.planned_maintenance_cost
        + true_positives * costs.planned_downtime_hours * costs.hourly_downtime_cost
        # Emergency repairs for missed failures
        + n_failures_per_year * (1 - model_recall) * costs.emergency_repair_cost
        + n_failures_per_year * (1 - model_recall) * costs.emergency_downtime_hours * costs.hourly_downtime_cost
        # Cost of false alarms (unnecessary inspections)
        + false_positives * costs.inspection_cost
        # Platform + hardware costs (amortized over 3 years)
        + (costs.sensor_hardware_cost * n_machines / n_years)
        + costs.software_platform_cost
        + costs.data_scientist_cost
    )
    predictive_total = predictive_annual * n_years

    predictive = {
        "strategy": "Predictive (ML-Driven)",
        "annual_cost": predictive_annual,
        "total_cost_3yr": predictive_total,
        "true_positives_per_year": true_positives,
        "false_positives_per_year": false_positives,
        "missed_failures_per_year": n_failures_per_year * (1 - model_recall),
        "downtime_hours_per_year": (
            true_positives * costs.planned_downtime_hours
            + n_failures_per_year * (1 - model_recall) * costs.emergency_downtime_hours
        ),
    }

    # --- ROI ---
    savings_vs_reactive = reactive_total - predictive_total
    roi = (savings_vs_reactive / (costs.sensor_hardware_cost * n_machines + costs.software_platform_cost * n_years + costs.data_scientist_cost * n_years)) * 100

    roi_summary = {
        "savings_vs_reactive_3yr": savings_vs_reactive,
        "savings_vs_preventive_3yr": preventive_total - predictive_total,
        "roi_percentage": roi,
        "payback_months": 12 * (costs.sensor_hardware_cost * n_machines + costs.software_platform_cost) / (reactive_annual - predictive_annual) if reactive_annual > predictive_annual else 0,
    }

    return {
        "reactive": reactive,
        "preventive": preventive,
        "predictive": predictive,
        "roi": roi_summary,
    }


def generate_cost_comparison_chart(
    results: Dict,
    output_path: Optional[Path] = None,
):
    """Generate a bar chart comparing maintenance costs across strategies."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_path = output_path or Path("reports/figures/cost_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    strategies = ["reactive", "preventive", "predictive"]
    labels = ["Reactive\n(Run-to-Failure)", "Preventive\n(Time-Based)", "Predictive\n(ML-Driven) 🏆"]
    annual_costs = [results[s]["annual_cost"] for s in strategies]
    total_costs = [results[s]["total_cost_3yr"] for s in strategies]
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Annual Cost
    bars1 = ax1.bar(labels, [c / 1e6 for c in annual_costs], color=colors, edgecolor="white", linewidth=2)
    ax1.set_ylabel("Annual Cost ($M)")
    ax1.set_title("Annual Maintenance Cost by Strategy")
    for bar, val in zip(bars1, annual_costs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"${val/1e6:.2f}M", ha="center", fontweight="bold", fontsize=12)
    ax1.grid(axis="y", alpha=0.3)

    # 3-Year Total
    bars2 = ax2.bar(labels, [c / 1e6 for c in total_costs], color=colors, edgecolor="white", linewidth=2)
    ax2.set_ylabel("3-Year Total Cost ($M)")
    ax2.set_title("3-Year Total Cost of Ownership")
    for bar, val in zip(bars2, total_costs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"${val/1e6:.2f}M", ha="center", fontweight="bold", fontsize=12)
    ax2.grid(axis="y", alpha=0.3)

    # ROI annotation
    roi = results["roi"]
    ax2.annotate(
        f"ROI: {roi['roi_percentage']:.0f}%\n"
        f"Payback: {roi['payback_months']:.1f} months\n"
        f"Savings: ${roi['savings_vs_reactive_3yr']/1e6:.1f}M vs Reactive",
        xy=(2, total_costs[2] / 1e6),
        xytext=(2, max(total_costs) / 1e6 * 0.85),
        fontsize=11, ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="gray", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Cost comparison chart saved to {output_path}")


def generate_roi_sensitivity_analysis(
    n_machines: int = 100,
    output_path: Optional[Path] = None,
):
    """
    Generate sensitivity analysis: how ROI changes with model performance.
    """
    import matplotlib.pyplot as plt

    output_path = output_path or Path("reports/figures/roi_sensitivity.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    recall_range = np.arange(0.5, 1.0, 0.05)
    precision_range = np.arange(0.5, 1.0, 0.05)

    roi_matrix = np.zeros((len(recall_range), len(precision_range)))

    for i, recall in enumerate(recall_range):
        for j, precision in enumerate(precision_range):
            results = simulate_maintenance_costs(
                n_machines=n_machines,
                model_recall=recall,
                model_precision=precision,
            )
            roi_matrix[i, j] = results["roi"]["roi_percentage"]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(roi_matrix, aspect="auto", origin="lower",
                   extent=[precision_range[0], precision_range[-1],
                           recall_range[0], recall_range[-1]],
                   cmap="RdYlGn")
    ax.set_xlabel("Model Precision")
    ax.set_ylabel("Model Recall")
    ax.set_title(f"ROI Sensitivity: PdM for {n_machines} Machines (3-Year Projection)")

    # Annotate cells
    for i in range(len(recall_range)):
        for j in range(len(precision_range)):
            color = "white" if roi_matrix[i, j] < 300 else "black"
            ax.text(precision_range[j], recall_range[i],
                   f"{roi_matrix[i,j]:.0f}%", ha="center", va="center",
                   fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="ROI (%)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROI sensitivity analysis saved to {output_path}")


if __name__ == "__main__":
    logger.info("Running cost analysis simulation...")
    results = simulate_maintenance_costs(n_machines=100)

    print("\n" + "=" * 60)
    print("MAINTENANCE COST SIMULATION")
    print("=" * 60)

    for strategy in ["reactive", "preventive", "predictive"]:
        s = results[strategy]
        print(f"\n{s['strategy']}:")
        print(f"  Annual Cost: ${s['annual_cost']:,.0f}")
        print(f"  3-Year Cost: ${s['total_cost_3yr']:,.0f}")

    roi = results["roi"]
    print(f"\n💰 ROI Analysis:")
    print(f"  Savings vs Reactive:  ${roi['savings_vs_reactive_3yr']:,.0f}")
    print(f"  Savings vs Preventive: ${roi['savings_vs_preventive_3yr']:,.0f}")
    print(f"  ROI: {roi['roi_percentage']:.0f}%")
    print(f"  Payback Period: {roi['payback_months']:.1f} months")

    generate_cost_comparison_chart(results)
    generate_roi_sensitivity_analysis()
