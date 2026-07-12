"""
Streamlit KPI Card Components for the PdM Dashboard.

Reusable card components that display key performance indicators
with color-coded status indicators and trend arrows.
"""

import streamlit as st
from typing import Optional


def kpi_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    status: str = "good",
    help_text: Optional[str] = None,
):
    """
    Render a styled KPI card.

    Args:
        label: KPI name (e.g., "Availability")
        value: Display value (e.g., "94.3%")
        delta: Change indicator (e.g., "+1.2%")
        status: "good" (green), "warning" (orange), "critical" (red)
        help_text: Optional tooltip
    """
    colors = {
        "good": "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)",
        "warning": "linear-gradient(135deg, #f2994a 0%, #f2c94c 100%)",
        "critical": "linear-gradient(135deg, #eb3349 0%, #f45c43 100%)",
        "neutral": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    }

    bg = colors.get(status, colors["neutral"])

    delta_html = ""
    if delta:
        delta_color = "lightgreen" if delta.startswith("+") else "salmon"
        delta_html = f'<div style="font-size:0.9rem; color:{delta_color}">{delta}</div>'

    help_attr = f'title="{help_text}"' if help_text else ""

    st.markdown(f"""
    <div style="background:{bg}; border-radius:12px; padding:18px;
                color:white; text-align:center; box-shadow:0 4px 12px rgba(0,0,0,0.1);
                margin-bottom:8px;" {help_attr}>
        <div style="font-size:0.75rem; opacity:0.9; text-transform:uppercase; letter-spacing:1.5px;
                    margin-bottom:4px;">{label}</div>
        <div style="font-size:1.6rem; font-weight:800;">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def metric_row(metrics: dict, cols_per_row: int = 4):
    """
    Render a row of KPI cards.

    Args:
        metrics: dict mapping label -> {value, delta, status}
        cols_per_row: Number of cards per row
    """
    cols = st.columns(cols_per_row)
    items = list(metrics.items())

    for i, (label, config) in enumerate(items):
        col_idx = i % cols_per_row
        with cols[col_idx]:
            kpi_card(
                label=label,
                value=config.get("value", "—"),
                delta=config.get("delta"),
                status=config.get("status", "good"),
                help_text=config.get("help"),
            )


def alert_banner(severity: str, message: str, machine_id: Optional[str] = None):
    """
    Render an alert banner.

    Args:
        severity: "critical", "warning", or "info"
        message: Alert message text
        machine_id: Optional machine identifier
    """
    icons = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
    colors = {
        "critical": ("#ffebee", "#ef5350"),
        "warning": ("#fff3e0", "#ff9800"),
        "info": ("#e3f2fd", "#2196f3"),
    }
    bg, border = colors.get(severity, colors["info"])
    icon = icons.get(severity, "ℹ️")

    machine_text = f" | {machine_id}" if machine_id else ""

    st.markdown(f"""
    <div style="background:{bg}; padding:12px; border-radius:8px; margin-bottom:8px;
                border-left:4px solid {border};">
        <strong>{icon} {severity.upper()}{machine_text}</strong><br>
        <span style="font-size:0.9em;">{message}</span>
    </div>
    """, unsafe_allow_html=True)
