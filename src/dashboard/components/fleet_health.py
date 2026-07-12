"""
Fleet health visualization components for the PdM Dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Optional


def render_fleet_health_overview(
    machines: List[str],
    health_scores: List[float],
    risk_levels: List[str],
):
    """
    Render the fleet health overview section.

    Args:
        machines: List of machine identifiers
        health_scores: Corresponding health scores (0-1)
        risk_levels: Risk level strings ("Low", "Medium", "High")
    """
    df = pd.DataFrame({
        "machine": machines,
        "health_score": health_scores,
        "risk": risk_levels,
    })

    col1, col2, col3 = st.columns(3)

    with col1:
        healthy_count = sum(1 for r in risk_levels if r == "Low")
        st.metric("Healthy Machines", f"{healthy_count}/{len(machines)}",
                 delta=f"{healthy_count/len(machines):.0%}")

    with col2:
        warning_count = sum(1 for r in risk_levels if r == "Medium")
        st.metric("At Risk", f"{warning_count}/{len(machines)}")

    with col3:
        critical_count = sum(1 for r in risk_levels if r == "High")
        st.metric("Critical", f"{critical_count}/{len(machines)}",
                 delta="⚠️" if critical_count > 0 else "✅")

    # Distribution chart
    fig = px.histogram(
        df, x="health_score", nbins=30,
        color="risk",
        color_discrete_map={"Low": "#4caf50", "Medium": "#ff9800", "High": "#ef5350"},
        title="Machine Health Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_health_timeline(
    dates: List[str],
    scores: List[float],
    machine_id: str = "Fleet Average",
):
    """Render a health score timeline."""
    df = pd.DataFrame({"date": pd.to_datetime(dates), "score": scores})

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["score"],
        mode="lines", fill="tozeroy",
        line=dict(color="#2196f3", width=2),
        name="Health Score",
    ))
    fig.add_hline(y=0.3, line_dash="dash", line_color="red",
                  annotation_text="Critical")
    fig.add_hline(y=0.6, line_dash="dash", line_color="orange",
                  annotation_text="Warning")
    fig.update_layout(
        title=f"Health Score Timeline — {machine_id}",
        yaxis_range=[0, 1],
        yaxis_title="Health Score",
        xaxis_title="Date",
    )
    st.plotly_chart(fig, use_container_width=True)
