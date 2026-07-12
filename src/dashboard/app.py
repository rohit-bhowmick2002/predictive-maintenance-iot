"""
Streamlit Dashboard — Predictive Maintenance Fleet Health Monitor.

Features:
  - Real-time KPI cards (Availability, MTBF, OEE, Failure Rate, Cost)
  - Fleet health heatmap
  - Machine-level detail view with sensor trends
  - Failure probability timeline
  - Maintenance cost tracker
  - Alert summary

Run with: streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random

# Page config
st.set_page_config(
    page_title="PdM Fleet Monitor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1a237e; margin-bottom: 1rem; }
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 20px; color: white;
        text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .kpi-value { font-size: 2rem; font-weight: 800; }
    .kpi-label { font-size: 0.85rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-good { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .kpi-warning { background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%); }
    .kpi-critical { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }
    .metric-card {
        background: white; border-radius: 8px; padding: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Mock Data Generator (replace with actual model predictions)
# ============================================================
@st.cache_data(ttl=60)
def generate_mock_data():
    """Generate realistic mock data for dashboard demonstration."""
    np.random.seed(42)
    machines = list(range(1, 101))
    dates = pd.date_range(start="2015-12-01", end="2015-12-31", freq="1h")

    # Fleet KPIs
    kpis = {
        "availability": 0.943 + np.random.normal(0, 0.01),
        "mtbf_hours": 1247 + np.random.normal(0, 50),
        "oee": 0.827 + np.random.normal(0, 0.01),
        "schedule_compliance": 0.88 + np.random.normal(0, 0.02),
        "false_alarm_rate": 0.048 + np.random.normal(0, 0.005),
        "maintenance_cost_mtd": 187000 + np.random.normal(0, 5000),
        "active_alarms": int(12 + np.random.normal(0, 3)),
        "machines_at_risk": int(18 + np.random.normal(0, 4)),
    }

    # Machine health data
    health_data = []
    for m in machines:
        base_health = np.random.beta(5, 1.5)
        if m in [17, 42, 89, 3, 55, 71]:
            base_health = np.random.beta(2, 3)
        health_data.append({
            "machine_id": f"M-{m:03d}",
            "health_score": round(base_health, 3),
            "risk_level": "🔴 High" if base_health < 0.3 else ("🟡 Medium" if base_health < 0.6 else "🟢 Low"),
            "last_failure_days": int(np.random.exponential(80)),
            "sensor_anomalies_7d": int(np.random.poisson(2 if base_health < 0.5 else 0.5)),
        })

    health_df = pd.DataFrame(health_data)

    # Time series — failure probability for top at-risk machine
    hours = 168
    prob_series = 0.05 + np.cumsum(np.random.normal(0.002, 0.01, hours))
    prob_series = np.clip(prob_series, 0, 1)
    prob_df = pd.DataFrame({
        "hours_ahead": list(range(hours, 0, -1)),
        "failure_probability": prob_series,
    })

    return kpis, health_df, prob_df


kpis, health_df, prob_df = generate_mock_data()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/factory.png", width=60)
    st.markdown("## 🏭 PdM Control Panel")

    st.markdown("---")
    view_mode = st.radio("View Mode", ["Fleet Overview", "Machine Detail", "Cost Analytics", "Alerts"])

    if view_mode == "Machine Detail":
        selected_machine = st.selectbox("Select Machine", sorted(health_df["machine_id"].tolist()))

    st.markdown("---")
    st.markdown("### ⚙️ Model Configuration")
    threshold = st.slider("Alert Threshold", 0.0, 1.0, 0.5, 0.05)
    pred_horizon = st.selectbox("Prediction Horizon", [6, 12, 24, 48, 72], index=2)

    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption("Data: Microsoft Azure PdM Dataset")
    st.caption("Models: XGBoost + LSTM Ensemble")


# ============================================================
# MAIN CONTENT
# ============================================================
st.markdown('<p class="main-header">🏭 Predictive Maintenance — Fleet Health Monitor</p>', unsafe_allow_html=True)

# ---- FLEET OVERVIEW ----
if view_mode == "Fleet Overview":
    # KPI Row
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    kpi_configs = [
        ("Availability", f"{kpis['availability']:.1%}", "kpi-good" if kpis["availability"] > 0.95 else "kpi-warning"),
        ("MTBF (Hrs)", f"{kpis['mtbf_hours']:,.0f}", "kpi-good" if kpis["mtbf_hours"] > 1200 else "kpi-warning"),
        ("OEE", f"{kpis['oee']:.1%}", "kpi-good" if kpis["oee"] > 0.85 else "kpi-warning"),
        ("MTD Cost", f"${kpis['maintenance_cost_mtd']/1000:.0f}K", "kpi-good" if kpis['maintenance_cost_mtd'] < 200000 else "kpi-warning"),
        ("Active Alarms", str(kpis["active_alarms"]), "kpi-critical" if kpis["active_alarms"] > 10 else "kpi-good"),
        ("At-Risk Machines", str(kpis["machines_at_risk"]), "kpi-critical" if kpis["machines_at_risk"] > 15 else "kpi-warning"),
    ]

    for col, (label, value, css_class) in zip([col1, col2, col3, col4, col5, col6], kpi_configs):
        with col:
            st.markdown(f'<div class="kpi-card {css_class}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Charts row
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📊 Fleet Health Distribution")
        fig = px.histogram(
            health_df, x="health_score", nbins=30,
            color="risk_level",
            color_discrete_map={"🔴 High": "#ef5350", "🟡 Medium": "#ff9800", "🟢 Low": "#4caf50"},
            title="Machine Health Score Distribution",
            labels={"health_score": "Health Score (0=Critical, 1=Healthy)", "count": "Number of Machines"},
        )
        fig.add_vline(x=0.3, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
        fig.add_vline(x=0.6, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("⚠️ Top Risk Machines")
        at_risk = health_df.nsmallest(10, "health_score")[["machine_id", "health_score", "sensor_anomalies_7d"]]
        at_risk.columns = ["Machine", "Health Score", "Anomalies (7d)"]

        def color_health(val):
            if val < 0.3: return 'color: #ef5350; font-weight: bold'
            elif val < 0.6: return 'color: #ff9800; font-weight: bold'
            return 'color: #4caf50'

        st.dataframe(
            at_risk.style.map(color_health, subset=["Health Score"]).format({"Health Score": "{:.1%}"}),
            use_container_width=True, height=350,
        )

    # Full-width heatmap
    st.markdown("---")
    st.subheader("🗺️ Fleet Health Heatmap")

    # Pivot for heatmap
    heatmap_data = health_df.copy()
    heatmap_data["machine_group"] = (heatmap_data["machine_id"].str.extract(r"(\d+)")[0].astype(int) // 10) * 10
    heatmap_pivot = heatmap_data.pivot_table(
        index="machine_group", columns=None, values="health_score", aggfunc="mean"
    ).sort_index(ascending=False)

    fig = go.Figure(data=go.Heatmap(
        z=[health_df.sort_values("machine_id")["health_score"].values.reshape(10, 10)],
        y=[f"M-{i+1:03d}" for i in range(0, 100, 10)],
        x=[str(i) for i in range(10)],
        colorscale="RdYlGn", zmin=0, zmax=1,
        colorbar=dict(title="Health Score"),
    ))
    fig.update_layout(height=400, title="Machine Health Matrix (10×10 Grid)")
    st.plotly_chart(fig, use_container_width=True)

    # Failure probability for at-risk fleet
    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📈 Top Machine — Failure Probability Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=prob_df["hours_ahead"], y=prob_df["failure_probability"],
            fill="tozeroy", fillcolor="rgba(239, 83, 80, 0.2)",
            line=dict(color="#ef5350", width=2),
            name="Failure Probability",
        ))
        fig.add_hline(y=threshold, line_dash="dash", line_color="orange", annotation_text="Alert Threshold")
        fig.add_hline(y=0.7, line_dash="dot", line_color="red", annotation_text="Critical Threshold")
        fig.update_layout(
            xaxis_title="Hours Ahead", yaxis_title="Failure Probability",
            yaxis_range=[0, 1], title="M-017: Predicted Failure Timeline",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("🔄 Maintenance Strategy Mix")
        strategy_data = pd.DataFrame({
            "Strategy": ["Reactive (Current)", "Preventive", "Predictive (ML)", "Optimal"],
            "Annual Cost ($K)": [1110, 780, 540, 420],
        })
        fig = px.bar(
            strategy_data, x="Strategy", y="Annual Cost ($K)", color="Strategy",
            color_discrete_sequence=["#ef5350", "#ff9800", "#4caf50", "#2196f3"],
            text="Annual Cost ($K)",
        )
        fig.update_traces(texttemplate="$%{text}K", textposition="outside")
        fig.update_layout(showlegend=False, title="Annual Maintenance Cost by Strategy")
        st.plotly_chart(fig, use_container_width=True)


# ---- MACHINE DETAIL ----
elif view_mode == "Machine Detail":
    machine_row = health_df[health_df["machine_id"] == selected_machine].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Health Score", f"{machine_row['health_score']:.1%}", delta="-2%")
    with col2:
        st.metric("Risk Level", machine_row["risk_level"])
    with col3:
        st.metric("Last Failure", f"{machine_row['last_failure_days']} days ago")
    with col4:
        st.metric("Anomalies (7d)", machine_row["sensor_anomalies_7d"])

    st.markdown("---")

    # Simulated sensor trends
    hours = 168
    dates = pd.date_range(end=datetime.now(), periods=hours, freq="1h")
    sensor_data = pd.DataFrame({
        "timestamp": dates,
        "vibration": 30 + np.cumsum(np.random.normal(0.02, 0.1, hours)) + np.sin(np.linspace(0, 8*np.pi, hours)) * 2,
        "voltage": 170 + np.cumsum(np.random.normal(-0.005, 0.05, hours)),
        "rotation": 450 + np.random.normal(0, 5, hours),
        "pressure": 100 + np.cumsum(np.random.normal(0.01, 0.08, hours)),
    })

    sensor_selection = st.multiselect(
        "Select Sensors", ["vibration", "voltage", "rotation", "pressure"],
        default=["vibration", "voltage"],
    )

    fig = go.Figure()
    colors = {"vibration": "#ef5350", "voltage": "#2196f3", "rotation": "#4caf50", "pressure": "#ff9800"}
    for sensor in sensor_selection:
        fig.add_trace(go.Scatter(
            x=sensor_data["timestamp"], y=sensor_data[sensor],
            name=sensor.title(), line=dict(color=colors[sensor], width=1.5),
        ))
    fig.update_layout(
        title=f"{selected_machine} — Sensor Trends (Last 7 Days)",
        xaxis_title="Time", yaxis_title="Sensor Value",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Failure probability
    st.subheader("🔮 Predicted Failure Probability")
    machine_prob = pd.DataFrame({
        "hours_ahead": list(range(72, 0, -1)),
        "probability": np.clip(np.cumsum(np.random.normal(0.003, 0.008, 72)) + machine_row["health_score"] * 0.3, 0.01, 0.99),
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=machine_prob["hours_ahead"], y=machine_prob["probability"],
        fill="tozeroy", line=dict(color="#ff9800", width=3),
    ))
    fig.update_layout(xaxis_title="Hours Ahead", yaxis_title="Failure Probability", yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)


# ---- COST ANALYTICS ----
elif view_mode == "Cost Analytics":
    st.subheader("💰 Maintenance Cost Analysis")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MTD Cost", f"${kpis['maintenance_cost_mtd']:,.0f}", delta="-$12,500")
    with col2:
        st.metric("YTD Cost", "$2.14M", delta="-$340K vs Budget")
    with col3:
        st.metric("Projected Annual", "$2.24M", delta="-$560K with PdM")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        months = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        cost_data = pd.DataFrame({
            "Month": months,
            "Reactive": [185, 178, 190, 172, 168, 165],
            "Preventive": [80, 85, 82, 88, 90, 92],
            "Predictive": [60, 58, 55, 52, 50, 48],
        })
        fig = px.bar(
            cost_data, x="Month", y=["Reactive", "Preventive", "Predictive"],
            title="Monthly Maintenance Cost Breakdown ($K)",
            barmode="group",
            color_discrete_sequence=["#ef5350", "#ff9800", "#4caf50"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        cost_by_comp = pd.DataFrame({
            "Component": ["Comp1", "Comp2", "Comp3", "Comp4"],
            "Emergency Cost": [320, 480, 210, 340],
            "Planned Cost": [80, 120, 60, 90],
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(x=cost_by_comp["Component"], y=cost_by_comp["Emergency Cost"],
                            name="Emergency", marker_color="#ef5350"))
        fig.add_trace(go.Bar(x=cost_by_comp["Component"], y=cost_by_comp["Planned Cost"],
                            name="Planned", marker_color="#4caf50"))
        fig.update_layout(title="Cost by Component ($K)", barmode="stack")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 ROI Projection — 3 Year Outlook")

    years = ["Year 1", "Year 2", "Year 3"]
    roi_data = pd.DataFrame({
        "Year": years,
        "Reactive": [2.4, 2.5, 2.6],
        "Preventive": [2.0, 1.95, 1.9],
        "Predictive": [1.8, 1.5, 1.3],
    })
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=roi_data["Reactive"], mode="lines+markers",
                            name="Reactive", line=dict(color="#ef5350", width=3, dash="dash")))
    fig.add_trace(go.Scatter(x=years, y=roi_data["Preventive"], mode="lines+markers",
                            name="Preventive", line=dict(color="#ff9800", width=3)))
    fig.add_trace(go.Scatter(x=years, y=roi_data["Predictive"], mode="lines+markers",
                            name="Predictive (ML)", line=dict(color="#4caf50", width=4)))
    fig.update_layout(title="Cumulative Cost Projection ($M)", yaxis_title="$M")
    st.plotly_chart(fig, use_container_width=True)


# ---- ALERTS ----
elif view_mode == "Alerts":
    st.subheader("🚨 Active Alerts & Notifications")

    alerts = [
        {"severity": "🔴 CRITICAL", "machine": "M-017", "message": "Failure probability 89% — comp2 predicted failure in 18h", "time": "10 min ago"},
        {"severity": "🔴 CRITICAL", "machine": "M-042", "message": "Vibration spike detected — 4.2σ above baseline", "time": "25 min ago"},
        {"severity": "🟡 WARNING", "machine": "M-089", "message": "Voltage drift detected — 2.8σ above baseline", "time": "1 hour ago"},
        {"severity": "🟡 WARNING", "machine": "M-003", "message": "Pressure anomaly in comp3 — trend analysis suggests inspection", "time": "2 hours ago"},
        {"severity": "🟡 WARNING", "machine": "M-055", "message": "Degradation rate increasing — 15% above expected", "time": "3 hours ago"},
        {"severity": "🔵 INFO", "machine": "M-071", "message": "Scheduled maintenance due in 48 hours", "time": "4 hours ago"},
        {"severity": "🔵 INFO", "machine": "M-012", "message": "Model retrained — accuracy improved to 94.2%", "time": "6 hours ago"},
        {"severity": "🔵 INFO", "machine": "FLEET", "message": "Data quality check passed — 99.8% completeness", "time": "8 hours ago"},
    ]

    for alert in alerts:
        color = "#ffebee" if "CRITICAL" in alert["severity"] else ("#fff3e0" if "WARNING" in alert["severity"] else "#e3f2fd")
        st.markdown(f"""
        <div style="background:{color}; padding:12px; border-radius:8px; margin-bottom:8px; border-left:4px solid {'#ef5350' if 'CRITICAL' in alert['severity'] else ('#ff9800' if 'WARNING' in alert['severity'] else '#2196f3')};">
            <strong>{alert['severity']}</strong> | {alert['machine']}<br>
            <span style="font-size:0.9em;">{alert['message']}</span>
            <span style="float:right; opacity:0.7; font-size:0.8em;">{alert['time']}</span>
        </div>
        """, unsafe_allow_html=True)


# Footer
st.markdown("---")
st.caption("⚡ Predictive Maintenance Dashboard | Data: Azure PdM | Models: XGBoost + LSTM Ensemble | Refresh: 30s")
