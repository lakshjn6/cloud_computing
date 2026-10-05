"""
Phase 4 — Streamlit Interactive Dashboard
==========================================
Directly addresses every clause of the problem statement:

  Section A — Workload Patterns        (historical CPU & Memory traces)
  Section B — Predictions              (ARIMA / XGBoost / LSTM forecasts)
  Section C — VM Provisioning Decisions (per-step scale-up / scale-down log)
  Section D — Resource Utilization     (cluster CPU % & host count over time)

Run:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

# ── Page config (must be first Streamlit call) ─────────────────────────
st.set_page_config(
    page_title="VM Resource Provisioning — Cloud AI Dashboard",
    page_icon="cloud",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Gradient page title */
.page-title {
    font-size: 2.2rem; font-weight: 700; line-height: 1.25;
    background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #34D399 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: .2rem;
}
.page-sub { color: #64748B; font-size: 1rem; margin-bottom: 1.8rem; }

/* Section header banners */
.section-banner {
    background: linear-gradient(90deg, rgba(56,189,248,0.15), rgba(129,140,248,0.08));
    border-left: 4px solid #38BDF8;
    border-radius: 0 8px 8px 0;
    padding: .6rem 1rem;
    margin-bottom: 1rem;
    font-size: 1.05rem; font-weight: 600; color: #E2E8F0;
}

/* KPI cards */
.kpi { background: rgba(255,255,255,0.04);
       border: 1px solid rgba(255,255,255,0.09);
       border-radius: 12px; padding: 1rem 1.2rem;
       box-shadow: 0 4px 24px rgba(0,0,0,.25); margin-bottom:.8rem; }
.kpi-label { font-size:.78rem; color:#94A3B8; font-weight:600;
             text-transform:uppercase; letter-spacing:.06em; }
.kpi-val   { font-size:1.65rem; font-weight:700; color:#F1F5F9; margin-top:.2rem; }
.kpi-note  { font-size:.78rem; color:#38BDF8; margin-top:.15rem; }

/* Sidebar */
[data-testid="stSidebar"] { background: rgba(10,18,35,0.9); }

/* Buttons */
div.stButton > button {
    background: linear-gradient(135deg,#2563EB,#4F46E5);
    color:#fff; border:none; border-radius:8px;
    padding:.5rem 1.3rem; font-weight:600;
    transition: all .2s ease;
}
div.stButton > button:hover {
    transform:translateY(-1px);
    box-shadow:0 6px 20px rgba(79,70,229,.4);
}

/* Decision log table colour helpers */
.scale-up   { color: #34D399 !important; font-weight:600; }
.scale-down { color: #F87171 !important; font-weight:600; }
.sla-viol   { color: #FB923C !important; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ──────────────────────────────────────────────────────
def pick(sim, label, metric):
    """sim mein se label (AI / Baseline) aur metric (sla / wast) ke hisaab se key dhoondhta hai."""
    lab = label.lower()
    base_words = ("react", "baseline", "static", "traditional")
    is_base = any(w in lab for w in base_words)
    words = base_words if is_base else ("ai", "predict", "proactive")
    keys = list(sim.keys())
    found = [k for k in keys
             if metric in str(k).lower() and any(w in str(k).lower() for w in words)]
    if not found:
        st.error(f"'{metric}' key not found for '{label}'. Available keys:")
        st.write(keys)
        st.stop()
    return sim[found[0]]


@st.cache_data
def load_raw(path="1.csv"):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "Timestamp [ms]" in df.columns:
        unit = "ms" if df["Timestamp [ms]"].iloc[0] > 1e12 else "s"
        df["datetime"] = pd.to_datetime(df["Timestamp [ms]"], unit=unit)
    elif "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"])
    else:
        df["datetime"] = pd.date_range("2013-08-12", periods=len(df), freq="5min")
    return df.sort_values("datetime").reset_index(drop=True)

@st.cache_data
def load_art(path="model_artifacts.joblib"):
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_data
def load_sim(path="simulation_results.joblib"):
    return joblib.load(path) if os.path.exists(path) else None


# ── Pipeline trigger ────────────────────────────────────────────────────
def run_pipeline():
    with st.spinner("Step 1/3 — Preprocessing 1.csv ..."):
        from data_prep import WorkloadDataPreprocessor
        WorkloadDataPreprocessor(filepath="1.csv").process_and_save()
    with st.spinner("Step 2/3 — Training ARIMA, XGBoost, LSTM ..."):
        from train_models import run_training_pipeline
        run_training_pipeline()
    with st.spinner("Step 3/3 — Running SimPy cloud simulation ..."):
        from cloud_sim import compare_predictive_vs_reactive
        compare_predictive_vs_reactive()
    st.success("Pipeline complete! Refreshing ...")
    st.cache_data.clear()
    st.rerun()


# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Control Panel")
    st.markdown("---")
    if st.button("Run / Re-train Full Pipeline", use_container_width=True):
        run_pipeline()
    st.markdown("---")
    st.markdown("### Chart Settings")
    win = st.slider("Display window (time steps)", 50, 600, 200, 25)
    models_to_show = st.multiselect(
        "Forecast models",
        ["ARIMA", "XGBoost", "LSTM"],
        default=["ARIMA", "XGBoost", "LSTM"],
    )
    sim_mode = st.radio("Simulation mode to display", ["Predictive (AI)", "Reactive (Baseline)"])
    st.markdown("---")
    st.caption("1 step = 5 minutes  |  Dataset: Bitbrains 1.csv")


# ── Load data ───────────────────────────────────────────────────────────
raw_df = load_raw()
art    = load_art()
sim    = load_sim()


# ── Page header ─────────────────────────────────────────────────────────
st.markdown('<div class="page-title">Predictive VM Resource Provisioning System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">Machine Learning Forecasting · Cloud Simulation · '
    'SLA-Aware Proactive Autoscaling</div>',
    unsafe_allow_html=True,
)

# ── Top KPI row ─────────────────────────────────────────────────────────
best_model  = art["best_model_name"]        if art else "—"
best_rmse   = float(art["metrics"].loc[art["metrics"]["Model"]==best_model,"RMSE"].values[0]) if art else 0
sla_pred    = sim["Predictive_SLA_Violations"] if sim else 0
sla_reac    = sim["Reactive_SLA_Violations"]   if sim else 0
sla_saved   = max(0, sla_reac - sla_pred)
cpu_saved   = sim["Reactive_Wasted_CPU"] - sim["Predictive_Wasted_CPU"] if sim else 0
true_pos    = sim["Predictive_TruePos"] if sim else 0

k1, k2, k3, k4, k5 = st.columns(5)
for col, label, val, note in [
    (k1, "Best Forecast Model",      best_model,            "Lowest RMSE on test set"),
    (k2, "Forecast RMSE",            f"{best_rmse:.3f} %",  "Mean squared deviation"),
    (k3, "AI SLA Violations",        str(sla_pred),         f"Reactive has {sla_reac}"),
    (k4, "CPU Waste Saved",          f"{abs(cpu_saved):.0f}","vs Reactive controller"),
    (k5, "Proactive True-Positives", str(true_pos),         "Correct early scale-ups"),
]:
    with col:
        st.markdown(
            f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-val">{val}</div>'
            f'<div class="kpi-note">{note}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════
# SECTION A — WORKLOAD PATTERNS
# ════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-banner">SECTION A — Historical Workload Patterns '
    '(CPU &amp; Memory from Bitbrains Dataset)</div>',
    unsafe_allow_html=True,
)

if raw_df is not None:
    disp = raw_df.tail(win * 12)   # up to win*12 rows (12 = steps per hour)

    cpu_col = "CPU usage [%]"     if "CPU usage [%]"     in raw_df.columns else None
    mem_col = "Memory usage [KB]" if "Memory usage [KB]" in raw_df.columns else None

    if cpu_col:
        fig_a = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=("CPU Utilisation (%)", "Memory Utilisation (%)"),
            vertical_spacing=0.08,
        )
        fig_a.add_trace(go.Scatter(
            x=disp["datetime"], y=disp[cpu_col],
            name="CPU Usage (%)", line=dict(color="#38BDF8", width=1.8),
            fill="tozeroy", fillcolor="rgba(56,189,248,0.07)",
        ), row=1, col=1)

        if mem_col:
            mem_pct = disp[mem_col] / raw_df[mem_col].max() * 100.0
            fig_a.add_trace(go.Scatter(
                x=disp["datetime"], y=mem_pct,
                name="Memory Usage (norm %)", line=dict(color="#818CF8", width=1.8),
                fill="tozeroy", fillcolor="rgba(129,140,248,0.07)",
            ), row=2, col=1)

        fig_a.update_layout(
            template="plotly_dark", height=400, showlegend=True,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
        )
        st.plotly_chart(fig_a, use_container_width=True)

        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        col_stat1.metric("Avg CPU Usage",  f"{raw_df[cpu_col].mean():.1f}%")
        col_stat2.metric("Peak CPU Usage", f"{raw_df[cpu_col].max():.1f}%")
        col_stat3.metric("Min CPU Usage",  f"{raw_df[cpu_col].min():.1f}%")
        col_stat4.metric("Total Records",  f"{len(raw_df):,} rows")

        with st.expander("View raw dataset sample (first 100 rows)"):
            st.dataframe(raw_df.head(100), use_container_width=True)
else:
    st.warning("Dataset '1.csv' not found. Place it in the project folder.")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════
# SECTION B — ML PREDICTIONS (Forecasting)
# ════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-banner">SECTION B — ML Model Predictions vs Actual CPU Demand '
    '(ARIMA · XGBoost · LSTM)</div>',
    unsafe_allow_html=True,
)

if art is None:
    st.info("No trained models found. Click 'Run / Re-train Full Pipeline' in the sidebar.")
else:
    metrics_df  = art["metrics"]
    y_actual    = art["y_actual"]
    preds_dict  = art["predictions"]

    # ── Metrics table ──
    col_m, col_c = st.columns([1, 2])
    with col_m:
        st.markdown("##### Model Evaluation Leaderboard")
        st.caption("Lower RMSE / MAE / MAPE = better prediction accuracy")

        # Highlight best RMSE row
        def hl(row):
            best = metrics_df["RMSE"].min()
            return ["background-color:#1E3A5F;font-weight:700" if row["RMSE"]==best else ""]*len(row)

        st.dataframe(
            metrics_df.style.apply(hl, axis=1).format({"RMSE":"{:.4f}","MAE":"{:.4f}","MAPE (%)":"{:.2f}"}),
            use_container_width=True,
            height=160,
        )

        st.markdown("**What these metrics mean:**")
        st.caption("• **RMSE** — Root Mean Squared Error (penalises large errors heavily)")
        st.caption("• **MAE** — Mean Absolute Error (average prediction error in %CPU)")
        st.caption("• **MAPE** — Mean Absolute % Error (relative accuracy)")

    with col_c:
        st.markdown("##### Actual vs Forecasted CPU Utilisation (Test Window)")
        st.caption(
            f"Showing last {min(win, len(y_actual))} steps of the held-out test set. "
            "The model is predicting CPU demand it has never seen during training."
        )

        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(
            y=y_actual[-win:],
            name="Actual CPU Demand",
            line=dict(color="#F1F5F9", width=2.5, dash="dot"),
        ))

        colour_map = {"ARIMA": "#FB923C", "XGBoost": "#34D399", "LSTM": "#818CF8"}
        for m in models_to_show:
            if m in preds_dict:
                fig_b.add_trace(go.Scatter(
                    y=preds_dict[m][-win:],
                    name=f"{m} Prediction",
                    line=dict(color=colour_map[m], width=2),
                ))

        fig_b.update_layout(
            template="plotly_dark", height=320,
            xaxis_title="Time Step (each = 5 min)",
            yaxis_title="CPU Utilisation (%)",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_b, use_container_width=True)

    # Saved matplotlib chart
    if os.path.exists("model_performance_comparison.png"):
        with st.expander("View static comparison chart (saved by train_models.py)"):
            st.image("model_performance_comparison.png", use_container_width=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════
# SECTION C — VM PROVISIONING DECISIONS
# ════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-banner">SECTION C — VM Provisioning Decisions '
    '(SimPy Cloud Data Centre Simulation)</div>',
    unsafe_allow_html=True,
)

if sim is None:
    st.warning("No simulation results found. Run the pipeline first.")
else:
    df_show = sim["df_predictive"] if "Predictive" in sim_mode else sim["df_reactive"]
    mode_label = "AI Predictive Controller" if "Predictive" in sim_mode else "Reactive Baseline"

    st.caption(
        f"**{mode_label}** — Each row is one 5-minute decision tick. "
        "The controller decides how many physical hosts to keep active based on "
        "predicted (or current) CPU demand. VMs run on top of these hosts."
    )

    # ── Provisioning timeline chart ──
    fig_c = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=(
            "CPU Demand vs Provisioned Cluster Capacity (%)",
            "Active Physical Host Count (VM Pool Size)",
        ),
        vertical_spacing=0.1,
    )

    twindow = df_show.tail(win)

    # Demand vs capacity
    fig_c.add_trace(go.Scatter(
        y=twindow["actual_demand_pct"],
        name="Actual CPU Demand (%)",
        line=dict(color="#38BDF8", width=2),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.06)",
    ), row=1, col=1)

    fig_c.add_trace(go.Scatter(
        y=twindow["predicted_demand_pct"],
        name="Predicted CPU Demand (%)",
        line=dict(color="#34D399", width=1.6, dash="dash"),
    ), row=1, col=1)

    fig_c.add_trace(go.Scatter(
        y=twindow["cluster_capacity_pct"],
        name="Provisioned Capacity (%)",
        line=dict(color="#F59E0B", width=2),
        line_shape="hv",
    ), row=1, col=1)

    # SLA violations as red markers
    sla_rows = twindow[twindow["sla_violation"]]
    if len(sla_rows):
        fig_c.add_trace(go.Scatter(
            x=sla_rows.index - twindow.index[0],
            y=sla_rows["actual_demand_pct"],
            mode="markers",
            marker=dict(color="#EF4444", size=9, symbol="x"),
            name="SLA Violation",
        ), row=1, col=1)

    # Host count
    fig_c.add_trace(go.Scatter(
        y=twindow["active_hosts"],
        name="Active Hosts",
        line=dict(color="#A78BFA", width=2),
        line_shape="hv",
        fill="tozeroy", fillcolor="rgba(167,139,250,0.07)",
    ), row=2, col=1)

    fig_c.add_trace(go.Scatter(
        y=twindow["active_vms"],
        name="Active VMs",
        line=dict(color="#F472B6", width=1.5, dash="dot"),
        line_shape="hv",
    ), row=2, col=1)

    fig_c.update_layout(
        template="plotly_dark", height=480,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.06, x=1, xanchor="right"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_c, use_container_width=True)

    # ── Scaling decision summary ──
    st.markdown("##### Scaling Decision Breakdown")
    decision_counts = twindow["scaling_action"].value_counts().reset_index()
    decision_counts.columns = ["Decision", "Count"]

    col_dc, col_log = st.columns([1, 2])
    with col_dc:
        st.dataframe(decision_counts, use_container_width=True, hide_index=True)
        sla_count = int(twindow["sla_violation"].sum())
        scale_up  = int(twindow["scaling_action"].str.startswith("SCALE_UP").sum())
        scale_dn  = int(twindow["scaling_action"].str.startswith("SCALE_DOWN").sum())
        st.metric("Scale-UP events",   scale_up)
        st.metric("Scale-DOWN events", scale_dn)
        st.metric("SLA Violations",    sla_count, delta=f"{sla_count} breaches", delta_color="inverse")

    with col_log:
        st.markdown("##### Live Decision Log (last 60 steps)")
        st.caption("SCALE_UP = more hosts added | SCALE_DOWN = hosts removed | HOLD = no change")

        log_disp = twindow.tail(60)[[
            "sim_time_min", "actual_demand_pct", "predicted_demand_pct",
            "active_hosts", "active_vms", "cluster_capacity_pct",
            "cluster_utilization_pct", "sla_violation", "scaling_action"
        ]].copy()
        log_disp.columns = [
            "Time (min)", "Actual CPU%", "Predicted CPU%",
            "Hosts", "VMs", "Capacity%", "Utilisation%", "SLA Violated", "Decision"
        ]
        st.dataframe(log_disp, use_container_width=True, height=260)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════
# SECTION D — RESOURCE UTILIZATION
# ════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-banner">SECTION D — Cluster Resource Utilization '
    '(Predictive AI vs Reactive Baseline)</div>',
    unsafe_allow_html=True,
)

if sim is None:
    st.warning("No simulation data found. Run the pipeline.")
else:
    df_pred = sim["df_predictive"]
    df_reac = sim["df_reactive"]

    col_p, col_r = st.columns(2)

    for col, df, label, cap_color, title in [
        (col_p, df_pred, "AI Predictive", "#34D399", "Predictive (AI) Controller"),
        (col_r, df_reac, "Reactive",      "#EF4444", "Reactive (Baseline) Controller"),
    ]:
        with col:
            st.markdown(f"##### {title}")

            tw = df.tail(win)

            fig_d = go.Figure()
            fig_d.add_trace(go.Scatter(
                y=tw["cluster_utilization_pct"],
                name="Cluster Utilisation %",
                line=dict(color=cap_color, width=2),
                fill="tozeroy",
                fillcolor=f"rgba({','.join(str(int(cap_color[i:i+2],16)) for i in (1,3,5))},0.12)",
            ))
            # Mark SLA violations
            sla_r = tw[tw["sla_violation"]]
            if len(sla_r):
                fig_d.add_trace(go.Scatter(
                    x=sla_r.index - tw.index[0],
                    y=sla_r["cluster_utilization_pct"],
                    mode="markers",
                    marker=dict(color="#EF4444", size=9, symbol="x"),
                    name="SLA Violation",
                ))
            fig_d.update_layout(
                template="plotly_dark", height=260,
                yaxis_title="Utilisation (%)", xaxis_title="Time Steps",
                margin=dict(l=8, r=8, t=28, b=8),
                showlegend=True,
                legend=dict(orientation="h", y=1.1, x=1, xanchor="right"),
            )
            st.plotly_chart(fig_d, use_container_width=True)
             
            avg_u = df["cluster_utilization_pct"].mean()
            sla_v  = pick(sim, label, "sla")
            wasted = pick(sim, label, "wast")

            m1, m2, m3 = st.columns(3)
            m1.metric("Avg Utilisation", f"{avg_u:.1f}%")
            m2.metric("SLA Violations", sla_v)
            m3.metric("Wasted CPU Units", f"{wasted:.0f}")

    # ── Side-by-side utilisation overlay ──
    st.markdown("##### Predictive vs Reactive — Utilisation Overlay")
    fig_ov = go.Figure()
    fig_ov.add_trace(go.Scatter(
        y=df_pred.tail(win)["cluster_utilization_pct"],
        name="Predictive (AI)",
        line=dict(color="#34D399", width=2),
    ))
    fig_ov.add_trace(go.Scatter(
        y=df_reac.tail(win)["cluster_utilization_pct"],
        name="Reactive (Baseline)",
        line=dict(color="#EF4444", width=2, dash="dash"),
    ))
    fig_ov.update_layout(
        template="plotly_dark", height=280,
        xaxis_title="Time Step (5 min each)",
        yaxis_title="Cluster Utilisation (%)",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
        margin=dict(l=10, r=10, t=28, b=10),
    )
    st.plotly_chart(fig_ov, use_container_width=True)

    # ── Donut charts — over/under provisioning breakdown ──
    st.markdown("##### Resource Efficiency Breakdown (Full Simulation)")
    col_d1, col_d2 = st.columns(2)
    for c, label, wp, up in [
        (col_d1, "Predictive (AI)", sim["Predictive_Wasted_CPU"],
         sim["df_predictive"]["cpu_under_prov"].sum()),
        (col_d2, "Reactive",        sim["Reactive_Wasted_CPU"],
         sim["df_reactive"]["cpu_under_prov"].sum()),
    ]:
        with c:
            total = wp + up + 1e-6
            fig_pie = go.Figure(go.Pie(
                labels=["Over-provisioned (wasted)", "Under-provisioned (SLA risk)"],
                values=[wp, up],
                hole=0.55,
                marker_colors=["#F59E0B", "#EF4444"],
            ))
            fig_pie.update_layout(
                template="plotly_dark",
                title_text=label,
                height=260,
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════
# SECTION E — INTERACTIVE SPIKE SIMULATOR
# ════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-banner">SECTION E — Interactive Workload Spike Simulator</div>',
    unsafe_allow_html=True,
)
st.caption(
    "Manually simulate a traffic burst to see how the AI predictive controller "
    "responds vs the reactive baseline in real-time."
)

c_sl, c_btn = st.columns([3, 1])
with c_sl:
    base  = st.slider("Base Cluster Utilisation (%)", 10, 85, 50, 5)
    spike = st.slider("Simulated Spike (+% CPU)",     5,  70, 30, 5)
with c_btn:
    st.markdown("<br><br>", unsafe_allow_html=True)
    go_btn = st.button("Trigger Spike", use_container_width=True)

if go_btn:
    demand = min(99.0, base + spike)
    # AI uses a forecast with small noise (simulating model prediction error)
    np.random.seed(int(demand))
    pred = min(99.0, demand * np.random.uniform(0.93, 1.05))

    CAPACITY = 10.0
    BUFFER   = 1.20

    hosts_ai   = max(2, int(np.ceil(pred   * BUFFER / CAPACITY)))
    hosts_reac = max(2, int(np.ceil(demand * BUFFER / CAPACITY)))
    cap_ai     = hosts_ai   * CAPACITY
    cap_reac   = hosts_reac * CAPACITY
    sla_ai     = demand <= cap_ai
    sla_reac   = demand <= cap_reac

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Actual Spike Demand",  f"{demand:.1f} %")
    m2.metric("ML Forecast (Prediction)", f"{pred:.1f} %", delta=f"{pred-demand:+.1f}% error")
    m3.metric("AI Hosts Provisioned", f"{hosts_ai}",   delta="Proactive (look-ahead)")
    m4.metric("Reactive Hosts",       f"{hosts_reac}", delta="Current-demand based")

    st.markdown("---")
    c_ai, c_rx = st.columns(2)
    with c_ai:
        st.markdown("##### AI Predictive Controller")
        st.metric("Provisioned Capacity", f"{cap_ai:.0f} %")
        if sla_ai:
            st.success(f"SLA MAINTAINED — {demand:.1f}% demand absorbed by {cap_ai:.0f}% capacity.")
        else:
            st.error(f"SLA BREACHED — {demand:.1f}% demand exceeds {cap_ai:.0f}% capacity.")

    with c_rx:
        st.markdown("##### Reactive Baseline Controller")
        st.metric("Provisioned Capacity", f"{cap_reac:.0f} %")
        if sla_reac:
            st.success(f"SLA MAINTAINED — {demand:.1f}% demand absorbed by {cap_reac:.0f}% capacity.")
        else:
            st.error(f"SLA BREACHED — {demand:.1f}% demand exceeds {cap_reac:.0f}% capacity.")

    # Visual gauge
    fig_g = go.Figure()
    for val, name, color in [
        (demand,   "Actual Demand",             "#38BDF8"),
        (pred,     "AI Prediction",             "#34D399"),
        (cap_ai,   "AI Provisioned Capacity",   "#F59E0B"),
        (cap_reac, "Reactive Capacity",         "#EF4444"),
    ]:
        fig_g.add_trace(go.Bar(name=name, x=[val], y=[name], orientation="h",
                               marker_color=color))
    fig_g.update_layout(
        template="plotly_dark", height=220, barmode="overlay",
        xaxis=dict(range=[0, 110], title="% CPU"),
        margin=dict(l=160, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_g, use_container_width=True)
