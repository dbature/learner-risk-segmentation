"""Model monitoring dashboard: performance, drift and fairness by batch.

    streamlit run monitoring/app.py

Reads monitoring/data/monitoring.json, built by src.monitoring.build on the
machine that holds the data. Aggregates only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/dbature/learner-risk-segmentation"
BLUE, ORANGE, AQUA, GREY, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#a9a8a2", "#52514e"
GOOD, WARN, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
ATT = {"gender": "Sex", "age_band": "Age band", "disability": "Disability"}
RESPONSE = {
    "performance": "Check data quality first; if confirmed on two consecutive batches, open a retraining review "
                   "(src/models/retrain.py). The challenger stays in Staging until it passes every gate.",
    "capacity": "Tell coaching leads; agree whether to hold the 25% line or re-cut the threshold for this cohort.",
    "drift": "Look at which inputs moved. Course-mix shifts are expected; unexplained shifts go to the "
             "measurement lead within one week.",
    "fairness": "High: incident response (docs/ethics/incident_response.md), Ethics Committee told within "
                "5 working days. Low: no action beyond watching the running total.",
}

st.set_page_config(page_title="Model monitoring", page_icon=":chart_with_upwards_trend:", layout="wide")


@st.cache_data
def load() -> dict:
    return json.loads((ROOT / "monitoring" / "data" / "monitoring.json").read_text())


def style(fig, height=320, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), showlegend=legend,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.15, x=0))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


def badge(status: str) -> str:
    c, label = {"ok": (GOOD, "✓ OK"), "warning": (WARN, "! Warning"), "alert": (CRITICAL, "✗ Alert")}[status]
    return f"<span style='background:{c};color:white;padding:2px 10px;border-radius:10px;font-weight:600'>{label}</span>"


m = load()
B = pd.DataFrame([{k: v for k, v in b.items() if k not in ("data_drift", "fairness", "alerts")} for b in m["batches"]])
runs = list(B["batch"])

st.sidebar.title("Model monitoring")
st.sidebar.caption("Day-30 learner risk model. BAN6800 capstone, Desmond Amos Bature.")
view = st.sidebar.radio("View", ["Status", "Performance", "Drift", "Fairness", "Response plan"])
st.sidebar.divider()
reg = m["model"]["registry"] or {}
st.sidebar.markdown(f"**Model:** {m['model']['algorithm']} v{reg.get('version')} ({reg.get('stage')})  \n"
                    f"**File hash:** `{m['model']['sha256']}`  \n**Built:** {m['generated_utc'][:10]}")
st.sidebar.markdown(f"[Stakeholder dashboard](https://learner-risk-segmentation.streamlit.app) · [Repository]({REPO})")
st.sidebar.caption(m["note"])

if view == "Status":
    st.title("Is the model still doing its job?")
    st.markdown(f"Overall status: {badge(m['overall_status'])}", unsafe_allow_html=True)
    st.write("Each course run is scored as if it had just arrived, then checked once outcomes are known. "
             "Fairness is also checked on the running total, because single batches are too small to judge.")
    cols = st.columns(len(runs))
    for c, b in zip(cols, m["batches"]):
        c.markdown(f"**{b['batch']}**  \n{b['learners']:,} learners", unsafe_allow_html=True)
        c.markdown(badge(b["status"]), unsafe_allow_html=True)
        c.metric("ROC AUC", f"{b['roc_auc']:.3f}")
        c.metric("Non-completers reached", f"{100 * b['recall']:.0f}%")
    st.subheader("Open alerts")
    rows = [{"Where": "Running total", "Type": a["type"], "Severity": a["severity"], "Rule": a["rule"]}
            for a in m["cumulative_alerts"]]
    rows += [{"Where": b["batch"], "Type": a["type"], "Severity": a["severity"], "Rule": a["rule"]}
             for b in m["batches"] for a in b["alerts"]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.info("Reading it: performance and the score distribution are stable across runs, so retraining is not "
            "indicated. Input drift is mostly a change in course mix between runs. The open high alerts are the "
            "same two fairness failures found at validation (disability gap, IMD differences), which is why the "
            "model remains in Staging.")

elif view == "Performance":
    st.title("Performance by course run")
    v, oot = m["validation"], m["out_of_time"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=runs, y=B["roc_auc"], name="ROC AUC", mode="lines+markers", line=dict(color=BLUE, width=2),
                             marker=dict(size=9), hovertemplate="%{x}: AUC %{y:.3f}<extra></extra>"))
    fig.add_hline(y=m["rules"]["auc_floor"], line=dict(color=CRITICAL, dash="dot"), annotation_text="alert floor 0.70")
    fig.add_hline(y=v["roc_auc"], line=dict(color=MUTED, dash="dash"), annotation_text="validation")
    fig.add_hline(y=oot["roc_auc"], line=dict(color=ORANGE, dash="dash"), annotation_text="out-of-time 2014J",
                  annotation_position="bottom right")
    fig.update_yaxes(range=[0.65, 0.82], title="ROC AUC")
    st.plotly_chart(style(fig), use_container_width=True)
    fig = go.Figure()
    for col, name, colr in [("recall", "Non-completers reached", BLUE), ("precision", "Flags that were right", AQUA),
                            ("flag_rate", "Share flagged", ORANGE)]:
        fig.add_trace(go.Scatter(x=runs, y=100 * B[col], name=name, mode="lines+markers", line=dict(color=colr, width=2),
                                 marker=dict(size=8), hovertemplate="%{x}: %{y:.1f}%<extra>" + name + "</extra>"))
    fig.update_yaxes(range=[0, 100], title="Percent")
    st.plotly_chart(style(fig), use_container_width=True)
    st.caption(f"Validation: AUC {v['roc_auc']:.3f}, recall {100 * v['recall']:.1f}%. Out-of-time (trained on "
               f"{oot['train']}, tested on {oot['learners']:,} new learners): AUC {oot['roc_auc']:.3f}, recall "
               f"{100 * oot['recall']:.1f}%, the planning figure for a new cohort.")

elif view == "Drift":
    st.title("Drift: are the inputs and scores changing?")
    fig = go.Figure(go.Bar(x=runs, y=B["score_psi"], marker_color=BLUE, text=[f"{x:.3f}" for x in B["score_psi"]],
                           textposition="outside", hovertemplate="%{x}: PSI %{y:.3f}<extra></extra>"))
    fig.add_hline(y=m["rules"]["psi_warn"], line=dict(color=WARN, dash="dot"), annotation_text="watch 0.10")
    fig.add_hline(y=m["rules"]["psi_alert"], line=dict(color=CRITICAL, dash="dot"), annotation_text="alert 0.25")
    fig.update_yaxes(range=[0, 0.3], title="Population stability index (risk score)")
    st.subheader("Prediction drift (score distribution vs training)")
    st.plotly_chart(style(fig, legend=False), use_container_width=True)
    st.subheader("Input drift (Evidently, 20 model inputs)")
    feats = list(m["batches"][0]["data_drift"]["columns"])
    z = [[m["batches"][j]["data_drift"]["columns"][f]["score"] for j in range(len(runs))] for f in feats]
    txt = [["drift" if m["batches"][j]["data_drift"]["columns"][f]["drifted"] else "" for j in range(len(runs))]
           for f in feats]
    fig = go.Figure(go.Heatmap(z=z, x=runs, y=feats, text=txt, texttemplate="%{text}", colorscale="Blues",
                               colorbar=dict(title="drift score"),
                               hovertemplate="%{y}, %{x}: %{z:.3f}<extra></extra>"))
    st.plotly_chart(style(fig, height=560, legend=False), use_container_width=True)
    shares = ", ".join(f"{b['batch']} {b['data_drift']['n_drifted']}/{b['data_drift']['n_columns']}" for b in m["batches"])
    st.caption(f"Inputs flagged as drifted: {shares}. Evidently's default tests (Wasserstein distance for numeric "
               "inputs, Jensen-Shannon for the course) are sensitive at these sample sizes, and each run has a "
               "different course mix, so input drift alone does not trigger retraining.")

elif view == "Fairness":
    st.title("Fairness over time")
    st.write("Recall gap between groups (non-completers flagged), under the mitigated sex-specific thresholds. "
             "Dots are single runs; the line is the running total, which is what the release gate judges.")
    c = pd.DataFrame(m["cumulative_fairness"])
    for att, name in ATT.items():
        per = [b["fairness"]["mitigated"][att]["recall_gap"] for b in m["batches"]]
        run_tot = [r[att]["recall_gap"] for r in m["cumulative_fairness"]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=runs, y=[None if x is None else 100 * x for x in per], name="Single run",
                                 mode="markers", marker=dict(color=GREY, size=11),
                                 hovertemplate="%{x}: %{y:.1f} points<extra>single run</extra>"))
        fig.add_trace(go.Scatter(x=runs, y=[None if x is None else 100 * x for x in run_tot], name="Running total",
                                 mode="lines+markers", line=dict(color=BLUE, width=2), marker=dict(size=8),
                                 hovertemplate="through %{x}: %{y:.1f} points<extra>running total</extra>"))
        fig.add_hline(y=5, line=dict(color=CRITICAL, dash="dot"), annotation_text="tolerance 5 points")
        fig.update_yaxes(title="Recall gap (points)", range=[0, 15])
        st.subheader(name)
        st.plotly_chart(style(fig, height=260), use_container_width=True)
    ps = [r["imd_band"]["chi2_p"] for r in m["cumulative_fairness"]]
    st.subheader("Deprivation (IMD band): is recall equal across bands?")
    fig = go.Figure(go.Scatter(x=runs, y=ps, mode="lines+markers", line=dict(color=BLUE, width=2), marker=dict(size=8),
                               hovertemplate="through %{x}: p = %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0.05, line=dict(color=CRITICAL, dash="dot"), annotation_text="p = 0.05")
    fig.update_yaxes(title="Chi-square p (running total)", type="log")
    st.plotly_chart(style(fig, height=260, legend=False), use_container_width=True)
    st.caption("Per-run groups below 50 non-completers and running-total groups below 100 are reported but not gated.")

else:
    st.title("What happens when an alert fires")
    for k, v in RESPONSE.items():
        st.markdown(f"**{k.capitalize()}.** {v}")
    st.markdown("**Retraining.** Retraining is a decision, not a schedule. `src/models/retrain.py` trains a challenger "
                "on the latest data with the registered settings, logs it to MLflow and registers it in Staging only "
                "if it matches the champion on AUC and recall and passes every fairness gate. Nothing is promoted to "
                "Production without the Ethics Committee.")
    st.markdown(f"Full plans: [incident response]({REPO}/blob/main/docs/ethics/incident_response.md) · "
                f"[monitoring plan]({REPO}/blob/main/docs/monitoring.md) · "
                f"[decommissioning]({REPO}/blob/main/docs/ethics/decommissioning.md)")
    st.json(m["rules"], expanded=False)
