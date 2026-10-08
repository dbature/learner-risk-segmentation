"""Learner risk dashboard for programme leadership (BAN6800 Module 5).

    streamlit run dashboard/app.py

Reads only the aggregate files in dashboard/data/ (built by
src.dashboard.build_data) and the Module 4 model in models/. Predictions in
the what-if view go through the same input validation as the /predict API
(src.serving.api), so the dashboard and the API cannot disagree.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dashboard.contrib import PLAIN, contributions  # noqa: E402
from src.segments.trajectory import ORDER, TrajectorySegmenter  # noqa: E402

DATA = ROOT / "dashboard" / "data"
REPO = "https://github.com/dbature/learner-risk-segmentation"

# Colours: reference categorical order (blue, orange, aqua, yellow), a neutral
# grey for "before" and status colours that always travel with a text label.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY, INK, MUTED = "#a9a8a2", "#0b0b0b", "#52514e"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
SEG_COLOUR = {"Steady": BLUE, "Slipping": ORANGE, "Late starter": AQUA, "Disengaged": YELLOW}
ATT_NAME = {"gender": "Sex", "age_band": "Age band", "disability": "Disability", "imd_band": "Deprivation (IMD band)"}
GROUP_NAME = {"F": "Female", "M": "Male", "N": "No disability", "Y": "Disability declared", "0-35": "Under 35",
              "35-55": "35 to 55", "55<=": "55 and over"}

st.set_page_config(page_title="Learner risk dashboard", page_icon=":bar_chart:", layout="wide")


@st.cache_data
def load(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text())


@st.cache_resource
def model():
    import joblib

    return joblib.load(ROOT / "models" / "model.joblib")


@st.cache_resource
def segmenter():
    return TrajectorySegmenter.from_dict(load("segments")["model"])


def style(fig: go.Figure, height: int = 340, legend: bool = False) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), showlegend=legend,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(size=13), legend=dict(orientation="h", y=1.12, x=0),
                      hoverlabel=dict(bgcolor="white", font_color=INK))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


def pct(x: float, d: int = 0) -> str:
    return f"{100 * x:.{d}f}%"


summary, drivers, segs, fair, examples = (load(n) for n in ["summary", "drivers", "segments", "fairness", "examples"])
T = summary["test"]
THR = summary["model"]["threshold"]

st.sidebar.title("Learner risk dashboard")
st.sidebar.caption("Day-30 early warning for learners at risk of not completing. BAN6800 capstone, Desmond Amos Bature.")
page = st.sidebar.radio("View", ["Overview", "Cohort view", "What drives risk", "Example learners and what-if",
                                 "Learner segments", "Coach caseload", "Ethical compliance", "About this model"])
st.sidebar.divider()
st.sidebar.markdown(
    f"**Model status:** registered, in **Staging** (not released)  \n"
    f"Fairness gate: **Fail** on disability and deprivation  \n"
    f"[Repository]({REPO}) · [Model card]({REPO}/blob/main/docs/model_card.md)")
st.sidebar.caption("Data: Open University Learning Analytics Dataset (Kuzilek et al., 2017), CC BY 4.0, "
                   "used as an analogue. Only aggregates and three example learners are shown.")

# ------------------------------------------------------------------ overview
if page == "Overview":
    st.title("Who is likely not to complete, seen on day 30")
    st.write("On day 30 of a course, the model ranks every learner still enrolled by their chance of not completing "
             "(withdrawing or failing). A coach can work with about a quarter of a cohort, so the riskiest 25% are "
             "flagged. Figures are from 5,455 test learners the model never saw in training.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Learners who did not complete", pct(T["non_completion_rate"]), help="Share of the test cohort")
    c2.metric("Non-completers reached (25% flagged)", pct(T["recall"]),
              f"{T['recall'] / summary['baselines']['baseline_random']['recall']:.1f}x random choice", delta_color="off")
    c3.metric("Flags that were right", pct(T["precision"]), help="Of every 100 flagged learners, how many did not complete")
    c4.metric("Next cohort (later course run)", pct(summary["future_cohort"]["recall"]),
              "reached, tested on a later run", delta_color="off")

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("How many coaches reach depends on how many they can see")
        cap = st.slider("Share of the cohort coaches can work with", 5, 60, 25, 1, format="%d%%")
        cc = pd.DataFrame(summary["capacity_curve"])
        row = cc.iloc[(cc["share_flagged"] - cap / 100).abs().argmin()]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=100 * cc["share_flagged"], y=100 * cc["recall"], name="Non-completers reached",
                                 line=dict(color=BLUE, width=2),
                                 hovertemplate="%{x:.0f}% flagged: %{y:.0f}% of non-completers reached<extra></extra>"))
        fig.add_trace(go.Scatter(x=100 * cc["share_flagged"], y=100 * cc["share_flagged"], name="Random choice",
                                 line=dict(color=GREY, width=2, dash="dot"), hoverinfo="skip"))
        fig.add_vline(x=cap, line=dict(color=MUTED, width=1))
        fig.update_xaxes(title="Share of cohort flagged (%)")
        fig.update_yaxes(title="Non-completers reached (%)", range=[0, 100])
        st.plotly_chart(style(fig, legend=True), use_container_width=True)
        st.write(f"At **{cap}%** flagged, coaches would reach **{pct(row['recall'])}** of learners who go on not to "
                 f"complete, and **{pct(row['precision'])}** of the learners they see would be ones who need it.")
    with right:
        st.subheader("Higher scores really do mean higher risk")
        rb = pd.DataFrame(summary["risk_bands"])
        fig = go.Figure(go.Bar(x=rb["band"], y=100 * rb["non_completion"],
                               marker_color=[BLUE if f else GREY for f in rb["flagged"]],
                               customdata=np.c_[rb["learners"], 100 * rb["share"]],
                               hovertemplate="Risk %{x}: %{y:.0f}% did not complete<br>%{customdata[0]:,} learners "
                                             "(%{customdata[1]:.0f}% of cohort)<extra></extra>",
                               text=[pct(v) for v in rb["non_completion"]], textposition="outside"))
        fig.update_yaxes(title="Did not complete (%)", range=[0, 105])
        fig.update_xaxes(title="Predicted risk band")
        st.plotly_chart(style(fig), use_container_width=True)
        st.caption("Blue bands are flagged at the 25% capacity line. Grey bands are not.")

# ------------------------------------------------------------------ cohort view (#20)
elif page == "Cohort view":
    co = load("cohort")
    st.title("Where risk concentrates")
    st.write(f"How often learners did not complete, and how much risk the model gave them, across "
             f"{co['learners']:,} test learners still enrolled on day 30. Pick one breakdown at a time; groups with "
             f"fewer than {co['min_cell']} learners are hidden.")
    by = st.radio("Break down by", list(co["breakdowns"]), horizontal=True)
    t = pd.DataFrame(co["breakdowns"][by])
    shown = t[~t["suppressed"]].copy()
    if by == "Course run":
        shown = shown.sort_values("mean_risk", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=shown["group"], y=100 * shown["non_completion"], name="Did not complete", marker_color=GREY,
                         customdata=shown["learners"],
                         hovertemplate="%{x}: %{y:.0f}% did not complete (%{customdata:,} learners)<extra></extra>"))
    fig.add_trace(go.Bar(x=shown["group"], y=100 * shown["mean_risk"], name="Average risk score", marker_color=BLUE,
                         hovertemplate="%{x}: average risk %{y:.0f}%<extra></extra>"))
    fig.add_hline(y=100 * co["overall"]["non_completion"], line=dict(color=MUTED, width=1, dash="dot"),
                  annotation_text="all learners", annotation_position="top right")
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(title="Percent", range=[0, 75])
    st.plotly_chart(style(fig, height=380, legend=True), use_container_width=True)
    table = shown.assign(**{c: (100 * shown[c]).round(1) for c in ["non_completion", "withdrawal", "mean_risk", "flagged"]})
    st.dataframe(table.rename(columns={"group": by, "learners": "Learners", "non_completion": "Did not complete (%)",
                                       "withdrawal": "Withdrew after day 30 (%)", "mean_risk": "Average risk (%)",
                                       "flagged": "Flagged at 25% (%)"}).drop(columns=["suppressed"]),
                 hide_index=True, use_container_width=True)
    if by == "Deprivation (IMD band)":
        st.info("The model's average risk tracks the real pattern, but it under-estimates the most deprived band and "
                "over-estimates the least deprived. That is one reason the fairness gate looks at recall by group.")
    elif by == "Course run":
        st.info("Risk concentrates in particular courses (CCC and DDD runs sit highest), which is why the dashboard "
                "compares learners with others on the same course.")
    st.caption("Descriptive only. Withdrawal counts learners who left after day 30; non-completion adds those who failed.")

# ------------------------------------------------------------------ drivers
elif page == "What drives risk":
    st.title("What drives a learner's risk score")
    st.write("Each bar shows how much a factor moves risk scores on average, across all test learners (SHAP values). "
             "None of these is a personal characteristic such as sex, age, disability or where someone lives.")
    imp = pd.DataFrame(drivers["importance"]).iloc[::-1]
    fig = go.Figure(go.Bar(x=imp["mean_abs_effect"], y=imp["label"], orientation="h", marker_color=BLUE,
                           hovertemplate="%{y}: average effect %{x:.2f}<extra></extra>"))
    fig.update_xaxes(title="Average effect on the risk score (log-odds)")
    st.plotly_chart(style(fig, height=420), use_container_width=True)

    st.subheader("How each factor pushes risk up or down")
    opts = {v["label"]: k for k, v in drivers["effects"].items()}
    choice = st.selectbox("Factor", list(opts))
    eff = drivers["effects"][opts[choice]]
    b = pd.DataFrame(eff["bands"])
    lab = [f"{lo:g}" if lo == hi else f"{lo:g} to {hi:g}" for lo, hi in zip(b["low"], b["high"])]
    vals = list(b["effect"])
    if "missing" in eff:
        lab.append("Nothing due yet")
        vals.append(eff["missing"]["effect"])
    fig = go.Figure(go.Bar(x=lab, y=vals, marker_color=[CRITICAL if v > 0 else BLUE for v in vals],
                           hovertemplate="%{x}: %{y:+.2f}<extra></extra>"))
    fig.update_yaxes(title="Effect on risk (above 0 raises it)")
    st.plotly_chart(style(fig), use_container_width=True)
    st.caption("Red bars raise risk and blue bars lower it.")
    ce = drivers["course_effects"]
    st.write("**The course itself matters too.** Learners on some courses carry more risk than others with the same "
             "behaviour: " + ", ".join(f"{m} {v:+.2f}" for m, v in sorted(ce.items(), key=lambda kv: kv[1])) + ".")

# ------------------------------------------------------------------ examples + what-if
elif page == "Example learners and what-if":
    from fastapi import HTTPException
    from pydantic import ValidationError

    from src.serving.api import Learner, to_frame

    st.title("Example learners, and what would change their score")
    labels = {f"Learner {k}: {examples[k]['story']}": k for k in ["A", "B", "C"]}
    key = labels[st.radio("Learner", list(labels), horizontal=True)]
    ex = examples[key]
    f0 = ex["features"]
    st.caption(f"Course {f0['code_module']}, segment: {ex['segment']}. Actual outcome: "
               f"{'did not complete' if ex['outcome'] else 'completed'}.")

    st.subheader("Change what a coach could influence")
    c1, c2, c3 = st.columns(3)
    due = int(f0["n_due_by_30"])
    with c1:
        sub = st.slider("Early assignments handed in", 0, max(due, 1), int(f0["n_submitted_by_30"]),
                        disabled=due == 0, help="None were due by day 30" if due == 0 else None)
        score = None
        if sub > 0:
            score = st.slider("Average mark (%)", 0, 100, int(f0["mean_score_by_30"] or 60))
    with c2:
        weeks = [st.slider(f"Online clicks, week {w}", 0, 600, int(f0[f"clicks_wk{w}"]), step=5,
                           disabled=w < 3, help="Weeks 1 and 2 are history" if w < 3 else None) for w in (1, 2, 3, 4)]
    with c3:
        active_weeks = sum(1 for w in weeks if w > 0)
        max_days = int(min(30, sum(min(d, w) for d, w in zip([7, 7, 7, 9], weeks))))
        days0 = int(min(max(f0["active_days_0_29"], active_weeks), max_days)) if max_days else 0
        days = st.slider("Days active online (of 30)", 0, max(max_days, 1), days0, disabled=max_days == 0)
        days = int(min(max(days, active_weeks), max_days))
    total = int(sum(weeks))
    first = f0["first_active_day"]
    if total > 0 and first < 0:
        first = [0, 7, 14, 21][next(i for i, w in enumerate(weeks) if w > 0)]
    payload = {**{k: f0[k] for k in Learner.model_fields if k in f0},
               "n_submitted_by_30": sub, "mean_score_by_30": score,
               "clicks_wk1": weeks[0], "clicks_wk2": weeks[1], "clicks_wk3": weeks[2], "clicks_wk4": weeks[3],
               "clicks_0_29": total, "active_days_0_29": days, "first_active_day": int(first),
               "distinct_sites_0_29": int(min(f0["distinct_sites_0_29"], total) if total else 0)}
    for k in ["num_of_prev_attempts", "studied_credits", "n_due_by_30", "n_banked_by_30", "clicks_pre_start"]:
        payload[k] = int(payload[k])
    try:
        X = to_frame(Learner(**payload))
    except (ValidationError, HTTPException) as e:
        st.error(f"That combination is not valid input for the model: {getattr(e, 'detail', e)}")
        st.stop()
    risk = float(model().predict_proba(X)[0, 1])
    g, _ = contributions(model(), X)
    seg_now = segmenter().predict(X)[0]

    m1, m2, m3 = st.columns(3)
    m1.metric("Risk score", pct(risk), f"{100 * (risk - ex['risk']):+.0f} points vs actual", delta_color="inverse")
    m2.metric("Flagged at 25% capacity", "Yes" if risk >= THR else "No", f"line at {pct(THR)}", delta_color="off")
    m3.metric("Engagement segment", seg_now, "was " + ex["segment"] if seg_now != ex["segment"] else "unchanged",
              delta_color="off")

    row = g.iloc[0]
    top = row.reindex(row.abs().sort_values(ascending=False).index)[:6][::-1]
    fig = go.Figure(go.Bar(x=top.values, y=[PLAIN.get(f, f) for f in top.index], orientation="h",
                           marker_color=[CRITICAL if v > 0 else BLUE for v in top.values],
                           hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
    fig.update_xaxes(title="Effect on this learner's score (above 0 raises risk)")
    st.subheader("Why the model scores this learner this way")
    st.plotly_chart(style(fig, height=320), use_container_width=True)
    if key == "A" and "counterfactual" in ex:
        cf = ex["counterfactual"]
        st.info(f"The counterfactual search in Module 4 found that about {cf['clicks_wk4']:.0f} clicks in week 4, over "
                f"{cf['active_days_0_29'] - f0['active_days_0_29']:.0f} more active days, would have brought this "
                f"learner to {pct(cf['risk_score'])}, below the line. Try it with the sliders.")
    if key == "C":
        st.warning("This learner did everything the model looks for and still did not complete. Reasons that leave "
                   "no trace in the first 30 days (work, health, family) are invisible to it, so a low score is "
                   "never a guarantee.")

# ------------------------------------------------------------------ segments
elif page == "Learner segments":
    st.title("Four engagement patterns in the first month")
    st.write("Learners are grouped by how their online activity changed over weeks 1 to 4, compared with others on the "
             "same course. The groups suggest the kind of support that fits, while the risk score says who to see first.")
    sm = pd.DataFrame(segs["summary"]).set_index("segment").loc[ORDER]
    cols = st.columns(4)
    for c, s in zip(cols, ORDER):
        r = sm.loc[s]
        c.markdown(f"<div style='border-top:4px solid {SEG_COLOUR[s]};padding-top:6px'><b>{s}</b></div>",
                   unsafe_allow_html=True)
        c.metric("Share of learners", pct(r["share"]))
        c.metric("Did not complete", pct(r["non_completion"]))
        c.caption(segs["description"][s])
    left, right = st.columns(2)
    with left:
        st.subheader("Typical weekly activity")
        md = pd.DataFrame(segs["median_weekly_clicks"]).set_index("segment")
        fig = go.Figure()
        for s in ORDER:
            fig.add_trace(go.Scatter(x=["Week 1", "Week 2", "Week 3", "Week 4"], y=md.loc[s].values, name=s,
                                     mode="lines+markers", line=dict(color=SEG_COLOUR[s], width=2), marker=dict(size=8),
                                     hovertemplate=s + ", %{x}: median %{y:.0f} clicks<extra></extra>"))
        fig.update_yaxes(title="Median clicks")
        st.plotly_chart(style(fig, legend=True), use_container_width=True)
    with right:
        st.subheader("Who the risk model already reaches")
        fig = go.Figure(go.Bar(x=ORDER, y=100 * sm["non_completers_flagged"], marker_color=[SEG_COLOUR[s] for s in ORDER],
                               text=[pct(v) for v in sm["non_completers_flagged"]], textposition="outside",
                               hovertemplate="%{x}: %{y:.0f}% of non-completers flagged<extra></extra>"))
        fig.update_yaxes(title="Non-completers flagged (%)", range=[0, 100])
        st.plotly_chart(style(fig), use_container_width=True)
    st.info(f"Slipping and late-starting learners fail to complete at the same rate (about "
            f"{pct(sm.loc['Slipping', 'non_completion'])}), but the risk score flags fewer than two in five of "
            "them. Their warning sign is the shape of their activity, which the segments make visible.")
    sil = segs["model"]["silhouette"]
    st.caption(f"K-means with four groups, fitted on training learners only. Separation is modest (silhouette "
               f"{sil['4']:.2f}; {sil['3']:.2f} for three groups and {sil['5']:.2f} for five), so these are broad "
               "patterns along a continuum, not fixed types of learner.")

# ------------------------------------------------------------------ coach caseload (#19)
elif page == "Coach caseload":
    import io

    from src.dashboard.caseload import rank, template

    st.title("Coach caseload: who to see first")
    st.write("Upload your cohort as a CSV with one row per learner, using the day-30 fields the API takes. Each row is "
             "checked exactly as the /predict API checks it, scored, and ranked. The top share that fits your "
             "capacity becomes the caseload.")
    st.warning("Privacy: an uploaded file stays in this browser session only. Nothing is saved, logged or sent "
               "anywhere else. Do not include names or other personal details; use your own reference codes.")
    c1, c2 = st.columns([1, 1])
    source = c1.radio("Cohort", ["Synthetic demo cohort (60 invented learners)", "Upload my cohort (CSV)"])
    cap = c2.slider("Coach capacity: share of the cohort you can see", 5, 50, 25, 1, format="%d%%", key="ccap")
    c2.download_button("Download the CSV template", template().to_csv(index=False).encode(), "cohort_template.csv",
                       "text/csv")
    if source.startswith("Synthetic"):
        cohort = pd.read_csv(ROOT / "dashboard" / "sample_cohort.csv")
        st.caption("Demo file: every learner is invented (references start with SYN-). See scripts/make_synthetic_cohort.py.")
    else:
        up = st.file_uploader("Cohort CSV", type="csv")
        if up is None:
            st.stop()
        cohort = pd.read_csv(io.BytesIO(up.getvalue()))
    try:
        ranked, errors = rank(cohort, model(), segmenter(), cap / 100)
    except ValueError as e:
        st.error(f"The file cannot be read: {e}")
        st.stop()
    if errors:
        st.error(f"{len(errors)} row(s) were not scored because the API would reject them.")
        st.dataframe(pd.DataFrame(errors), hide_index=True, use_container_width=True)
    if ranked.empty:
        st.stop()
    k = int(ranked["in_caseload"].sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Learners scored", f"{len(ranked):,}")
    m2.metric("In this week's caseload", f"{k:,}", f"top {cap}%", delta_color="off")
    m3.metric("Lowest risk in the caseload", pct(ranked.loc[ranked["in_caseload"], "risk"].min()))
    view = ranked.assign(risk=(100 * ranked["risk"]).round(0).astype(int).astype(str) + "%")
    only = st.toggle("Show the caseload only", value=True)
    if only:
        view = view[view["in_caseload"]]
    st.dataframe(view.rename(columns={"rank": "Rank", "learner_ref": "Learner", "course": "Course", "risk": "Risk",
                                      "segment": "Segment", "top_reasons": "Main reasons", "in_caseload": "In caseload"}),
                 hide_index=True, use_container_width=True)
    st.download_button("Download the ranked list", ranked.to_csv(index=False).encode(), "ranked_caseload.csv", "text/csv")
    st.caption("A ranking is a starting point for a coach's judgement, not a decision. The model is in Staging and "
               "has not been released for use with real learners.")

# ------------------------------------------------------------------ fairness
elif page == "Ethical compliance":
    from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate

    rep = fair["report"]
    st.title("Ethical compliance: does the model treat groups fairly?")
    st.write("The commitment from Module 1 is **equal opportunity**: among learners who will not complete, every group "
             "should have the same chance of being flagged for support, within 5 percentage points. "
             "Sex, age, disability and deprivation are never model inputs; they are used only to check the results.")
    st.subheader("Release gate (validated, 25% capacity, after mitigation)")
    rows = []
    for a in ["gender", "age_band", "disability", "imd_band"]:
        g = rep["gate_after"][a]
        s = rep["after"][a]
        rows.append({"Group": ATT_NAME[a], "Test applied": "Gap of 5 points or less" if "gap" in g["rule"]
                     else "No real difference (chi-square p of 0.05 or more)",
                     "Gap before": f"{100 * rep['before'][a]['recall_gap']:.1f} pts",
                     "Gap after": f"{100 * s['recall_gap']:.1f} pts", "p-value": f"{s['heterogeneity_p']:.3f}",
                     "Result": "✓ Pass" if g["pass"] else "✗ Fail"})
    gate = pd.DataFrame(rows)
    st.dataframe(gate.style.map(lambda v: f"color:{GOOD if 'Pass' in v else CRITICAL};font-weight:bold"
                                if isinstance(v, str) and ("Pass" in v or "Fail" in v) else "", subset=["Result"]),
                 hide_index=True, use_container_width=True)
    st.error("The model fails the gate on disability (5.1 points, just over the limit, in favour of disabled learners) "
             "and on deprivation (the most deprived learners are reached more often than the least deprived). "
             "It stays in Staging and cannot be released until the Ethics Committee rules.")

    st.subheader("Explore: recall by group at any capacity")
    c1, c2 = st.columns([1, 1])
    att = {v: k for k, v in ATT_NAME.items()}[c1.selectbox("Group", list(ATT_NAME.values()))]
    cap = c2.slider("Share of cohort flagged", 10, 50, 25, 1, format="%d%%", key="fcap")
    bins = np.array(fair["bins"])
    hist = fair["histograms"][att]
    total = sum(np.array(v["1"]) + np.array(v["0"]) for v in hist.values())
    above = np.cumsum(total[::-1])[::-1]
    # at 25% use the validated flag count, so the figures match the Module 4 report
    target = T["tp"] + T["fp"] if cap == 25 else cap / 100 * total.sum()
    i_thr = int(np.argmin(np.abs(above - target)))
    yt, yp, sf = [], [], []
    for grp, v in hist.items():
        for lab in ("0", "1"):
            h = np.array(v[lab])
            n_flag, n = int(h[i_thr:].sum()), int(h.sum())
            yt += [int(lab)] * n
            yp += [1] * n_flag + [0] * (n - n_flag)
            sf += [grp] * n
    mf = MetricFrame(metrics={"Recall": true_positive_rate, "Flag rate": selection_rate},
                     y_true=np.array(yt), y_pred=np.array(yp), sensitive_features=np.array(sf))
    bg = mf.by_group.reset_index().rename(columns={"sensitive_feature_0": "group"})
    pos = {g: int(np.sum(v["1"])) for g, v in hist.items()}
    bg["Non-completers"] = bg["group"].map(pos)
    bg["Gated"] = bg["Non-completers"] >= rep["min_positives_gated"]
    order = sorted(bg["group"], key=lambda x: (x == "Missing", x))
    bg = bg.set_index("group").loc[order].reset_index()
    fig = go.Figure(go.Bar(x=[GROUP_NAME.get(g, g) for g in bg["group"]], y=100 * bg["Recall"],
                           marker_color=[BLUE if gt else GREY for gt in bg["Gated"]],
                           customdata=np.c_[bg["Non-completers"], 100 * bg["Flag rate"]],
                           text=[pct(v) for v in bg["Recall"]], textposition="outside",
                           hovertemplate="%{x}: %{y:.1f}% of %{customdata[0]:,} non-completers flagged"
                                         "<br>%{customdata[1]:.1f}% of the group flagged<extra></extra>"))
    fig.update_yaxes(title="Non-completers flagged (%)", range=[0, 100])
    st.plotly_chart(style(fig), use_container_width=True)
    gated = bg[bg["Gated"]]
    gap = 100 * (gated["Recall"].max() - gated["Recall"].min())
    di = gated["Flag rate"].min() / gated["Flag rate"].max()
    k1, k2, k3 = st.columns(3)
    k1.metric("Recall gap (largest minus smallest)", f"{gap:.1f} pts", "within 5" if gap <= 5 else "over 5",
              delta_color="normal" if gap <= 5 else "inverse")
    k2.metric("Flag-rate ratio (lowest to highest)", f"{di:.2f}", "four-fifths benchmark 0.80", delta_color="off")
    k3.metric("Single threshold used", f"{THR if cap == 25 else bins[i_thr]:.3f}", "no mitigation applied here",
              delta_color="off")
    st.caption("Computed live with Fairlearn's MetricFrame from counts of test learners by score band, group and "
               "outcome. Grey bars are groups with fewer than 100 non-completers, shown but not gated. "
               "This view uses one threshold for everyone; the validated mitigation (separate thresholds by sex at "
               "25% capacity) is in the gate table above.")
    for a in ["gender", "imd_band"]:
        if a == att:
            st.write("**Validated mitigation result for this group (25% capacity)**")
            after = pd.DataFrame(rep["after"][a]["groups"])
            after["group"] = after["group"].map(lambda g: GROUP_NAME.get(g, g))
            after["recall"] = after["recall"].map(lambda v: pct(v, 1))
            after["selection_rate"] = after["selection_rate"].map(lambda v: pct(v, 1))
            after["positives"] = after["positives"].map(lambda v: f"{int(v):,}")
            st.dataframe(after[["group", "positives", "recall", "selection_rate"]].rename(columns={
                "group": "Group", "positives": "Non-completers", "recall": "Non-completers flagged",
                "selection_rate": "Share of group flagged"}), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ about
else:
    st.title("About this model: what it can and cannot do")
    a, b = st.columns(2)
    with a:
        st.subheader("It can")
        st.markdown("- Rank learners still enrolled on day 30 by their chance of not completing.\n"
                    "- Show, for each learner, the factors behind the score.\n"
                    f"- Reach about {pct(T['recall'])} of non-completers when coaches see 25% of a cohort, "
                    f"against {pct(summary['baselines']['baseline_random']['recall'])} by chance.\n"
                    "- Be checked: every number here comes from files in the public repository.")
    with b:
        st.subheader("It cannot")
        st.markdown("- Help learners who left before day 30 (about half of all withdrawals).\n"
                    "- See reasons that leave no trace online, such as work, health or family.\n"
                    "- Decide anything. A named coach decides; the model never contacts a learner.\n"
                    "- Be assumed to work elsewhere. It was trained on UK adult distance learners and must be "
                    "revalidated before any use with another population.")
    st.subheader("Transparency statement")
    st.write("This is decision support for coaches, built on public data used as an analogue. It is registered but not "
             "released: it fails the fairness gate on two groups, and the Programme Data Ethics Committee decides "
             "what happens next. No personal characteristic is a model input, no learner is shown here, and the "
             "dashboard holds only summary figures and three pseudonymous examples.")
    st.subheader("Calling the model directly")
    st.code("uvicorn src.serving.api:app --port 8000\npython scripts/api_demo.py   # sends Learners A, B and C to /predict",
            language="bash")
    st.markdown(f"[Repository]({REPO}) · [Model card]({REPO}/blob/main/docs/model_card.md) · "
                f"[Fairness report]({REPO}/blob/main/reports/fairness.json) · "
                f"[Dashboard code]({REPO}/blob/main/dashboard/app.py)")
    st.caption("Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. "
               "Scientific Data, 4, 170171. https://doi.org/10.1038/sdata.2017.171")
