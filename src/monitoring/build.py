"""Monitoring: performance, drift and fairness for each new batch of learners.

    python -m src.monitoring.build

A deployed model sees one new course run at a time. This module replays that:
the held-out test learners (never seen in training) are split by course run
(2013B, 2013J, 2014B, 2014J) and each run is scored as if it had just
arrived, against the training data as reference. For every batch it
records:

* performance once outcomes are known: ROC AUC, recall and precision at the
  validated threshold, flag rate, calibration gap
* prediction drift: population stability index (PSI) of the risk score
* data drift: Evidently's DataDriftPreset on the 20 model inputs (share of
  drifted features, and which ones)
* fairness: recall by sex, age band and disability with the gap against the
  5-point tolerance, the chi-square test across IMD bands, and the flag-rate
  ratio, computed with Fairlearn's MetricFrame, under both the single
  threshold and the mitigated sex-specific thresholds
* alerts from the rules in ALERTS, each tied to a response in
  docs/ethics/incident_response.md

Only aggregates are written (monitoring/data/monitoring.json); no learner
row leaves the machine that holds the data. A limitation is stated in the
output itself: these runs share the training period, so this is a replay of
held-out learners, not a true future. The out-of-time check (train on
2013B to 2014B, test on new 2014J learners) is carried in as its own line.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate
from scipy.stats import chi2_contingency
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from src.models import data

ROOT = Path(__file__).resolve().parents[2]
ORDER = ["2013B", "2013J", "2014B", "2014J"]
MIN_POSITIVES = 100      # same rule as the Module 4 gate
SMALL_BATCH_FAIRNESS = 50  # per-batch groups are smaller; below this, report but do not alert

ALERTS = {
    "auc_floor": 0.70,               # Module 4 acceptance: future-cohort AUC at least 0.70
    "recall_drop_points": 5.0,       # recall more than 5 points below validation
    "flag_rate_band": (0.15, 0.35),  # coach capacity planned at 25%
    "psi_warn": 0.10, "psi_alert": 0.25,
    "drift_share_alert": 0.30,       # Evidently: share of inputs drifted
    "recall_gap_points": 5.0,        # Module 1 equal-opportunity tolerance
    "imd_p_floor": 0.05,             # many-level gate (pending ratification)
}


def _round(obj, d: int = 6):
    if isinstance(obj, float):
        return round(float(obj), d)
    if isinstance(obj, dict):
        return {k: _round(v, d) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, d) for v in obj]
    return obj


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    cuts = np.quantile(expected, np.linspace(0, 1, bins + 1))
    cuts[0], cuts[-1] = -np.inf, np.inf
    e = np.histogram(expected, cuts)[0] / len(expected)
    a = np.histogram(actual, cuts)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def evidently_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    rep = Report(metrics=[DataDriftPreset()])
    rep.run(reference_data=reference, current_data=current)
    res = rep.as_dict()["metrics"]
    summary = next(m["result"] for m in res if m["metric"] == "DatasetDriftMetric")
    table = next(m["result"] for m in res if m["metric"] == "DataDriftTable")
    cols = {c: {"drifted": bool(v["drift_detected"]), "score": float(v["drift_score"]), "test": v["stattest_name"]}
            for c, v in table["drift_by_columns"].items()}
    return {"share_drifted": float(summary["share_of_drifted_columns"]),
            "n_drifted": int(summary["number_of_drifted_columns"]), "n_columns": int(summary["number_of_columns"]),
            "dataset_drift": bool(summary["dataset_drift"]), "columns": cols}


def fairness(y: np.ndarray, pred: np.ndarray, attrs: pd.DataFrame, min_pos: int) -> dict:
    out = {}
    for att in ["gender", "age_band", "disability"]:
        g = attrs[att].astype(str).to_numpy()
        mf = MetricFrame(metrics={"recall": true_positive_rate, "flag_rate": selection_rate},
                         y_true=y, y_pred=pred, sensitive_features=g)
        bg = mf.by_group
        pos = pd.Series(y).groupby(g).sum()
        gated = [k for k in bg.index if pos.get(k, 0) >= min_pos]
        gap = float(bg.loc[gated, "recall"].max() - bg.loc[gated, "recall"].min()) if len(gated) > 1 else None
        sel = bg.loc[gated, "flag_rate"]
        table = [[int(pred[(g == k) & (y == 1)].sum()), int(((g == k) & (y == 1)).sum() - pred[(g == k) & (y == 1)].sum())]
                 for k in gated]
        p_val = float(chi2_contingency(np.array(table))[1]) if len(gated) > 1 else None
        out[att] = {"recall_gap": gap, "chi2_p": p_val,
                    "flag_rate_ratio": float(sel.min() / sel.max()) if len(gated) > 1 else None,
                    "groups": {k: {"recall": float(bg.loc[k, "recall"]), "flag_rate": float(bg.loc[k, "flag_rate"]),
                                   "non_completers": int(pos.get(k, 0)), "gated": k in gated} for k in bg.index}}
    g = attrs["imd_band"].astype(str).to_numpy()
    rows = [[int(pred[(g == k) & (y == 1)].sum()), int(((g == k) & (y == 1)).sum() - pred[(g == k) & (y == 1)].sum())]
            for k in np.unique(g) if k != "Missing" and ((g == k) & (y == 1)).sum() >= min_pos]
    out["imd_band"] = {"chi2_p": float(chi2_contingency(np.array(rows))[1]) if len(rows) > 1 else None,
                       "bands_tested": len(rows)}
    return out


def batch_alerts(b: dict, validation_recall: float) -> list[dict]:
    a, out = ALERTS, []
    if b["roc_auc"] < a["auc_floor"]:
        out.append({"type": "performance", "severity": "high", "rule": f"AUC below {a['auc_floor']}"})
    if 100 * (validation_recall - b["recall"]) > a["recall_drop_points"]:
        out.append({"type": "performance", "severity": "medium", "rule": "recall more than 5 points below validation"})
    lo, hi = a["flag_rate_band"]
    if not lo <= b["flag_rate"] <= hi:
        out.append({"type": "capacity", "severity": "medium", "rule": f"flag rate outside {lo:.0%} to {hi:.0%}"})
    if b["score_psi"] >= a["psi_alert"]:
        out.append({"type": "drift", "severity": "high", "rule": f"score PSI at least {a['psi_alert']}"})
    elif b["score_psi"] >= a["psi_warn"]:
        out.append({"type": "drift", "severity": "low", "rule": f"score PSI at least {a['psi_warn']}"})
    if b["data_drift"]["share_drifted"] >= a["drift_share_alert"]:
        out.append({"type": "drift", "severity": "medium", "rule": "30% or more of inputs drifted"})
    for view in ["mitigated"]:
        f = b["fairness"][view]
        for att in ["gender", "age_band", "disability"]:
            gap, p = f[att]["recall_gap"], f[att]["chi2_p"]
            if gap is not None and 100 * gap > a["recall_gap_points"]:
                real = p is not None and p < a["imd_p_floor"]
                out.append({"type": "fairness", "severity": "high" if real else "low",
                            "rule": f"{att} recall gap {100 * gap:.1f} points (tolerance 5), "
                                    + (f"statistically detectable (p = {p:.3f})" if real
                                       else f"within batch noise (p = {p:.2f}); watch the running total")})
        p = f["imd_band"]["chi2_p"]
        if p is not None and p < a["imd_p_floor"]:
            out.append({"type": "fairness", "severity": "high", "rule": f"IMD recall differs (p = {p:.3f})"})
    return out


def main(base: Path | None = None) -> dict:
    root = Path(base) if base else ROOT
    meta = json.loads((root / "models" / "model_metadata.json").read_text())
    metrics = json.loads((root / "reports" / "metrics.json").read_text())
    model = joblib.load(root / "models" / "model.joblib")
    thr, gthr = float(meta["threshold"]), meta["group_thresholds"]

    df = data.load(root / "data" / "processed")
    split = data.grouped_holdout(df, seed=42)
    ref_scores = model.predict_proba(split.X_train)[:, 1]
    p_all = model.predict_proba(split.X_test)[:, 1]
    pres = split.attrs_test["code_presentation"].to_numpy()
    sel = metrics["candidates"][metrics["selected"]]["test"]
    val_recall = float(sel["recall"])

    batches = []
    for run in ORDER:
        m = pres == run
        X, y, p, attrs = split.X_test[m], split.y_test[m], p_all[m], split.attrs_test[m].reset_index(drop=True)
        pred = (p >= thr).astype(int)
        sex = attrs["gender"].astype(str).to_numpy()
        pred_m = (p >= np.array([gthr.get(s, thr) for s in sex])).astype(int)
        b = {"batch": run, "learners": int(m.sum()), "non_completion": float(y.mean()),
             "roc_auc": float(roc_auc_score(y, p)), "flag_rate": float(pred.mean()),
             "recall": float(recall_score(y, pred)), "precision": float(precision_score(y, pred, zero_division=0)),
             "mean_score": float(p.mean()), "calibration_gap": float(p.mean() - y.mean()),
             "score_psi": psi(ref_scores, p),
             "data_drift": evidently_drift(split.X_train.reset_index(drop=True), X.reset_index(drop=True)),
             "fairness": {"single_threshold": fairness(y, pred, attrs, SMALL_BATCH_FAIRNESS),
                          "mitigated": fairness(y, pred_m, attrs, SMALL_BATCH_FAIRNESS)}}
        b["alerts"] = batch_alerts(b, val_recall)
        b["status"] = ("alert" if any(x["severity"] == "high" for x in b["alerts"])
                       else "warning" if b["alerts"] else "ok")
        batches.append(b)

    # running total across all batches so far: small per-batch groups are noisy,
    # so the release-level fairness test is applied to the accumulated learners
    sex_all = split.attrs_test["gender"].astype(str).to_numpy()
    pred_all = (p_all >= np.array([gthr.get(s, thr) for s in sex_all])).astype(int)
    cumulative = []
    for i, run in enumerate(ORDER):
        m = np.isin(pres, ORDER[: i + 1])
        f = fairness(split.y_test[m], pred_all[m], split.attrs_test[m].reset_index(drop=True), MIN_POSITIVES)
        cumulative.append({"through": run, "learners": int(m.sum()),
                           **{att: {"recall_gap": f[att]["recall_gap"], "chi2_p": f[att]["chi2_p"]}
                              for att in ["gender", "age_band", "disability"]},
                           "imd_band": f["imd_band"]})
    last = cumulative[-1]
    cum_alerts = [{"type": "fairness", "severity": "high", "rule": f"{att} running recall gap "
                   f"{100 * last[att]['recall_gap']:.1f} points exceeds the 5-point release gate"}
                  for att in ["gender", "age_band", "disability"]
                  if last[att]["recall_gap"] is not None and 100 * last[att]["recall_gap"] > ALERTS["recall_gap_points"]]
    if last["imd_band"]["chi2_p"] is not None and last["imd_band"]["chi2_p"] < ALERTS["imd_p_floor"]:
        cum_alerts.append({"type": "fairness", "severity": "high",
                           "rule": f"IMD recall differs across bands (running p = {last['imd_band']['chi2_p']:.3f})"})

    ta = metrics["sensitivity"]["time_aware"]
    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {"algorithm": meta["algorithm"], "registry": meta.get("registry"), "sha256": meta["model_sha256"][:12],
                  "threshold": thr, "group_thresholds": gthr},
        "validation": {"roc_auc": sel["roc_auc"], "recall": val_recall, "precision": sel["precision"],
                       "flag_rate": sel["flag_rate"]},
        "out_of_time": {"train": "2013B to 2014B", "test": "2014J (new learners)", "learners": ta["n_test"],
                        "roc_auc": ta["roc_auc"], "recall": ta["recall"], "precision": ta["precision"],
                        "below_auc_floor": ta["roc_auc"] < ALERTS["auc_floor"]},
        "rules": {k: list(v) if isinstance(v, tuple) else v for k, v in ALERTS.items()},
        "batches": batches,
        "cumulative_fairness": cumulative,
        "cumulative_alerts": cum_alerts,
        "overall_status": "alert" if cum_alerts or any(b["status"] == "alert" for b in batches) else
                          ("warning" if any(b["status"] == "warning" for b in batches) else "ok"),
        "note": "Batches are held-out learners grouped by course run within the training period: a replay, "
                "not a true future. The out-of-time line is the closer guide to performance on a new cohort.",
    }
    path = root / "monitoring" / "data" / "monitoring.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Rounded to 6 decimals and written with LF endings so a rebuild on Windows or Linux gives the
    # same bytes (last-digit float differences between platforms are noise, not change).
    path.write_text(json.dumps(_round(out), indent=1, default=float), encoding="utf-8", newline="\n")
    print(f"wrote {path}: " + ", ".join(f"{b['batch']} {b['status']}" for b in batches))
    return out


if __name__ == "__main__":
    main()
