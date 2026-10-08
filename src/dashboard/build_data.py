"""Builds the files the public dashboard reads (Module 5).

    python -m src.dashboard.build_data

Runs where the processed data lives (the laptop or the model image). It
writes only aggregates to dashboard/data/, never learner rows, in line with
the Module 2 privacy plan:

* summary.json: headline metrics, capacity curve, calibration and risk bands
* drivers.json: global importance and binned effect curves for the top drivers
* segments.json: segment model (module reference statistics and four
  centroids) and segment summaries
* fairness.json: the Module 4 fairness report plus, for each audited group,
  counts of test learners by score band and outcome, so the dashboard can
  recompute recall at any capacity with Fairlearn
* examples.json: the three worked examples used in the deck (Learners A, B
  and C). These are the only individual records, they are pseudonymous and
  hold no protected attribute, and they were already published in Module 4
  (A and B) or are needed to show a missed case honestly (C).

Cells counting fewer than MIN_CELL learners are suppressed in the summary
tables (not in the score histograms, which hold no attribute combinations).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.dashboard.contrib import PLAIN, contributions
from src.models import data
from src.segments.trajectory import DESCRIPTION, ORDER, TrajectorySegmenter, summary

ROOT = Path(__file__).resolve().parents[2]
MIN_CELL = 10
BINS = np.round(np.linspace(0, 1, 1001), 3)


def _round(obj, d=6):
    if isinstance(obj, float):
        return round(obj, d)
    if isinstance(obj, dict):
        return {k: _round(v, d) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, d) for v in obj]
    return obj


def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(_round(obj), indent=1, default=float), encoding="utf-8")


def capacity_curve(y, p):
    order = np.argsort(-p)
    ys = y[order]
    out = []
    for share in np.arange(0.05, 0.61, 0.01):
        n = int(round(share * len(y)))
        tp = int(ys[:n].sum())
        out.append({"share_flagged": float(share), "recall": tp / y.sum(), "precision": tp / n})
    return out


def calibration(y, p, q=10):
    t = pd.DataFrame({"y": y, "p": p})
    t["band"] = pd.qcut(t["p"], q, labels=False, duplicates="drop")
    g = t.groupby("band").agg(predicted=("p", "mean"), observed=("y", "mean"), learners=("y", "size"))
    return g.reset_index(drop=True).to_dict(orient="records")


def risk_bands(y, p, thr):
    edges = [0, 0.2, 0.4, thr, 0.8, 1.0001]
    names = ["Under 20%", "20% to 40%", f"40% to {100 * thr:.0f}%", f"{100 * thr:.0f}% to 80%", "80% and over"]
    t = pd.DataFrame({"y": y, "band": pd.cut(p, edges, labels=names, right=False)})
    g = t.groupby("band", observed=False).agg(learners=("y", "size"), non_completion=("y", "mean"))
    g["share"] = g["learners"] / g["learners"].sum()
    g["flagged"] = [False, False, False, True, True]
    return g.reset_index().to_dict(orient="records")


def effect_curves(contrib: pd.DataFrame, X: pd.DataFrame, features: list[str]):
    """Average contribution by value band, for the strongest numeric drivers."""
    out = {}
    for f in features:
        x, v = X[f].astype(float), contrib[f]
        t = pd.DataFrame({"x": x.to_numpy(), "v": v.to_numpy()})
        miss = t["x"].isna()
        if f == "submit_rate_by_30":  # mostly 0 or 1: none, some, all handed in
            cuts = np.array([-0.001, 0.001, 0.999, 1.0])
        else:
            cuts = np.unique(np.nanquantile(t.loc[~miss, "x"], np.linspace(0, 1, 9)))
        t.loc[~miss, "band"] = pd.cut(t.loc[~miss, "x"], cuts, include_lowest=True, duplicates="drop").astype(str)
        g = t.loc[~miss].groupby("band", sort=False).agg(low=("x", "min"), high=("x", "max"), effect=("v", "mean"),
                                                        learners=("v", "size")).sort_values("low")
        g = g[g["learners"] >= MIN_CELL]
        rec = {"label": PLAIN.get(f, f), "bands": g.to_dict(orient="records")}
        if miss.sum() >= MIN_CELL:
            rec["missing"] = {"effect": float(t.loc[miss, "v"].mean()), "learners": int(miss.sum())}
        out[f] = rec
    return out


def pick_example_c(X, y, p, thr):
    """A non-completer the model missed although an assessment was due: the
    median-risk case among non-completers scored below 0.6 x threshold."""
    pool = np.flatnonzero((y == 1) & (p < 0.6 * thr) & (X["n_due_by_30"] > 0).to_numpy())
    pool = pool[np.argsort(p[pool])]
    return int(pool[len(pool) // 2])


def main(base: Path | None = None) -> None:
    root = Path(base) if base else ROOT
    out = root / "dashboard" / "data"
    out.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((root / "reports" / "metrics.json").read_text())
    fairness = json.loads((root / "reports" / "fairness.json").read_text())
    expl = json.loads((root / "reports" / "explanations.json").read_text())
    meta = json.loads((root / "models" / "model_metadata.json").read_text())
    model = joblib.load(root / "models" / "model.joblib")
    thr = float(meta["threshold"])

    df = data.load(root / "data" / "processed")
    split = data.grouped_holdout(df, seed=42)
    X, y = split.X_test, split.y_test
    p = model.predict_proba(X)[:, 1]
    sel = metrics["candidates"][metrics["selected"]]["test"]

    # ---- summary
    sens = metrics["sensitivity"]
    _write(out / "summary.json", {
        "model": {"algorithm": meta["algorithm"], "registry": meta.get("registry"), "threshold": thr,
                  "capacity": meta["capacity"], "data_sha256": meta["data_sha256"][:12]},
        "test": {"learners": int(len(y)), "non_completion_rate": float(y.mean()), "roc_auc": sel["roc_auc"],
                 "recall": sel["recall"], "precision": sel["precision"], "flag_rate": sel["flag_rate"],
                 "tp": sel["tp"], "fp": sel["fp"], "fn": sel["fn"], "tn": sel["tn"]},
        "baselines": {k: {"recall": v["recall"], "precision": v["precision"]} for k, v in metrics["baselines"].items()},
        "future_cohort": {"roc_auc": sens["time_aware"]["roc_auc"], "recall": sens["time_aware"]["recall"],
                          "learners": sens["time_aware"]["n_test"]},
        "capacity_curve": capacity_curve(y, p),
        "calibration": calibration(y, p),
        "risk_bands": risk_bands(y, p, thr),
    })

    # ---- drivers
    contrib, base_value = contributions(model, X)
    imp = contrib.abs().mean().sort_values(ascending=False)
    numeric_top = [f for f in imp.index if f != "code_module"][:4]
    _write(out / "drivers.json", {
        "importance": [{"feature": f, "label": PLAIN.get(f, f), "mean_abs_effect": float(v)} for f, v in imp.head(10).items()],
        "effects": effect_curves(contrib, X, numeric_top),
        "course_effects": {m: float(contrib.loc[(X["code_module"] == m).to_numpy(), "code_module"].mean())
                           for m in sorted(X["code_module"].unique())},
        "base_value": float(np.mean(base_value)),
    })

    # ---- segments (fitted on training data only)
    seg = TrajectorySegmenter().fit(split.X_train)
    s_test = seg.predict(X)
    s_sum = summary(s_test, y, p, thr, X)
    medians = pd.DataFrame(X[["clicks_wk1", "clicks_wk2", "clicks_wk3", "clicks_wk4"]].to_numpy(),
                           columns=["wk1", "wk2", "wk3", "wk4"]).assign(segment=s_test).groupby("segment").median().loc[ORDER]
    _write(out / "segments.json", {
        "model": seg.to_dict(),
        "summary": s_sum.reset_index().to_dict(orient="records"),
        "profile_z": seg.profile().reset_index(names="segment").to_dict(orient="records"),
        "median_weekly_clicks": medians.reset_index().to_dict(orient="records"),
        "description": DESCRIPTION,
    })

    # ---- fairness: score histograms by group and outcome (no cross-tabulation of attributes)
    hist = {}
    for att in data.PROTECTED:
        g = split.attrs_test[att].astype(str).to_numpy()
        hist[att] = {}
        for grp in sorted(np.unique(g)):
            m = g == grp
            hist[att][grp] = {str(lab): np.histogram(p[m & (y == lab)], bins=BINS)[0].tolist() for lab in (0, 1)}
    _write(out / "fairness.json", {"report": fairness, "bins": BINS.tolist(), "histograms": hist,
                                    "single_threshold": thr})

    # ---- examples
    a = expl["learner_a"]
    idx_a = int(np.flatnonzero((np.isclose(p, a["risk"], atol=1e-6)) & (X["code_module"] == a["features"]["code_module"]).to_numpy())[0])
    near = np.flatnonzero((np.abs(p - thr) < 0.03) & (X["n_due_by_30"] > 0).to_numpy())
    idx_b = int(near[0])
    idx_c = pick_example_c(X, y, p, thr)
    ex = {}
    def story(i):
        side = "flagged" if p[i] >= thr else ("just below the line, not flagged" if p[i] >= thr - 0.03 else "not flagged")
        return f"{side.capitalize()}; {'did not complete' if y[i] else 'completed'}"

    for key, i in [("A", idx_a), ("B", idx_b), ("C", idx_c)]:
        row = X.iloc[[i]]
        c = contrib.iloc[i]
        ex[key] = {"story": story(i), "risk": float(p[i]), "flagged": bool(p[i] >= thr), "outcome": int(y[i]),
                   "segment": str(s_test[i]),
                   "features": {k: (None if pd.isna(v) else (v if isinstance(v, str) else float(v)))
                                for k, v in row.iloc[0].to_dict().items()},
                   "reasons": [{"feature": f, "reason": PLAIN.get(f, f), "effect": float(c[f])}
                               for f in c.abs().sort_values(ascending=False).index[:6]]}
    if expl["learner_a"]["counterfactuals"]:
        ex["A"]["counterfactual"] = expl["learner_a"]["counterfactuals"][0]
    _write(out / "examples.json", ex)
    print(f"wrote dashboard data to {out}")


if __name__ == "__main__":
    main()
