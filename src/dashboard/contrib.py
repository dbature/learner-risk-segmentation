"""Plain-language reasons for one prediction, for the stakeholder dashboard.

Contributions are XGBoost's own TreeSHAP values (pred_contribs), which are the
same numbers the shap library returns for this model (checked in the tests),
without needing shap and numba on the dashboard host. Columns the pipeline
creates are folded back into the feature a person would recognise: the
seven module columns become "Course", and a missing-value indicator is added
to its base feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PLAIN = {
    "submit_rate_by_30": "Share of early assignments handed in",
    "mean_score_by_30": "Average mark on early assignments",
    "n_submitted_by_30": "Early assignments handed in",
    "n_due_by_30": "Early assignments due",
    "n_banked_by_30": "Marks carried over from a previous attempt",
    "active_days_0_29": "Days active online in the first month",
    "clicks_0_29": "Online activity in the first month",
    "clicks_wk1": "Online activity, week 1",
    "clicks_wk2": "Online activity, week 2",
    "clicks_wk3": "Online activity, week 3",
    "clicks_wk4": "Online activity, week 4",
    "clicks_pre_start": "Online activity before the course started",
    "distinct_sites_0_29": "Number of different course resources used",
    "first_active_day": "Day of first online activity",
    "no_vle_activity": "No online activity at all",
    "studied_credits": "Study load (credits)",
    "num_of_prev_attempts": "Previous attempts at this course",
    "date_registration": "How early they registered",
    "date_registration_missing": "Registration date not recorded",
    "code_module": "Which course they are on",
}


def _base(col: str) -> str:
    if col.startswith("code_module_"):
        return "code_module"
    if col.startswith("missingindicator_"):
        return col.replace("missingindicator_", "")
    return col


def contributions(pipeline, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-feature contributions in log-odds (rows = learners) and the base value."""
    import xgboost as xgb

    prep, clf = pipeline.named_steps["prep"], pipeline.named_steps["clf"]
    cols = list(prep.get_feature_names_out())
    raw = clf.get_booster().predict(xgb.DMatrix(prep.transform(X), feature_names=cols), pred_contribs=True)
    df = pd.DataFrame(raw[:, :-1], columns=cols)
    grouped = df.T.groupby(_base).sum().T
    return grouped, raw[:, -1]


def reasons(pipeline, X: pd.DataFrame, k: int = 5) -> list[list[dict]]:
    """Top-k plain-language reasons per learner, largest effect first."""
    g, _ = contributions(pipeline, X)
    out = []
    for i in range(len(g)):
        row = g.iloc[i]
        top = row.reindex(row.abs().sort_values(ascending=False).index)[:k]
        out.append([{"feature": f, "reason": PLAIN.get(f, f), "effect": float(v),
                     "direction": "raises risk" if v > 0 else "lowers risk"} for f, v in top.items()])
    return out
