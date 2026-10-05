"""Stage 4: transform. Features observable on day 30.

Two feature families, both restricted to what a coach could actually know on
day 30 of a presentation:

* assessment behaviour: for non-exam assessments due by day 30, how many were
  due, how many the learner submitted by day 30, and the mean score of those
  submissions. Banked submissions (credit carried from an earlier attempt)
  count as submitted and are also counted separately.
* VLE engagement from the early-window aggregate. Registrations with no VLE
  activity before day 30 get zeros, first_active_day = -1 (never active, not
  "active on day 0"), and no_vle_activity = 1.

FEATURE_COLUMNS is the single list the model is allowed to see. The leakage
tests in tests/test_features.py import it directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import KEY

WINDOW_DAYS = 30

REGISTRATION_FEATURES = [
    "num_of_prev_attempts",
    "studied_credits",
    "date_registration",
    "date_registration_missing",
]
ASSESSMENT_FEATURES = [
    "n_due_by_30",
    "n_submitted_by_30",
    "n_banked_by_30",
    "submit_rate_by_30",
    "mean_score_by_30",
]
VLE_FEATURES = [
    "clicks_pre_start", "clicks_0_29",
    "clicks_wk1", "clicks_wk2", "clicks_wk3", "clicks_wk4",
    "active_days_0_29", "distinct_sites_0_29", "first_active_day",
    "no_vle_activity",
]
CONTEXT_COLUMNS = ["code_module", "code_presentation"]

FEATURE_COLUMNS = REGISTRATION_FEATURES + ASSESSMENT_FEATURES + VLE_FEATURES

LABEL_COLUMNS = ["label_withdrew_after_30", "label_non_completion"]

# Never allowed into FEATURE_COLUMNS. Each either is the outcome or is
# derived from it.
FORBIDDEN_FEATURES = {
    "date_unregistration", "final_result", "withdrew",
    "label_withdrew_after_30", "label_non_completion", "id_student",
}


def assessment_features(
    assessments: pd.DataFrame,
    student_assessment: pd.DataFrame,
    registrations: pd.DataFrame,
    window: int = WINDOW_DAYS,
) -> pd.DataFrame:
    early = assessments[
        assessments["assessment_type"].ne("Exam")
        & assessments["date"].notna()
        & (assessments["date"] <= window)
    ]
    due = (
        early.groupby(CONTEXT_COLUMNS).size().rename("n_due_by_30").reset_index()
    )
    subs = student_assessment.merge(
        early[["id_assessment"] + CONTEXT_COLUMNS], on="id_assessment", how="inner"
    )
    subs = subs[subs["date_submitted"] <= window]
    per_student = (
        subs.groupby(KEY)
        .agg(
            n_submitted_by_30=("id_assessment", "nunique"),
            n_banked_by_30=("is_banked", "sum"),
            mean_score_by_30=("score", "mean"),
        )
        .reset_index()
    )
    out = registrations[KEY].merge(due, on=CONTEXT_COLUMNS, how="left")
    out = out.merge(per_student, on=KEY, how="left", validate="1:1")
    out["n_due_by_30"] = out["n_due_by_30"].fillna(0).astype("int64")
    out["n_submitted_by_30"] = out["n_submitted_by_30"].fillna(0).astype("int64")
    out["n_banked_by_30"] = out["n_banked_by_30"].fillna(0).astype("int64")
    out["submit_rate_by_30"] = np.where(
        out["n_due_by_30"] > 0, out["n_submitted_by_30"] / out["n_due_by_30"].replace(0, np.nan), np.nan
    )
    out["mean_score_by_30"] = out["mean_score_by_30"].astype("float64")
    return out


def vle_features(vle_window: pd.DataFrame, registrations: pd.DataFrame) -> pd.DataFrame:
    out = registrations[KEY].merge(vle_window, on=KEY, how="left", validate="1:1")
    out["no_vle_activity"] = out["clicks_0_29"].isna().astype("int64")
    counts = [c for c in VLE_FEATURES if c not in ("first_active_day", "no_vle_activity")]
    out[counts] = out[counts].fillna(0).astype("int64")
    out["first_active_day"] = out["first_active_day"].fillna(-1).astype("int64")
    return out
