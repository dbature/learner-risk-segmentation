"""Coach caseload (ticket #19): rank an uploaded cohort under a capacity limit.

A coach supplies one row per learner with the day-30 fields the /predict API
takes, plus an optional learner_ref of their own. Every row goes through the
API's own schema and checks (src/serving/api.py), so a row the API would
reject is reported, not silently scored. Valid rows are scored, ranked by
risk, and the top share that fits the coach's capacity forms the caseload.

Nothing here writes to disk. The dashboard keeps the uploaded table in the
viewer's session only.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.dashboard.contrib import PLAIN, contributions

INPUT_FIELDS = ["code_module", "num_of_prev_attempts", "studied_credits", "date_registration", "n_due_by_30",
                "n_submitted_by_30", "n_banked_by_30", "mean_score_by_30", "clicks_pre_start", "clicks_0_29",
                "clicks_wk1", "clicks_wk2", "clicks_wk3", "clicks_wk4", "active_days_0_29", "distinct_sites_0_29",
                "first_active_day"]
OPTIONAL = {"date_registration", "mean_score_by_30", "n_banked_by_30"}
FLOATS = {"date_registration", "mean_score_by_30"}
REF = "learner_ref"


def template() -> pd.DataFrame:
    return pd.DataFrame(columns=[REF] + INPUT_FIELDS)


def _payload(row: pd.Series) -> dict:
    out = {}
    for c in INPUT_FIELDS:
        v = row.get(c)
        if v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip()):
            if c in OPTIONAL:
                out[c] = 0 if c == "n_banked_by_30" else None
                continue
            raise ValueError(f"{c} is missing")
        if c == "code_module":
            out[c] = str(v).strip().upper()
        elif c in FLOATS:
            out[c] = float(v)
        else:
            f = float(v)
            if f != int(f):
                raise ValueError(f"{c} must be a whole number")
            out[c] = int(f)
    return out


def validate(cohort: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Model-ready frame for valid rows (index = input row) and a list of errors."""
    from fastapi import HTTPException
    from pydantic import ValidationError

    from src.serving.api import Learner, to_frame

    missing = [c for c in INPUT_FIELDS if c not in cohort.columns and c not in OPTIONAL]
    if missing:
        raise ValueError("missing columns: " + ", ".join(missing))
    frames, errors = [], []
    for i, row in cohort.iterrows():
        ref = str(row.get(REF, "")) if pd.notna(row.get(REF, None)) else f"row {i + 2}"
        try:
            X = to_frame(Learner(**_payload(row)))
            X.index = [i]
            frames.append(X)
        except ValidationError as e:
            first = e.errors()[0]
            errors.append({"row": i + 2, REF: ref, "problem": f"{'.'.join(map(str, first['loc']))}: {first['msg']}"})
        except HTTPException as e:
            errors.append({"row": i + 2, REF: ref, "problem": str(e.detail)})
        except ValueError as e:
            errors.append({"row": i + 2, REF: ref, "problem": str(e)})
    return (pd.concat(frames) if frames else pd.DataFrame()), errors


def rank(cohort: pd.DataFrame, model, segmenter, capacity: float) -> tuple[pd.DataFrame, list[dict]]:
    """Ranked list of valid learners with risk, caseload flag, segment and top reasons."""
    X, errors = validate(cohort)
    if X.empty:
        return pd.DataFrame(), errors
    risk = model.predict_proba(X)[:, 1]
    contrib, _ = contributions(model, X)
    reasons = []
    for i in range(len(contrib)):
        row = contrib.iloc[i]
        top = row.reindex(row.abs().sort_values(ascending=False).index)[:3]
        reasons.append("; ".join(f"{PLAIN.get(f, f)} ({'raises' if v > 0 else 'lowers'} risk)" for f, v in top.items()))
    refs = cohort.loc[X.index, REF].astype(str) if REF in cohort else pd.Series([f"row {i + 2}" for i in X.index], index=X.index)
    out = pd.DataFrame({REF: refs.to_numpy(), "course": X["code_module"].to_numpy(), "risk": risk,
                        "segment": segmenter.predict(X), "top_reasons": reasons}, index=X.index)
    out = out.sort_values("risk", ascending=False, kind="mergesort").reset_index(drop=True)
    k = int(math.ceil(capacity * len(out)))
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out["in_caseload"] = out["rank"] <= k
    return out, errors
