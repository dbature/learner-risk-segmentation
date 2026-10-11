"""FastAPI service for the day-30 non-completion risk model.

    uvicorn src.serving.api:app --host 0.0.0.0 --port 8000

POST /predict            one learner's day-30 features -> risk score, flag, top reasons
GET  /health             liveness and the loaded model's identity
GET  /model-info         features, threshold, decision rule, registry version
GET  /explanations/{id}  the stored audit record for one earlier prediction

Every prediction is written to the explanation store
(src/governance/explanation_store.py) and its id is returned, so the reason
for any flag can be retrieved later for audit.

Decision rule. Without `sex`, the single capacity threshold is used. With
`sex`, the sex-specific equal-opportunity threshold is applied, which is the
mitigated rule described in the model card. Sex is never a model input and is
not stored or logged by this service.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.data import FEATURES

MODULES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]


class Learner(BaseModel):
    code_module: Literal["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
    num_of_prev_attempts: int = Field(ge=0, le=6)
    studied_credits: int = Field(ge=30, le=655)
    date_registration: Optional[float] = Field(default=None, ge=-400, le=200)
    n_due_by_30: int = Field(ge=0)
    n_submitted_by_30: int = Field(ge=0)
    n_banked_by_30: int = Field(default=0, ge=0)
    mean_score_by_30: Optional[float] = Field(default=None, ge=0, le=100)
    clicks_pre_start: int = Field(ge=0)
    clicks_0_29: int = Field(ge=0)
    clicks_wk1: int = Field(ge=0)
    clicks_wk2: int = Field(ge=0)
    clicks_wk3: int = Field(ge=0)
    clicks_wk4: int = Field(ge=0)
    active_days_0_29: int = Field(ge=0, le=30)
    distinct_sites_0_29: int = Field(ge=0)
    first_active_day: int = Field(ge=-1, le=29)
    sex: Optional[Literal["M", "F"]] = Field(default=None, description="Only used to apply the mitigated threshold")

    model_config = {"json_schema_extra": {"example": {
        "code_module": "FFF", "num_of_prev_attempts": 0, "studied_credits": 120, "date_registration": -22,
        "n_due_by_30": 1, "n_submitted_by_30": 0, "n_banked_by_30": 0, "mean_score_by_30": None,
        "clicks_pre_start": 81, "clicks_0_29": 555, "clicks_wk1": 432, "clicks_wk2": 67,
        "clicks_wk3": 56, "clicks_wk4": 0, "active_days_0_29": 15, "distinct_sites_0_29": 46,
        "first_active_day": 0, "sex": None}}}


class Prediction(BaseModel):
    risk_score: float
    flagged: bool
    threshold: float
    decision_rule: str
    top_factors: list[dict]
    model: dict
    explanation_id: Optional[str] = None


def model_dir() -> Path:
    return Path(os.environ.get("MODEL_DIR", "models"))


@lru_cache(maxsize=1)
def load():
    d = model_dir()
    model = joblib.load(d / "model.joblib")
    meta = json.loads((d / "model_metadata.json").read_text())
    return model, meta


def to_frame(x: Learner) -> pd.DataFrame:
    row = x.model_dump(exclude={"sex"})
    if row["n_submitted_by_30"] > row["n_due_by_30"]:
        raise HTTPException(422, "n_submitted_by_30 cannot exceed n_due_by_30")
    if row["clicks_wk1"] + row["clicks_wk2"] + row["clicks_wk3"] + row["clicks_wk4"] != row["clicks_0_29"]:
        raise HTTPException(422, "weekly clicks must sum to clicks_0_29")
    due = row["n_due_by_30"]
    row["submit_rate_by_30"] = (row["n_submitted_by_30"] / due) if due > 0 else None
    row["date_registration_missing"] = int(row["date_registration"] is None)
    row["no_vle_activity"] = int(row["clicks_0_29"] == 0 and row["clicks_pre_start"] == 0)
    df = pd.DataFrame([row])[FEATURES]
    num = [c for c in FEATURES if c != "code_module"]
    df[num] = df[num].astype(float)
    return df


def contributions(model, X: pd.DataFrame) -> dict:
    """Per-feature TreeSHAP contributions (log-odds) for a tree model; empty
    for other model types. Uses XGBoost's own pred_contribs, tested equal to
    the shap library's values, so the service does not need shap installed."""
    clf = model.named_steps.get("clf")
    if not hasattr(clf, "get_booster"):
        return {}
    from src.dashboard.contrib import contributions as tree_contributions

    g, _ = tree_contributions(model, X)
    return {k: float(v) for k, v in g.iloc[0].items()}


def top_factors(contrib: dict, k: int = 3) -> list[dict]:
    from src.dashboard.contrib import PLAIN

    top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:k]
    return [{"factor": PLAIN.get(f, f), "direction": "raises risk" if v > 0 else "lowers risk",
             "contribution": round(v, 3)} for f, v in top]


app = FastAPI(title="Learner risk: day-30 non-completion", version="1.0.0",
              description="BAN6800 capstone. Decision support for coaches, not an automated decision.")


@app.get("/health")
def health():
    _, meta = load()
    return {"status": "ok", "algorithm": meta.get("algorithm"),
            "registry": meta.get("registry"), "model_sha256": meta.get("model_sha256")}


@app.get("/model-info")
def model_info():
    _, meta = load()
    return {k: meta.get(k) for k in ["algorithm", "features", "target", "threshold", "group_thresholds",
                                     "capacity", "registry", "data_sha256"]}


@app.post("/predict", response_model=Prediction)
def predict(learner: Learner):
    from src.governance import explanation_store

    model, meta = load()
    X = to_frame(learner)
    score = float(model.predict_proba(X)[0, 1])
    groups = meta.get("group_thresholds") or {}
    if learner.sex and learner.sex in groups:
        thr, rule = float(groups[learner.sex]), "sex-specific equal-opportunity threshold (mitigated)"
    else:
        thr, rule = float(meta["threshold"]), "single capacity threshold"
    contrib = contributions(model, X)
    model_id = {"algorithm": meta.get("algorithm"), "version": (meta.get("registry") or {}).get("version"),
                "stage": (meta.get("registry") or {}).get("stage"), "sha256": meta.get("model_sha256")}
    rec_id = explanation_store.append(inputs=learner.model_dump(exclude={"sex"}), score=score, threshold=thr,
                                      rule=rule, flagged=score >= thr, contributions=contrib, model=model_id)
    return Prediction(
        risk_score=round(score, 4), flagged=score >= thr, threshold=round(thr, 4), decision_rule=rule,
        top_factors=top_factors(contrib),
        model={"algorithm": meta.get("algorithm"), "version": (meta.get("registry") or {}).get("version")},
        explanation_id=rec_id,
    )


@app.get("/explanations/{explanation_id}")
def explanation(explanation_id: str):
    """Audit retrieval by id only. There is no endpoint that lists records."""
    from src.governance import explanation_store

    rec = explanation_store.get(explanation_id)
    if rec is None:
        raise HTTPException(404, "no explanation with that id")
    return rec
