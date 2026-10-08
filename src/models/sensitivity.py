"""Sensitivity analysis: how much do the model's decisions move when the
inputs, the split or the random seed move?

1. Input noise. Click and activity counts are multiplied by random factors
   (plus or minus 10% and 25%). Clicks are logged imperfectly, so a model whose
   flags flip under small noise would not be trustworthy.
2. Missing assessment evidence. Every learner's early assessment fields are
   set to 'nothing due yet', as happens in modules whose first assessment
   falls after day 30.
3. Time. Train on earlier presentations, test on the latest (2014J), which is
   how the model would actually be used.
4. Seed. Retrain with three seeds and compare test AUC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from src.models.evaluate import capacity_threshold, classification_metrics

NOISY = ["clicks_pre_start", "clicks_0_29", "clicks_wk1", "clicks_wk2", "clicks_wk3", "clicks_wk4",
         "active_days_0_29", "distinct_sites_0_29"]
ASSESS = ["n_due_by_30", "n_submitted_by_30", "n_banked_by_30", "submit_rate_by_30", "mean_score_by_30"]


def input_noise(model, X: pd.DataFrame, y: np.ndarray, threshold: float, levels=(0.10, 0.25), seed: int = 7) -> list[dict]:
    base_p = model.predict_proba(X)[:, 1]
    base_flag = base_p >= threshold
    out = []
    rng = np.random.default_rng(seed)
    for lv in levels:
        Xn = X.copy()
        for c in NOISY:
            Xn[c] = np.round(Xn[c] * rng.uniform(1 - lv, 1 + lv, len(Xn)))
        Xn["active_days_0_29"] = Xn["active_days_0_29"].clip(0, 30)
        p = model.predict_proba(Xn)[:, 1]
        flag = p >= threshold
        out.append({"noise": lv, "auc": float(roc_auc_score(y, p)),
                    "flag_flip_rate": float(np.mean(flag != base_flag)),
                    "mean_abs_score_change": float(np.mean(np.abs(p - base_p)))})
    return out


def no_assessment_evidence(model, X: pd.DataFrame, y: np.ndarray, threshold: float) -> dict:
    Xn = X.copy()
    Xn[["n_due_by_30", "n_submitted_by_30", "n_banked_by_30"]] = 0.0
    Xn[["submit_rate_by_30", "mean_score_by_30"]] = np.nan
    p = model.predict_proba(Xn)[:, 1]
    return {"auc": float(roc_auc_score(y, p)),
            **{k: v for k, v in classification_metrics(y, p, threshold).items()
               if k in ("recall", "precision", "flag_rate")}}


def time_aware(model, split_time) -> dict:
    m = clone(model).fit(split_time.X_train, split_time.y_train)
    p = m.predict_proba(split_time.X_test)[:, 1]
    # deployment rule: flag the riskiest 25% of the cohort being scored (no labels used)
    res = classification_metrics(split_time.y_test, p, capacity_threshold(p))
    return {"n_train": int(len(split_time.y_train)), "n_test": int(len(split_time.y_test)),
            **{k: res[k] for k in ("roc_auc", "average_precision", "recall", "precision", "flag_rate")}}


def seed_stability(model, split, seeds=(1, 7, 2024)) -> list[dict]:
    out = []
    for s in seeds:
        m = clone(model).set_params(clf__random_state=s).fit(split.X_train, split.y_train)
        out.append({"seed": s, "auc": float(roc_auc_score(split.y_test, m.predict_proba(split.X_test)[:, 1]))})
    return out
