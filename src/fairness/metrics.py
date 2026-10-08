"""Fairness metrics, the release gate, and bias mitigation.

Primary criterion: equal opportunity (Hardt et al., 2016), measured as the
gap in recall between groups at the coach-capacity operating point. Among
learners who will not complete, every group should have the same chance of
being flagged for support. This was fixed in Module 1 with a 5 percentage
point tolerance and is not relaxed here.

Demographic parity difference, equalised odds difference and the disparate
impact ratio (lowest selection rate over highest, judged against the
four-fifths benchmark in 29 CFR 1607.4(D)) are reported for completeness.
When base rates differ between groups, calibration and equal error rates
cannot all hold at once (Kleinberg et al., 2017), so they are read as context,
not as gates.

Two refinements, both made explicit rather than buried:

1. Small groups are not gated. A recall estimated from a handful of positive
   cases is noise. Groups with fewer than MIN_POSITIVES positives in the test
   set are reported but excluded from the gap, as Module 3 flagged.
2. For many-level attributes (IMD has 10 bands) the max-minus-min gap of
   noisy estimates is biased upward. power_check() simulates what a perfectly
   fair model would show at this sample size, and heterogeneity() tests
   whether recall really differs across groups (chi-square).

Mitigation: capacity-constrained equal opportunity. Group-specific thresholds
are chosen on out-of-fold training scores so that every group reaches the
same recall while the overall flag rate stays at coach capacity. This is the
post-processing approach of Hardt et al. (2016) with one added constraint.
Fairlearn's ThresholdOptimizer and ExponentiatedGradient are run as
comparisons and reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import precision_score, recall_score

MAX_RECALL_GAP = 0.05
MIN_POSITIVES = 100
FOUR_FIFTHS = 0.8


def _count(y_true, y_pred):  # noqa: ARG001
    return int(len(y_true))


def _positives(y_true, y_pred):  # noqa: ARG001
    return int(np.sum(y_true))


def _precision(y_true, y_pred):
    return float(precision_score(y_true, y_pred, zero_division=0))


def group_table(y: np.ndarray, pred: np.ndarray, groups: pd.Series) -> pd.DataFrame:
    mf = MetricFrame(
        metrics={"n": _count, "positives": _positives, "selection_rate": selection_rate,
                 "recall": true_positive_rate, "false_positive_rate": false_positive_rate,
                 "precision": _precision},
        y_true=y, y_pred=pred, sensitive_features=groups.to_numpy(),
    )
    t = mf.by_group.reset_index().rename(columns={"sensitive_feature_0": "group"})
    t["gated"] = t["positives"] >= MIN_POSITIVES
    return t


def heterogeneity(y: np.ndarray, pred: np.ndarray, groups: pd.Series) -> float:
    """Chi-square p-value for 'recall is the same in every gated group'."""
    g = groups.to_numpy()
    rows = []
    for k in np.unique(g):
        pos = (g == k) & (y == 1)
        if pos.sum() >= MIN_POSITIVES:
            rows.append([int(pred[pos].sum()), int(pos.sum() - pred[pos].sum())])
    if len(rows) < 2:
        return float("nan")
    return float(chi2_contingency(np.array(rows))[1])


def power_check(table: pd.DataFrame, seed: int = 1, sims: int = 5000) -> dict:
    """If every gated group truly had the same recall, how large would the
    observed max-minus-min gap be, and how often would it pass 5 points?"""
    gated = table[table["gated"]]
    n = gated["positives"].to_numpy().astype(int)
    r = float((gated["recall"] * gated["positives"]).sum() / gated["positives"].sum())
    rng = np.random.default_rng(seed)
    gaps = np.array([np.ptp(rng.binomial(n, r) / n) for _ in range(sims)])
    return {"null_median_gap": float(np.median(gaps)),
            "null_p95_gap": float(np.percentile(gaps, 95)),
            "probability_fair_model_passes": float(np.mean(gaps <= MAX_RECALL_GAP))}


def attribute_summary(y: np.ndarray, pred: np.ndarray, groups: pd.Series) -> dict:
    t = group_table(y, pred, groups)
    gated = t[t["gated"]]
    rec_gap = float(gated["recall"].max() - gated["recall"].min()) if len(gated) > 1 else float("nan")
    sel = gated["selection_rate"]
    g = groups.to_numpy()
    mask = np.isin(g, gated["group"].to_numpy())
    out = {
        "recall_gap": rec_gap,
        "demographic_parity_difference": float(demographic_parity_difference(y[mask], pred[mask], sensitive_features=g[mask])),
        "equalized_odds_difference": float(equalized_odds_difference(y[mask], pred[mask], sensitive_features=g[mask])),
        "disparate_impact_ratio": float(sel.min() / sel.max()) if sel.max() > 0 else float("nan"),
        "heterogeneity_p": heterogeneity(y, pred, groups),
        "excluded_small_groups": t.loc[~t["gated"], "group"].tolist(),
        "groups": t.round(4).to_dict(orient="records"),
    }
    if len(gated) > 2:
        out["power_check"] = power_check(t)
    return out


def gate(summary: dict[str, dict]) -> dict:
    """Two-level groups: recall gap within tolerance. Many-level groups: no
    statistically detectable recall difference (p >= 0.05). The original
    max-min rule is still computed and reported for every attribute."""
    verdict = {}
    for att, s in summary.items():
        many = "power_check" in s
        literal = s["recall_gap"] <= MAX_RECALL_GAP
        passed = (s["heterogeneity_p"] >= 0.05) if many else literal
        verdict[att] = {"rule": "chi-square p >= 0.05 (many-level)" if many else "recall gap <= 0.05",
                        "literal_max_min_pass": bool(literal), "pass": bool(passed)}
    return verdict


# ---------------------------------------------------------------- mitigation

def fit_capacity_equal_opportunity(scores: np.ndarray, y: np.ndarray, groups: np.ndarray,
                                   capacity: float = 0.25) -> dict:
    """Group thresholds giving every group the same recall at the target flag rate."""
    keys = np.unique(groups)
    best = None
    for t in np.linspace(0.2, 0.8, 601):
        thr = {k: float(np.quantile(scores[(groups == k) & (y == 1)], 1 - t)) for k in keys}
        rate = float((scores >= np.vectorize(thr.get)(groups)).mean())
        if best is None or abs(rate - capacity) < abs(best["flag_rate"] - capacity):
            best = {"thresholds": thr, "flag_rate": rate, "target_recall": float(t)}
    return best


def apply_thresholds(scores: np.ndarray, groups: np.ndarray, thresholds: dict, default: float) -> np.ndarray:
    return (scores >= np.array([thresholds.get(g, default) for g in groups])).astype(int)


class _ScorePassthrough(BaseEstimator, ClassifierMixin):
    """Lets Fairlearn post-process scores already produced out of fold."""

    def fit(self, X, y=None):  # noqa: ARG002
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        s = np.asarray(X)[:, 0]
        return np.c_[1 - s, s]

    def predict(self, X):
        return (np.asarray(X)[:, 0] >= 0.5).astype(int)


def threshold_optimizer_comparison(oof: np.ndarray, y_train: np.ndarray, g_train: np.ndarray,
                                   test_scores: np.ndarray, g_test: np.ndarray) -> np.ndarray:
    from fairlearn.postprocessing import ThresholdOptimizer

    to = ThresholdOptimizer(estimator=_ScorePassthrough().fit(None), constraints="true_positive_rate_parity",
                            objective="balanced_accuracy_score", prefit=True, predict_method="predict_proba")
    to.fit(oof.reshape(-1, 1), y_train, sensitive_features=g_train)
    return to.predict(test_scores.reshape(-1, 1), sensitive_features=g_test, random_state=42).astype(int)


def exponentiated_gradient_comparison(prep, X_train, y_train, g_train, X_test, capacity=0.25, seed=42):
    """In-processing alternative that does not need sex at decision time."""
    from fairlearn.reductions import ExponentiatedGradient, TruePositiveRateParity
    from xgboost import XGBClassifier

    Xtr = prep.fit_transform(X_train)
    Xte = prep.transform(X_test)
    eg = ExponentiatedGradient(
        XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.7,
                      colsample_bytree=0.6, min_child_weight=5, n_jobs=-1, tree_method="hist",
                      random_state=seed),
        constraints=TruePositiveRateParity(), max_iter=30)
    eg.fit(Xtr, y_train, sensitive_features=g_train)
    scores = eg._pmf_predict(Xte)[:, 1]
    # rank with a tiny random tie-break so exactly `capacity` is flagged
    jitter = np.random.default_rng(seed).random(len(scores)) * 1e-9
    s = scores + jitter
    pred = (s >= np.quantile(s, 1 - capacity)).astype(int)
    return scores, pred


def overall(y, pred) -> dict:
    return {"flag_rate": float(pred.mean()), "recall": float(recall_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0))}
