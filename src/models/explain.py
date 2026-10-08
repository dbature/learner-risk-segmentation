"""Explainability: SHAP (global and local), LIME (local), DiCE (counterfactual).

* SHAP values come from TreeExplainer on the fitted XGBoost inside the
  pipeline (Lundberg and Lee, 2017). They are computed on the transformed
  feature space, then reported under readable names.
* LIME fits a local surrogate around one learner (Ribeiro et al., 2016). It
  is a second, independent method, so agreement with SHAP is evidence and
  disagreement is a warning.
* DiCE searches for small changes that would move a flagged learner below
  the threshold (Mothilal et al., 2020). Only behaviours a coach could
  influence are allowed to vary: no change to module, credits, prior
  attempts or registration date.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.models.evaluate import GREY, INK, NAVY  # noqa: E402

READABLE = {
    "mean_score_by_30": "Mean score, assessments by day 30",
    "submit_rate_by_30": "Share of early assessments submitted",
    "n_submitted_by_30": "Assessments submitted by day 30",
    "n_due_by_30": "Assessments due by day 30",
    "n_banked_by_30": "Banked assessments",
    "clicks_0_29": "VLE clicks, days 0 to 29",
    "clicks_pre_start": "VLE clicks before start",
    "clicks_wk1": "Clicks, week 1", "clicks_wk2": "Clicks, week 2",
    "clicks_wk3": "Clicks, week 3", "clicks_wk4": "Clicks, week 4",
    "active_days_0_29": "Active days, days 0 to 29",
    "distinct_sites_0_29": "Distinct resources used",
    "first_active_day": "First active day",
    "no_vle_activity": "No VLE activity before day 30",
    "studied_credits": "Credits studied",
    "num_of_prev_attempts": "Previous attempts",
    "date_registration": "Registration day (relative to start)",
    "date_registration_missing": "Registration date missing",
}

ACTIONABLE = ["n_submitted_by_30", "clicks_0_29", "active_days_0_29", "distinct_sites_0_29",
              "clicks_wk3", "clicks_wk4"]


def readable(name: str) -> str:
    if name.startswith("missingindicator_"):
        base = name.replace("missingindicator_", "")
        return f"{READABLE.get(base, base)} is missing"
    if name.startswith("code_module_"):
        return "Module " + name.replace("code_module_", "")
    return READABLE.get(name, name)


SENTINEL = {"date_registration": -999.0, "submit_rate_by_30": -1.0, "mean_score_by_30": -1.0}


def encode_missing(X: pd.DataFrame) -> pd.DataFrame:
    """LIME and DiCE cannot take NaN. Missing values become an out-of-range
    sentinel, decoded back to NaN before the pipeline sees them, so the model's
    own missing-value handling is preserved."""
    X = X.copy()
    for c, v in SENTINEL.items():
        X[c] = X[c].fillna(v)
    return X


def decode_missing(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c, v in SENTINEL.items():
        X[c] = X[c].astype(float).where(X[c].astype(float) != v, np.nan)
    return X


def shap_values(pipeline, X: pd.DataFrame):
    """SHAP on the transformed space; displayed with the original, unscaled values."""
    import shap

    prep, clf = pipeline.named_steps["prep"], pipeline.named_steps["clf"]
    names = list(prep.get_feature_names_out())
    Xt = pd.DataFrame(prep.transform(X), columns=names)
    sv = shap.TreeExplainer(clf)(Xt)
    display = Xt.copy()
    for c in names:
        if c in X.columns:
            display[c] = X[c].to_numpy()
    sv = shap.Explanation(values=sv.values, base_values=sv.base_values,
                          data=display.to_numpy(dtype=float, na_value=np.nan), feature_names=names)
    return sv, Xt


def global_importance(sv) -> pd.DataFrame:
    imp = np.abs(sv.values).mean(axis=0)
    df = pd.DataFrame({"feature": sv.feature_names, "mean_abs_shap": imp})
    df["label"] = df["feature"].map(readable)
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def plot_global(imp: pd.DataFrame, out: Path, top: int = 12) -> Path:
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.barh(d["label"], d["mean_abs_shap"], color=NAVY)
    ax.set_xlabel("Mean |SHAP value| (log-odds of non-completion)")
    ax.set_title("Global feature importance, test set", loc="left", fontsize=11, color=INK, fontweight="bold")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9, colors=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_beeswarm(sv, out: Path) -> Path:
    import shap

    sv = shap.Explanation(values=sv.values, base_values=sv.base_values, data=sv.data,
                          feature_names=[readable(f) for f in sv.feature_names])
    plt.figure(figsize=(7.6, 5.2))
    shap.plots.beeswarm(sv, max_display=12, show=False, color_bar=True)
    plt.title("SHAP summary, test set", loc="left", fontsize=11)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


def plot_waterfall(sv, i: int, title: str, out: Path) -> Path:
    import shap

    one = shap.Explanation(values=sv.values[i], base_values=sv.base_values[i], data=sv.data[i],
                           feature_names=[readable(f) for f in sv.feature_names])
    plt.figure(figsize=(7.4, 4.6))
    shap.plots.waterfall(one, max_display=9, show=False)
    plt.title(title, loc="left", fontsize=11, pad=22)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


def local_reasons(sv, i: int, k: int = 6) -> list[dict]:
    v = sv.values[i]
    order = np.argsort(-np.abs(v))[:k]
    return [{"feature": sv.feature_names[j], "label": readable(sv.feature_names[j]),
             "value": float(sv.data[i][j]), "shap": float(v[j])} for j in order]


def lime_explanation(pipeline, X_train: pd.DataFrame, x_row: pd.DataFrame, seed: int = 42) -> list[dict]:
    """LIME on the original feature scale, so its rules read in clicks, days and scores."""
    from lime.lime_tabular import LimeTabularExplainer

    from src.models.data import FEATURES

    modules = sorted(X_train["code_module"].unique())
    code = {m: i for i, m in enumerate(modules)}

    def to_matrix(X):
        X = encode_missing(X[FEATURES])
        X["code_module"] = X["code_module"].map(code)
        return X.to_numpy(dtype=float)

    def predict(M):
        X = pd.DataFrame(M, columns=FEATURES)
        X["code_module"] = [modules[int(round(v))] for v in X["code_module"].clip(0, len(modules) - 1)]
        X = decode_missing(X)
        return pipeline.predict_proba(X)

    explainer = LimeTabularExplainer(
        to_matrix(X_train), feature_names=[readable(f) for f in FEATURES], class_names=["completes", "non-completion"],
        categorical_features=[0], categorical_names={0: modules}, mode="classification",
        discretize_continuous=True, random_state=seed)
    exp = explainer.explain_instance(to_matrix(x_row)[0], predict, num_features=6)
    out = []
    for rule, w in exp.as_list():
        for c, v in SENTINEL.items():
            marker = f"{readable(c)} <= {v:.2f}"
            if rule.startswith(marker):
                rule = f"{readable(c)} is missing"
        out.append({"rule": rule, "weight": float(w)})
    return out


WEEK_DAYS = {"clicks_wk1": (0, 6), "clicks_wk2": (7, 13), "clicks_wk3": (14, 20), "clicks_wk4": (21, 29)}


def feasible(cf: pd.DataFrame, original: pd.Series) -> pd.Series:
    """Rows a learner could actually have produced. DiCE varies features
    independently, so it can propose, say, 29 active days when only week 4
    gained clicks. Clicks in weeks 1 and 2 are history and cannot change, so
    new active days can only come from clicks added in weeks 3 and 4, and a
    week can add no more active days than it has days or new clicks. Any
    activity needs at least one site, and adding clicks to a week before the
    learner's first active day would contradict that (fixed) field."""
    weeks = list(WEEK_DAYS)
    gain = sum(np.minimum(hi - lo + 1, (cf[w] - float(original[w])).clip(lower=0))
               for w, (lo, hi) in WEEK_DAYS.items())
    max_days = (float(original["active_days_0_29"]) + gain).clip(upper=30)
    active_weeks = (cf[weeks] > 0).sum(axis=1)
    first = float(original["first_active_day"])
    early_ok = pd.Series(True, index=cf.index)
    for w, (lo, hi) in WEEK_DAYS.items():
        if first >= 0 and hi < first:
            early_ok &= cf[w] <= float(original[w])
    has = cf["clicks_0_29"] > 0
    return ((cf["active_days_0_29"] <= max_days) & (cf["active_days_0_29"] >= active_weeks)
            & (cf["active_days_0_29"] <= cf["clicks_0_29"])
            & (~has | (cf["distinct_sites_0_29"] >= 1)) & (has == (cf["active_days_0_29"] > 0))
            & (cf["distinct_sites_0_29"] <= cf["clicks_0_29"]) & early_ok)


def counterfactuals(pipeline, X_train: pd.DataFrame, y_train: np.ndarray, x_row: pd.DataFrame,
                    threshold: float, n: int = 3, seed: int = 42) -> pd.DataFrame:
    """Smallest behaviour changes that take a flagged learner below threshold."""
    import dice_ml

    from src.models.data import CATEGORICAL, FEATURES, NUMERIC

    train = encode_missing(X_train)
    train["outcome"] = y_train
    d = dice_ml.Data(dataframe=train, continuous_features=NUMERIC, categorical_features=CATEGORICAL,
                     outcome_name="outcome")

    class Wrapped:
        def __init__(self, pipe, thr):
            self.pipe, self.thr = pipe, thr

        def predict_proba(self, X):
            s = self.pipe.predict_proba(decode_missing(X[FEATURES]))[:, 1]
            # rescale so DiCE's 0.5 boundary sits at the operating threshold
            z = np.where(s >= self.thr, 0.5 + 0.5 * (s - self.thr) / (1 - self.thr),
                         0.5 * s / self.thr)
            return np.c_[1 - z, z]

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    m = dice_ml.Model(model=Wrapped(pipeline, threshold), backend="sklearn")
    exp = dice_ml.Dice(d, m, method="random")
    def upper(c, q=0.95):
        # never below the learner's own value, so the range is always valid
        return max(float(X_train[c].quantile(q)), float(x_row[c].iloc[0]))

    ranges = {
        "n_submitted_by_30": [float(x_row["n_submitted_by_30"].iloc[0]), float(x_row["n_due_by_30"].iloc[0])],
        "active_days_0_29": [float(x_row["active_days_0_29"].iloc[0]), 30.0],
        **{c: [float(x_row[c].iloc[0]), upper(c)]
           for c in ["clicks_0_29", "distinct_sites_0_29", "clicks_wk3", "clicks_wk4"]},
    }
    from raiutils.exceptions import UserConfigValidationException

    try:
        res = exp.generate_counterfactuals(encode_missing(x_row[FEATURES]), total_CFs=4 * n, desired_class=0,
                                           features_to_vary=ACTIONABLE, permitted_range=ranges,
                                           random_seed=seed)
    except UserConfigValidationException:
        # DiCE raises when it finds nothing; for this learner no actionable change
        # inside the permitted ranges crosses the threshold
        return pd.DataFrame()
    cf = res.cf_examples_list[0].final_cfs_df
    if cf is None or cf.empty:
        return pd.DataFrame()
    cf = decode_missing(cf[FEATURES].copy())
    for c in NUMERIC:
        cf[c] = cf[c].astype(float)
    # keep each counterfactual internally consistent: whole counts, and total
    # clicks equal to the sum of the weekly clicks, then re-score
    counts = [c for c in ACTIONABLE]
    cf[counts] = cf[counts].round()
    cf["active_days_0_29"] = cf["active_days_0_29"].clip(0, 30)
    cf["clicks_0_29"] = cf[["clicks_wk1", "clicks_wk2", "clicks_wk3", "clicks_wk4"]].sum(axis=1)
    cf = cf[feasible(cf, x_row.iloc[0])].drop_duplicates(subset=ACTIONABLE)
    if cf.empty:
        return pd.DataFrame()
    cf["risk_score"] = pipeline.predict_proba(cf[FEATURES])[:, 1]
    cf["below_threshold"] = cf["risk_score"] < threshold
    return cf.sort_values("risk_score").head(n).reset_index(drop=True)


def pick_learner_a(pipeline, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame,
                   y_test: np.ndarray, scores: np.ndarray, threshold: float, max_tries: int = 25):
    """Learner A for the local explanation: a correctly flagged non-completer
    with at least one assessment due and some VLE activity in days 0 to 29
    (so that the fixed fields first_active_day and no_vle_activity stay true
    when clicks change), taken from the upper quartile of risk
    among such learners (typical of a coach's list, not the most extreme
    case). DiCE cannot always find an actionable change that crosses the
    threshold, so candidates are tried in order, starting at that point and
    moving towards lower risk, until one has a valid counterfactual that is
    still below the threshold after rounding and passes feasible(). Returns (index, counterfactuals,
    tries); the number of tries is reported so the choice stays transparent."""
    pool = np.flatnonzero((scores >= threshold) & (y_test == 1) & (X_test["n_due_by_30"] > 0).to_numpy()
                          & (X_test["clicks_0_29"] > 0).to_numpy())
    pool = pool[np.argsort(-scores[pool])]
    start = len(pool) // 4
    order = list(pool[start:]) + list(pool[:start][::-1])
    for tries, i in enumerate(order[:max_tries], start=1):
        cf = counterfactuals(pipeline, X_train, y_train, X_test.iloc[[int(i)]], threshold)
        if not cf.empty and bool(cf["below_threshold"].iloc[0]):
            return int(i), cf, tries
    i = int(order[0])
    return i, pd.DataFrame(), min(max_tries, len(order))


def plot_counterfactual(original: pd.Series, cf: pd.DataFrame, orig_score: float, out: Path) -> Path:
    best = cf.iloc[0]
    rows = [f for f in ACTIONABLE + ["clicks_0_29"] if f in cf and not np.isclose(float(best[f]), float(original[f]))]
    rows = list(dict.fromkeys(rows))
    fig, axes = plt.subplots(len(rows), 1, figsize=(7.2, 0.9 + 0.75 * len(rows)), squeeze=False)
    for ax, f in zip(axes[:, 0], rows):
        a, b = float(original[f]), float(best[f])
        ax.plot([a, b], [0, 0], color=GREY, lw=2, zorder=1)
        ax.scatter([a], [0], color=GREY, s=60, zorder=2)
        ax.scatter([b], [0], color=NAVY, s=60, zorder=2)
        ax.text(a, 0.25, f"{a:g}", ha="center", fontsize=9, color=GREY)
        ax.text(b, 0.25, f"{b:g}", ha="center", fontsize=9, color=NAVY)
        ax.set_yticks([0], [readable(f)])
        ax.set_ylim(-0.5, 0.7)
        ax.set_xlim(min(a, b) - 0.1 * abs(b - a) - 1, max(a, b) + 0.1 * abs(b - a) + 1)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=9, colors=INK)
    axes[0, 0].set_title(f"Counterfactual: grey is what happened (risk {orig_score:.2f}), "
                         f"blue would have scored {best['risk_score']:.2f}", loc="left", fontsize=10,
                         color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out
