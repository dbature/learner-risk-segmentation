"""Module 4 end to end: train, validate, explain, audit, mitigate, document.

    python -m src.models.run_all

Reads data/processed (the Module 3 output). Writes:
  mlruns/                      MLflow tracking store and registry (sqlite)
  models/model.joblib          the selected pipeline, with model_metadata.json
  reports/metrics.json         baselines, candidates, sensitivity
  reports/fairness.json        fairness before and after mitigation, gate
  reports/explanations.json    SHAP and LIME reasons, counterfactuals
  reports/figures/*.png        every figure used in the validation report
  docs/model_card.md           Mitchell et al. (2019) model card
"""
from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path

os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
warnings.filterwarnings("ignore")
logging.getLogger("alembic").setLevel(logging.WARNING)
logging.getLogger("mlflow").setLevel(logging.WARNING)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import ROOT, load_params  # noqa: E402
from src.fairness import metrics as fm  # noqa: E402
from src.models import data, explain, sensitivity, train  # noqa: E402
from src.models import evaluate as ev  # noqa: E402
from src.models.model_card import write_model_card  # noqa: E402
from src.models.preprocess import preprocessor  # noqa: E402


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def plot_fairness(before: dict, after: dict, out: Path) -> Path:
    atts = ["gender", "age_band", "disability", "imd_band"]
    titles = {"gender": "Sex", "age_band": "Age band", "disability": "Disability", "imd_band": "IMD band"}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1, 1, 1, 3.2]})
    for ax, att in zip(axes, atts):
        b = pd.DataFrame(before[att]["groups"])
        a = pd.DataFrame(after[att]["groups"])
        b = b[b["gated"]]
        a = a[a["group"].isin(b["group"])]
        x = np.arange(len(b))
        ax.bar(x - 0.2, b["recall"] * 100, width=0.4, color=ev.GREY, label="Before mitigation")
        ax.bar(x + 0.2, a["recall"] * 100, width=0.4, color=ev.NAVY, label="After mitigation")
        ax.set_xticks(x, b["group"], rotation=45 if att == "imd_band" else 0, fontsize=8)
        ax.set_ylim(0, 70)
        ax.set_title(titles[att], loc="left", fontsize=10, color=ev.INK, fontweight="bold")
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Recall at 25% capacity (%)")
    axes[-1].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("Equal opportunity: share of non-completers flagged, by group (test set)",
                 x=0.01, ha="left", fontsize=11, color=ev.INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main(base: Path | None = None) -> dict:
    root = Path(base) if base else ROOT
    params = load_params()
    processed = root / "data" / "processed"
    reports = root / "reports"
    figs = reports / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    artifacts = reports / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    capacity = float(params.get("model", {}).get("capacity", 0.25))

    df = data.load(processed)
    split = data.grouped_holdout(df, seed=params["seed"])
    data_hash = train.file_sha256(processed / "model_ready.parquet")
    uri = train.setup_mlflow(root)
    print(f"MLflow tracking: {uri}")

    # ---- baselines and candidates
    baselines = train.run_baselines(split, data_hash)
    cands = train.tune(split, data_hash, artifacts)
    best_name = max(cands, key=lambda k: cands[k]["cv"]["cv_ap_mean"])
    best = cands[best_name]
    model, thr = best["model"], best["threshold"]
    p_test = best["scores"]
    print(f"selected {best_name}: CV AP {best['cv']['cv_ap_mean']:.3f}, test AUC {best['test']['roc_auc']:.3f}")

    # ---- figures: comparison, confusion, calibration, capacity
    pretty = {"logistic_regression": "Logistic regression", "random_forest": "Random forest", "xgboost": "XGBoost"}
    curves = {pretty[n]: (split.y_test, c["scores"]) for n, c in cands.items()}
    curves["Baseline: registration only"] = (split.y_test, baselines["baseline_registration_lr"]["scores"])
    ev.plot_roc_pr(curves, figs / "roc_pr.png")
    ev.plot_confusion(best["test"], f"{pretty[best_name]}: riskiest 25% flagged", figs / "confusion.png")
    ev.plot_calibration(split.y_test, p_test, figs / "calibration.png")
    ev.plot_capacity(split.y_test, p_test, figs / "capacity.png")

    # ---- explainability
    sv, Xt = explain.shap_values(model, split.X_test)
    imp = explain.global_importance(sv)
    explain.plot_global(imp, figs / "shap_global.png")
    explain.plot_beeswarm(sv, figs / "shap_beeswarm.png")
    near = np.flatnonzero((np.abs(p_test - thr) < 0.03) & (split.X_test["n_due_by_30"] > 0).to_numpy())
    case_a, cf_a, cf_tries = explain.pick_learner_a(model, split.X_train, split.y_train, split.X_test,
                                                    split.y_test, p_test, thr)
    case_b = int(near[0])
    explain.plot_waterfall(sv, case_a, f"Learner A: risk {p_test[case_a]:.2f}, flagged", figs / "shap_local_a.png")
    explain.plot_waterfall(sv, case_b, f"Learner B: risk {p_test[case_b]:.2f}, near the threshold",
                           figs / "shap_local_b.png")
    lime_a = explain.lime_explanation(model, split.X_train, split.X_test.iloc[[case_a]])
    lime_b = explain.lime_explanation(model, split.X_train, split.X_test.iloc[[case_b]])
    if not cf_a.empty:
        explain.plot_counterfactual(split.X_test.iloc[case_a], cf_a, float(p_test[case_a]),
                                    figs / "counterfactual_a.png")
    explanations = {
        "global_importance": imp.head(15).to_dict(orient="records"),
        "learner_a": {"risk": float(p_test[case_a]), "label": int(split.y_test[case_a]),
                      "features": split.X_test.iloc[case_a].to_dict(),
                      "shap_top": explain.local_reasons(sv, case_a), "lime": lime_a,
                      "counterfactuals": cf_a.to_dict(orient="records"), "candidates_tried": cf_tries},
        "learner_b": {"risk": float(p_test[case_b]), "label": int(split.y_test[case_b]),
                      "features": split.X_test.iloc[case_b].to_dict(),
                      "shap_top": explain.local_reasons(sv, case_b), "lime": lime_b},
    }

    # ---- fairness before and after mitigation
    pred = (p_test >= thr).astype(int)
    before = {a: fm.attribute_summary(split.y_test, pred, split.attrs_test[a]) for a in data.PROTECTED}
    g_tr = split.attrs_train["gender"].to_numpy()
    g_te = split.attrs_test["gender"].to_numpy()
    eo = fm.fit_capacity_equal_opportunity(best["oof_scores"], split.y_train, g_tr, capacity)
    pred_m = fm.apply_thresholds(p_test, g_te, eo["thresholds"], thr)
    after = {a: fm.attribute_summary(split.y_test, pred_m, split.attrs_test[a]) for a in data.PROTECTED}
    plot_fairness(before, after, figs / "fairness_recall.png")
    pred_to = fm.threshold_optimizer_comparison(best["oof_scores"], split.y_train, g_tr, p_test, g_te)
    eg_scores, pred_eg = fm.exponentiated_gradient_comparison(preprocessor(), split.X_train, split.y_train,
                                                              g_tr, split.X_test, capacity)
    from sklearn.metrics import roc_auc_score

    def _sex_gap(pr):
        return fm.attribute_summary(split.y_test, pr, split.attrs_test["gender"])["recall_gap"]

    comparison = {
        "unmitigated": {**fm.overall(split.y_test, pred), "sex_recall_gap": _sex_gap(pred),
                        "auc": float(roc_auc_score(split.y_test, p_test)), "uses_sex_at_decision": False},
        "capacity_equal_opportunity (selected)": {**fm.overall(split.y_test, pred_m), "sex_recall_gap": _sex_gap(pred_m),
                                                  "auc": float(roc_auc_score(split.y_test, p_test)),
                                                  "uses_sex_at_decision": True},
        "fairlearn_threshold_optimizer": {**fm.overall(split.y_test, pred_to), "sex_recall_gap": _sex_gap(pred_to),
                                          "auc": None, "uses_sex_at_decision": True},
        "fairlearn_exponentiated_gradient": {**fm.overall(split.y_test, pred_eg), "sex_recall_gap": _sex_gap(pred_eg),
                                             "auc": float(roc_auc_score(split.y_test, eg_scores)),
                                             "uses_sex_at_decision": False},
    }
    gate_before, gate_after = fm.gate(before), fm.gate(after)
    released = all(v["pass"] for v in gate_after.values())
    fairness = {
        "criterion": "equal opportunity (recall parity) at 25% capacity",
        "tolerance": fm.MAX_RECALL_GAP, "min_positives_gated": fm.MIN_POSITIVES,
        "before": before, "after": after, "gate_before": gate_before, "gate_after": gate_after,
        "mitigation": {"method": "capacity-constrained equal opportunity thresholds by sex",
                       "thresholds": eo["thresholds"], "fitted_on": "out-of-fold training scores"},
        "comparison": comparison,
        # read by tests/test_fairness.py
        "recall_gap": {a: after[a]["recall_gap"] for a in data.PROTECTED if "power_check" not in after[a]},
        "heterogeneity_p": {a: after[a]["heterogeneity_p"] for a in data.PROTECTED if "power_check" in after[a]},
        "all_gates_pass": released,
    }

    # ---- sensitivity
    split_time = data.time_aware(df)
    sens = {
        "input_noise": sensitivity.input_noise(model, split.X_test, split.y_test, thr),
        "no_assessment_evidence": sensitivity.no_assessment_evidence(model, split.X_test, split.y_test, thr),
        "time_aware": sensitivity.time_aware(model, split_time),
        "seed_stability": sensitivity.seed_stability(model, split),
    }

    # ---- persist, register, document
    meta = {"algorithm": best_name, "params": best["params"], "data_sha256": data_hash,
            "group_thresholds": eo["thresholds"], "capacity": capacity,
            "mlflow_run_id": best["run_id"]}
    train.save(model, thr, meta, root / "models")
    reg = train.register(best_name, best["run_id"], stage="Staging")
    meta["registry"] = reg
    (root / "models" / "model_metadata.json").write_text(
        json.dumps(_jsonable({**json.loads((root / "models" / "model_metadata.json").read_text()), "registry": reg}), indent=2))

    metrics = {
        "split": {"name": split.name, "n_train": int(len(split.y_train)), "n_test": int(len(split.y_test)),
                  "test_base_rate": float(split.y_test.mean())},
        "baselines": {k: v["test"] for k, v in baselines.items()},
        "candidates": {k: {"cv": v["cv"], "test": v["test"], "params": v["params"]} for k, v in cands.items()},
        "selected": best_name, "threshold": thr, "sensitivity": sens, "registry": reg,
    }
    for name, obj in [("metrics.json", metrics), ("fairness.json", fairness), ("explanations.json", explanations)]:
        (reports / name).write_text(json.dumps(_jsonable(obj), indent=2, default=str), encoding="utf-8")

    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_artifacts(str(figs), "figures")
        for name in ["fairness.json", "explanations.json", "metrics.json"]:
            mlflow.log_artifact(str(reports / name), "reports")
        mlflow.log_metrics({"sex_recall_gap_before": before["gender"]["recall_gap"],
                            "sex_recall_gap_after": after["gender"]["recall_gap"],
                            "time_aware_auc": sens["time_aware"]["roc_auc"]})
        mlflow.set_tag("fairness_gates_pass", str(released))

    write_model_card(root / "docs" / "model_card.md", metrics, fairness, explanations)
    print(f"done: fairness gates pass = {released}; registered {reg}")
    return metrics


if __name__ == "__main__":
    main()
