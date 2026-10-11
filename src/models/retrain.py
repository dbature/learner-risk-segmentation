"""Gated retraining: train a challenger, compare it with the champion, never auto-release.

    python -m src.models.retrain --reason "AUC below floor on 2015B"

Triggered by a monitoring alert (docs/monitoring.md), not by a calendar.
The pipeline first refreshes data/processed with the newest course runs;
this script then:

1. retrains XGBoost with the champion's registered hyperparameters on the
   current processed data (same grouped 80/20 split, same seed), so the
   only thing that changes is the data;
2. sets the capacity threshold from out-of-fold training scores and refits
   the sex-specific mitigation thresholds the same way;
3. scores champion and challenger on the same hold-out and runs the
   fairness gate on the challenger;
4. logs everything to MLflow (experiment learner-risk-retraining) and
   registers the challenger in Staging only if decide() accepts it.

Production is never set here. promote() in src/models/train.py refuses any
model that fails the fairness gate, and release also needs the Ethics
Committee (docs/ethics/governance_structure.md).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "learner-risk-retraining"
MAX_AUC_DROP = 0.01     # challenger may not lose more than 0.01 AUC
MAX_RECALL_DROP = 0.02  # or more than 2 points of recall at capacity


def decide(champion: dict, challenger: dict, gate_pass: bool) -> dict:
    """Pure decision rule, unit tested without data."""
    reasons = []
    if challenger["roc_auc"] < champion["roc_auc"] - MAX_AUC_DROP:
        reasons.append(f"AUC {challenger['roc_auc']:.3f} is more than {MAX_AUC_DROP} below champion {champion['roc_auc']:.3f}")
    if challenger["recall"] < champion["recall"] - MAX_RECALL_DROP:
        reasons.append(f"recall {challenger['recall']:.3f} is more than {MAX_RECALL_DROP:.0%} points below champion")
    accepted = not reasons  # performance rules decide registration as a challenger
    if not gate_pass:       # the fairness gate decides eligibility for release review
        reasons.append("fairness gate fails")
    return {"register_as_challenger": accepted, "eligible_for_release_review": accepted and gate_pass,
            "reasons": reasons or ["matches or beats the champion and passes every gate"]}


def _portable(uri: str, root: Path) -> str:
    """Drop the machine's absolute path from a committed file (it exposes the local user name)."""
    return uri.replace(Path(root).resolve().as_posix() + "/", "").replace(Path(root).as_posix() + "/", "")


def main(argv: list[str] | None = None, base: Path | None = None) -> dict:
    import mlflow
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    from src.fairness import metrics as fm
    from src.models import data, train
    from src.models.evaluate import capacity_threshold, classification_metrics
    from src.models.preprocess import preprocessor

    args = argparse.ArgumentParser()
    args.add_argument("--reason", default="manual retraining review")
    args.add_argument("--capacity", type=float, default=0.25)
    opts = args.parse_args(argv)
    root = Path(base) if base else ROOT

    meta = json.loads((root / "models" / "model_metadata.json").read_text())
    champion_model = joblib.load(root / "models" / "model.joblib")
    df = data.load(root / "data" / "processed")
    split = data.grouped_holdout(df, seed=42)
    data_hash = train.file_sha256(root / "data" / "processed" / "model_ready.parquet")

    params = {k.replace("clf__", ""): v for k, v in meta["params"].items()}
    challenger = Pipeline([("prep", preprocessor()),
                           ("clf", XGBClassifier(**params, n_jobs=-1, tree_method="hist", random_state=42,
                                                 eval_metric="logloss"))])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(clone(challenger), split.X_train, split.y_train, groups=split.groups_train, cv=cv,
                            method="predict_proba")[:, 1]
    thr = capacity_threshold(oof, opts.capacity)
    challenger.fit(split.X_train, split.y_train)

    p_new = challenger.predict_proba(split.X_test)[:, 1]
    p_old = champion_model.predict_proba(split.X_test)[:, 1]
    m_new = classification_metrics(split.y_test, p_new, thr)
    m_old = classification_metrics(split.y_test, p_old, float(meta["threshold"]))

    g_tr = split.attrs_train["gender"].to_numpy()
    eo = fm.fit_capacity_equal_opportunity(oof, split.y_train, g_tr, opts.capacity)
    pred_m = fm.apply_thresholds(p_new, split.attrs_test["gender"].to_numpy(), eo["thresholds"], thr)
    after = {a: fm.attribute_summary(split.y_test, pred_m, split.attrs_test[a]) for a in data.PROTECTED}
    gate = fm.gate(after)
    gate_pass = all(v["pass"] for v in gate.values())
    decision = decide(m_old, m_new, gate_pass)

    uri = train.setup_mlflow(root)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="challenger") as run:
        mlflow.set_tags({"reason": opts.reason, "data_sha256": data_hash, "champion_sha256": meta["model_sha256"],
                         "fairness_gates_pass": str(gate_pass), "decision": json.dumps(decision)})
        mlflow.log_params(params)
        mlflow.log_metrics({f"challenger_{k}": v for k, v in m_new.items() if isinstance(v, (int, float))})
        mlflow.log_metrics({f"champion_{k}": v for k, v in m_old.items() if isinstance(v, (int, float))})
        mlflow.sklearn.log_model(challenger, "model")
        run_id = run.info.run_id
    reg = None
    if decision["register_as_challenger"]:
        mv = mlflow.register_model(f"runs:/{run_id}/model", train.REGISTERED_NAME)
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(train.REGISTERED_NAME, "challenger", mv.version)
        client.set_model_version_tag(train.REGISTERED_NAME, mv.version, "stage_intent", "Staging")
        try:  # same as train.register(): stages are deprecated, so record what the registry actually holds
            client.transition_model_version_stage(train.REGISTERED_NAME, mv.version, "Staging")
        except Exception:
            pass
        stage = client.get_model_version(train.REGISTERED_NAME, mv.version).current_stage
        reg = {"name": train.REGISTERED_NAME, "version": int(mv.version), "alias": "challenger", "stage": stage}

    out = {"run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "reason": opts.reason,
           "mlflow": {"tracking_uri": _portable(uri, root), "experiment": EXPERIMENT, "run_id": run_id},
           "registered": reg,
           "champion": {k: m_old[k] for k in ["roc_auc", "recall", "precision", "flag_rate"]},
           "challenger": {k: m_new[k] for k in ["roc_auc", "recall", "precision", "flag_rate"]},
           "challenger_threshold": thr, "challenger_group_thresholds": eo["thresholds"],
           "challenger_gate": gate, "decision": decision,
           "note": "Production is never set by this script; release needs every gate and the Ethics Committee."}
    path = root / "reports" / "retraining_decision.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8", newline="\n")
    print(json.dumps({"decision": decision, "challenger": out["challenger"], "champion": out["champion"]},
                     indent=2, default=float))
    return out


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
