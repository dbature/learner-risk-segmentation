"""Model development: baselines, three tuned algorithms, MLflow tracking.

Protocol
--------
* 80/20 hold-out grouped by learner and stratified on the target. The test
  set is touched once, after tuning.
* Tuning: randomised search, 5-fold StratifiedGroupKFold on the training set
  only, scored on average precision (refit) with ROC AUC also recorded.
* Baselines anchor every claim: the majority class, random flagging, and a
  logistic regression using only what is known at registration.
* Candidates: logistic regression (transparent), random forest (non-linear,
  robust), XGBoost (gradient boosting, Chen and Guestrin 2016).
* Every run is logged to MLflow (Zaharia et al., 2018): parameters, CV and
  test metrics, the cross-validation table, figures and the fitted pipeline.
  The selected model is registered in the MLflow Model Registry.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from scipy.stats import loguniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.models.data import FEATURES, TARGET, Split
from src.models.preprocess import preprocessor
from src.models.evaluate import capacity_threshold, classification_metrics

EXPERIMENT = "learner-risk-module4"
REGISTERED_NAME = "learner-risk-noncompletion"
SEED = 42

REGISTRATION_ONLY = ["code_module", "num_of_prev_attempts", "studied_credits",
                     "date_registration", "date_registration_missing"]
# The Module 1 headline signal: did the learner submit the first assessment?
FIRST_ASSESSMENT_ONLY = ["n_due_by_30", "n_submitted_by_30", "submit_rate_by_30"]


def candidates() -> dict[str, tuple[Pipeline, dict, int]]:
    lr = Pipeline([("prep", preprocessor()),
                   ("clf", LogisticRegression(max_iter=2000, random_state=SEED))])
    rf = Pipeline([("prep", preprocessor()),
                   ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1))])
    xgb = Pipeline([("prep", preprocessor()),
                    ("clf", XGBClassifier(random_state=SEED, n_jobs=-1, tree_method="hist",
                                          eval_metric="logloss"))])
    return {
        "logistic_regression": (lr, {
            "clf__C": loguniform(1e-3, 10),
            "clf__class_weight": [None, "balanced"],
        }, 12),
        "random_forest": (rf, {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [6, 10, 14, None],
            "clf__min_samples_leaf": [5, 20, 50],
            "clf__max_features": ["sqrt", 0.5],
            "clf__class_weight": [None, "balanced_subsample"],
        }, 12),
        "xgboost": (xgb, {
            "clf__n_estimators": [200, 400, 700],
            "clf__max_depth": [3, 4, 5, 6],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__subsample": [0.7, 0.85, 1.0],
            "clf__colsample_bytree": [0.6, 0.8, 1.0],
            "clf__min_child_weight": [1, 5, 10],
            "clf__reg_lambda": [1, 5],
        }, 20),
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def setup_mlflow(root: Path) -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(root / 'mlruns' / 'mlflow.db').as_posix()}")
    (root / "mlruns").mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(root / "mlruns" / "artifacts").resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)
    return uri


def _common_tags(split: Split, data_hash: str) -> dict:
    return {"split": split.name, "target": TARGET, "data_sha256": data_hash,
            "n_train": str(len(split.y_train)), "n_test": str(len(split.y_test)),
            "git_commit": os.environ.get("GIT_COMMIT", "unknown")}


def run_baselines(split: Split, data_hash: str) -> dict[str, dict]:
    out = {}
    specs = {
        "baseline_majority": Pipeline([("prep", preprocessor()), ("clf", DummyClassifier(strategy="most_frequent"))]),
        "baseline_random": Pipeline([("prep", preprocessor()), ("clf", DummyClassifier(strategy="uniform", random_state=SEED))]),
        "baseline_registration_lr": Pipeline([
            ("prep", preprocessor([c for c in REGISTRATION_ONLY if c != "code_module"], ["code_module"])),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "baseline_first_assessment_lr": Pipeline([
            ("prep", preprocessor(FIRST_ASSESSMENT_ONLY, [])),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
    }
    for name, pipe in specs.items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({**_common_tags(split, data_hash), "role": "baseline"})
            cols = {"baseline_registration_lr": REGISTRATION_ONLY,
                    "baseline_first_assessment_lr": FIRST_ASSESSMENT_ONLY}.get(name, FEATURES)
            pipe.fit(split.X_train[cols], split.y_train)
            p = pipe.predict_proba(split.X_test[cols])[:, 1]
            if name == "baseline_random":
                p = np.random.default_rng(SEED).random(len(p))
            elif name != "baseline_majority":
                # coarse baselines produce many tied scores; break ties at random so
                # that exactly the capacity share is flagged, as for the real models
                p = p + np.random.default_rng(SEED).random(len(p)) * 1e-6
            m = classification_metrics(split.y_test, p, capacity_threshold(p))
            if name == "baseline_majority":
                m.update(recall=0.0, precision=0.0, f1=0.0, flag_rate=0.0,
                         accuracy=float(1 - split.y_test.mean()))
            mlflow.log_metrics({f"test_{k}": v for k, v in m.items() if isinstance(v, float) and np.isfinite(v)})
            out[name] = {"test": m, "scores": p}
    return out


def tune(split: Split, data_hash: str, artifacts: Path) -> dict[str, dict]:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}
    for name, (pipe, space, n_iter) in candidates().items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({**_common_tags(split, data_hash), "role": "candidate",
                             "search": f"RandomizedSearchCV n_iter={n_iter}, 5-fold StratifiedGroupKFold"})
            search = RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=cv,
                                        scoring={"ap": "average_precision", "auc": "roc_auc"},
                                        refit="ap", random_state=SEED, n_jobs=1)
            search.fit(split.X_train, split.y_train, groups=split.groups_train)
            best = search.best_estimator_
            i = search.best_index_
            cvr = search.cv_results_
            cv_metrics = {
                "cv_ap_mean": float(cvr["mean_test_ap"][i]), "cv_ap_std": float(cvr["std_test_ap"][i]),
                "cv_auc_mean": float(cvr["mean_test_auc"][i]), "cv_auc_std": float(cvr["std_test_auc"][i]),
            }
            # Operating threshold from out-of-fold training predictions, so
            # the test set plays no part in choosing it.
            oof = cross_val_predict(clone(best), split.X_train, split.y_train, groups=split.groups_train,
                                    cv=cv, method="predict_proba")[:, 1]
            thr = capacity_threshold(oof)
            p = best.predict_proba(split.X_test)[:, 1]
            m = classification_metrics(split.y_test, p, thr)
            mlflow.log_params({k.replace("clf__", ""): v for k, v in search.best_params_.items()})
            mlflow.log_metrics(cv_metrics)
            mlflow.log_metrics({f"test_{k}": v for k, v in m.items() if isinstance(v, float)})
            table = pd.DataFrame(cvr).drop(columns=["params"])
            path = artifacts / f"cv_results_{name}.csv"
            table.to_csv(path, index=False)
            mlflow.log_artifact(str(path), "cv")
            sig = infer_signature(split.X_test.head(50), best.predict_proba(split.X_test.head(50)))
            mlflow.sklearn.log_model(best, "model", signature=sig, input_example=split.X_test.head(3))
            results[name] = {"run_id": run.info.run_id, "cv": cv_metrics, "test": m,
                             "params": search.best_params_, "model": best, "scores": p,
                             "oof_scores": oof, "threshold": thr}
    return results


def register(best_name: str, run_id: str, stage: str = "Staging") -> dict:
    mv = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_NAME)
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(REGISTERED_NAME, "champion", mv.version)
    try:
        client.transition_model_version_stage(REGISTERED_NAME, mv.version, stage)
    except Exception:  # stages are deprecated in newer MLflow; the alias carries the meaning
        pass
    client.set_model_version_tag(REGISTERED_NAME, mv.version, "algorithm", best_name)
    return {"name": REGISTERED_NAME, "version": int(mv.version), "alias": "champion", "stage": stage}


class ReleaseBlocked(RuntimeError):
    """Raised when a model that fails the fairness gate is put forward for Production."""


def promote(version: int, fairness: dict) -> dict:
    """The release gate. Production needs every fairness gate to pass; a
    model that fails stays in Staging until the gate passes or the Ethics
    Committee formally changes the rule. Checked before MLflow is touched."""
    failed = [a for a, v in fairness["gate_after"].items() if not v["pass"]]
    if failed or not fairness.get("all_gates_pass", False):
        raise ReleaseBlocked(f"fairness gate fails on {', '.join(failed) or 'unknown'}; model stays in Staging")
    client = mlflow.MlflowClient()
    client.transition_model_version_stage(REGISTERED_NAME, version, "Production")
    return {"name": REGISTERED_NAME, "version": int(version), "stage": "Production"}


def save(model, threshold: float, meta: dict, models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / "model.joblib"
    joblib.dump(model, path)
    meta = {**meta, "threshold": threshold, "features": FEATURES, "target": TARGET,
            "model_sha256": file_sha256(path)}
    (models_dir / "model_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return path
