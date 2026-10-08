"""Pre-registered target choice (Module 4).

Rule, fixed before any model was tuned: switch the target from withdrawal
after day 30 to non-completion (withdraw or fail) if it beats withdrawal by
0.03 ROC AUC or more in grouped five-fold cross-validation, with the same
features and the same untuned model for both.

    python -m src.models.target_check
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.models.data import FEATURES
from src.models.preprocess import preprocessor

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ["label_withdrew_after_30", "label_non_completion"]
MARGIN = 0.03


def main(base: Path | None = None) -> dict:
    import pandas as pd

    root = Path(base) if base else ROOT
    df = pd.read_parquet(root / "data" / "processed" / "model_ready.parquet")
    X, groups = df[FEATURES], df["learner_key"]
    out = {}
    for t in TARGETS:
        pipe = Pipeline([("prep", preprocessor()),
                         ("clf", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                               subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                                               tree_method="hist", random_state=42))])
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        auc = cross_val_score(pipe, X, df[t], groups=groups, cv=cv, scoring="roc_auc")
        out[t] = {"auc_mean": float(np.mean(auc)), "auc_std": float(np.std(auc)), "base_rate": float(df[t].mean())}
    diff = out["label_non_completion"]["auc_mean"] - out["label_withdrew_after_30"]["auc_mean"]
    out["difference"] = diff
    out["decision"] = "label_non_completion" if diff >= MARGIN else "label_withdrew_after_30"
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports" / "target_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
