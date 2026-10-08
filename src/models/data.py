"""Modelling dataset: features, target, protected attributes and splits.

Reads the two Module 3 outputs. model_ready carries the features and labels;
audit_attributes carries the protected attributes and is joined only for
fairness measurement, never into the feature matrix.

Target (Module 4 decision): label_non_completion, final result Withdrawn or
Fail, among learners still enrolled on day 30. Chosen over withdrawal-only by
the rule fixed before Module 3: it beat withdrawal by more than 0.03 AUC
(0.769 against 0.692 in grouped 5-fold cross-validation).

code_module moves from context to feature here. Module difficulty is known at
registration and lifts AUC from 0.564 to 0.595 on registration data alone.
code_presentation stays out: it is the calendar, and the time-aware check
holds the latest presentation back.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.features.build_features import FEATURE_COLUMNS, FORBIDDEN_FEATURES

TARGET = "label_non_completion"
GROUP = "learner_key"
CATEGORICAL = ["code_module"]
NUMERIC = list(FEATURE_COLUMNS)
FEATURES = CATEGORICAL + NUMERIC
PROTECTED = ["gender", "age_band", "disability", "imd_band"]
KEY = ["learner_key", "code_module", "code_presentation"]

assert not FORBIDDEN_FEATURES & set(FEATURES), "leaky column in the model feature list"


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    groups_train: np.ndarray
    attrs_test: pd.DataFrame
    attrs_train: pd.DataFrame
    name: str


def load(processed: Path) -> pd.DataFrame:
    processed = Path(processed)
    m = pd.read_parquet(processed / "model_ready.parquet")
    a = pd.read_parquet(processed / "audit_attributes.parquet")
    df = m.merge(a, on=KEY, how="left", validate="1:1")
    if len(df) != len(m):
        raise ValueError("audit join changed the row count")
    for c in NUMERIC:
        df[c] = df[c].astype("float64")
    for c in PROTECTED:
        df[c] = df[c].astype("string").fillna("Missing")
    return df


def _make(df: pd.DataFrame, tr: np.ndarray, te: np.ndarray, name: str) -> Split:
    return Split(
        X_train=df.iloc[tr][FEATURES].reset_index(drop=True),
        X_test=df.iloc[te][FEATURES].reset_index(drop=True),
        y_train=df.iloc[tr][TARGET].to_numpy(),
        y_test=df.iloc[te][TARGET].to_numpy(),
        groups_train=df.iloc[tr][GROUP].to_numpy(),
        attrs_test=df.iloc[te][PROTECTED + KEY].reset_index(drop=True),
        attrs_train=df.iloc[tr][PROTECTED + KEY].reset_index(drop=True),
        name=name,
    )


def grouped_holdout(df: pd.DataFrame, seed: int = 42) -> Split:
    """80/20 hold-out, stratified on the target, with no learner on both sides."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tr, te = next(cv.split(df, df[TARGET], groups=df[GROUP]))
    split = _make(df, tr, te, "grouped_holdout_80_20")
    assert not set(df.iloc[tr][GROUP]) & set(df.iloc[te][GROUP])
    return split


def time_aware(df: pd.DataFrame, test_presentation: str = "2014J") -> Split:
    """Train on earlier presentations, test on the latest, dropping any test
    learner who also appears in training."""
    is_test = df["code_presentation"].eq(test_presentation).to_numpy()
    train_learners = set(df.loc[~is_test, GROUP])
    clean_test = is_test & ~df[GROUP].isin(train_learners).to_numpy()
    tr = np.flatnonzero(~is_test)
    te = np.flatnonzero(clean_test)
    return _make(df, tr, te, f"time_aware_test_{test_presentation}")
