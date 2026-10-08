"""FastAPI /predict contract. Uses a small logistic model trained on synthetic
rows with the real feature schema, so it runs in CI without the dataset."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.models.data import FEATURES, NUMERIC  # noqa: E402

EXAMPLE = {
    "code_module": "DDD", "num_of_prev_attempts": 0, "studied_credits": 60, "date_registration": -96,
    "n_due_by_30": 1, "n_submitted_by_30": 0, "n_banked_by_30": 0, "mean_score_by_30": None,
    "clicks_pre_start": 185, "clicks_0_29": 327, "clicks_wk1": 259, "clicks_wk2": 68,
    "clicks_wk3": 0, "clicks_wk4": 0, "active_days_0_29": 9, "distinct_sites_0_29": 30,
    "first_active_day": 0,
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    from src.models.preprocess import preprocessor

    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({c: rng.integers(0, 50, n).astype(float) for c in NUMERIC})
    X["code_module"] = rng.choice(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"], n)
    X.loc[::7, "mean_score_by_30"] = np.nan
    y = (X["active_days_0_29"] < 20).astype(int).to_numpy()
    model = Pipeline([("prep", preprocessor()), ("clf", LogisticRegression(max_iter=500))]).fit(X[FEATURES], y)
    joblib.dump(model, tmp_path / "model.joblib")
    (tmp_path / "model_metadata.json").write_text(json.dumps({
        "algorithm": "logistic_regression", "threshold": 0.5,
        "group_thresholds": {"F": 0.3, "M": 0.7}, "features": FEATURES, "registry": {"version": 1}}))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    from src.serving import api

    api.load.cache_clear()
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_score_flag_and_rule(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["risk_score"] <= 1
    assert body["flagged"] == (body["risk_score"] >= body["threshold"])
    assert body["decision_rule"] == "single capacity threshold"


def test_sex_selects_the_mitigated_threshold(client):
    f = client.post("/predict", json={**EXAMPLE, "sex": "F"}).json()
    m = client.post("/predict", json={**EXAMPLE, "sex": "M"}).json()
    assert f["risk_score"] == m["risk_score"]  # sex is never a model input
    assert (f["threshold"], m["threshold"]) == (0.3, 0.7)


@pytest.mark.parametrize("bad", [
    {"active_days_0_29": 31},
    {"code_module": "ZZZ"},
    {"n_submitted_by_30": 2},
    {"clicks_wk1": 1},
])
def test_invalid_input_is_rejected(client, bad):
    assert client.post("/predict", json={**EXAMPLE, **bad}).status_code == 422
