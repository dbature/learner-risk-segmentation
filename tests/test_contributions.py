"""Dashboard reasons equal SHAP values for the registered model."""
import pathlib

import numpy as np
import pytest

xgb = pytest.importorskip("xgboost")
shap = pytest.importorskip("shap")
MODEL = pathlib.Path("models/model.joblib")
EXAMPLES = pathlib.Path("dashboard/data/examples.json")


@pytest.mark.skipif(not (MODEL.exists() and EXAMPLES.exists()), reason="model or dashboard data not present")
def test_contributions_match_shap():
    import joblib
    import pandas as pd

    from src.dashboard.contrib import _base, contributions
    from src.models import explain
    from src.models.data import FEATURES

    model = joblib.load(MODEL)
    ex = __import__("json").loads(pathlib.Path("dashboard/data/examples.json").read_text())
    X = pd.DataFrame([{k: (np.nan if v["features"][k] is None else v["features"][k]) for k in FEATURES}
                      for v in ex.values()])
    num = [c for c in FEATURES if c != "code_module"]
    X[num] = X[num].astype(float)
    g, _ = contributions(model, X)
    sv, _ = explain.shap_values(model, X)
    ref = pd.DataFrame(sv.values, columns=sv.feature_names).T.groupby(_base).sum().T
    assert np.allclose(ref[g.columns].to_numpy(), g.to_numpy(), atol=1e-5)
