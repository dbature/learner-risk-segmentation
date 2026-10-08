"""Coach caseload: ranking under capacity, and rows the API would reject are reported."""
import json
import math
import pathlib

import pytest

pytest.importorskip("xgboost")
pytest.importorskip("fastapi")
SAMPLE = pathlib.Path("dashboard/sample_cohort.csv")
MODEL = pathlib.Path("models/model.joblib")
pytestmark = pytest.mark.skipif(not (SAMPLE.exists() and MODEL.exists()), reason="model or demo cohort not present")


@pytest.fixture(scope="module")
def parts():
    import joblib
    import pandas as pd

    from src.segments.trajectory import TrajectorySegmenter

    seg = TrajectorySegmenter.from_dict(json.loads(pathlib.Path("dashboard/data/segments.json").read_text())["model"])
    return pd.read_csv(SAMPLE), joblib.load(MODEL), seg


@pytest.mark.parametrize("capacity", [0.1, 0.25, 0.5])
def test_caseload_is_the_riskiest_share(parts, capacity):
    from src.dashboard.caseload import rank

    cohort, model, seg = parts
    out, errors = rank(cohort, model, seg, capacity)
    assert not errors and len(out) == len(cohort)
    assert out["risk"].is_monotonic_decreasing
    assert out["in_caseload"].sum() == math.ceil(capacity * len(out))
    assert out.loc[out["in_caseload"], "risk"].min() >= out.loc[~out["in_caseload"], "risk"].max()


def test_rows_the_api_would_reject_are_reported_not_scored(parts):
    from src.dashboard.caseload import rank

    cohort, model, seg = parts
    bad = cohort.copy()
    bad.loc[0, "n_submitted_by_30"] = bad.loc[0, "n_due_by_30"] + 1
    bad.loc[1, "clicks_0_29"] = bad.loc[1, "clicks_0_29"] + 7
    bad.loc[2, "code_module"] = "XYZ"
    out, errors = rank(bad, model, seg, 0.25)
    assert len(out) == len(cohort) - 3
    assert [e["row"] for e in errors] == [2, 3, 4]


def test_demo_cohort_is_synthetic():
    import pandas as pd

    refs = pd.read_csv(SAMPLE)["learner_ref"]
    assert refs.str.startswith("SYN-").all()
