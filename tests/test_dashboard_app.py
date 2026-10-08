"""Smoke test: every dashboard view renders without an exception, and the
what-if view reproduces the Module 4 counterfactual for Learner A."""
import pathlib

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
APP = "dashboard/app.py"
VIEWS = ["Overview", "What drives risk", "Example learners and what-if", "Learner segments",
         "Ethical compliance", "About this model"]
pytestmark = pytest.mark.skipif(not pathlib.Path("dashboard/data/summary.json").exists(),
                                reason="dashboard data not built yet")


def open_view(view):
    at = st_testing.AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio[0].set_value(view).run()
    return at


@pytest.mark.parametrize("view", VIEWS)
def test_view_renders(view):
    at = open_view(view)
    assert not at.exception, [e.value for e in at.exception]


def test_what_if_reaches_the_counterfactual():
    at = open_view("Example learners and what-if")
    labels = [s.label for s in at.slider]
    at.slider[labels.index("Online clicks, week 4")].set_value(175).run()
    labels = [s.label for s in at.slider]
    at.slider[labels.index("Days active online (of 30)")].set_value(24).run()
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Risk score"] == "57%"
    assert metrics["Flagged at 25% capacity"] == "No"


@pytest.mark.parametrize("group", ["Sex", "Age band", "Disability", "Deprivation (IMD band)"])
def test_fairness_view_matches_report_at_25_percent(group):
    import json

    at = open_view("Ethical compliance")
    at.selectbox[0].set_value(group).run()
    key = {"Sex": "gender", "Age band": "age_band", "Disability": "disability",
           "Deprivation (IMD band)": "imd_band"}[group]
    expected = json.loads(pathlib.Path("reports/fairness.json").read_text())["before"][key]["recall_gap"]
    shown = {m.label: m.value for m in at.metric}["Recall gap (largest minus smallest)"]
    assert shown == f"{100 * expected:.1f} pts"
