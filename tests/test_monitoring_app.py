"""Monitoring dashboard: every view renders, and its running total matches the Module 4 gate."""
import json
import pathlib

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
DATA = pathlib.Path("monitoring/data/monitoring.json")
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="monitoring data not built yet")
VIEWS = ["Status", "Performance", "Drift", "Fairness", "Response plan"]


@pytest.mark.parametrize("view", VIEWS)
def test_view_renders(view):
    at = st_testing.AppTest.from_file("monitoring/app.py", default_timeout=120).run()
    at.sidebar.radio[0].set_value(view).run()
    assert not at.exception, [e.value for e in at.exception]


def test_running_total_equals_the_release_gate_figures():
    m = json.loads(DATA.read_text())
    gate = json.loads(pathlib.Path("reports/fairness.json").read_text())["after"]
    last = m["cumulative_fairness"][-1]
    for att in ["gender", "age_band", "disability"]:
        assert last[att]["recall_gap"] == pytest.approx(gate[att]["recall_gap"], abs=1e-9)
    assert last["imd_band"]["chi2_p"] == pytest.approx(gate["imd_band"]["heterogeneity_p"], abs=1e-9)
