"""Fairness release gate.

Equal opportunity: recall at the 25% capacity cut-off must not differ by more
than 5 percentage points between groups (Module 1). For IMD band (10 groups)
the max-minus-min gap of noisy estimates cannot test that rule, so the gate
for many-level attributes is a chi-square test of equal recall (p >= 0.05),
pending Ethics Committee ratification. Both are reported in full.

The gate controls release, not the build. A model that fails it can be
trained, tracked and reviewed in Staging, but cannot be promoted to
Production. These tests check three things: the tolerance has not been
relaxed, the verdict in the report really follows from the measured numbers
(so a failing model cannot be reported as passing), and a failing model is
neither registered in nor promotable to Production.
"""
import json
import pathlib

import pytest

MAX_RECALL_GAP = 0.05
MIN_P_VALUE = 0.05
FAIRNESS = pathlib.Path("reports/fairness.json")
METRICS = pathlib.Path("reports/metrics.json")


def test_tolerance_is_not_relaxed():
    """Guards the commitment itself. Module 1 set 5 percentage points."""
    from src.fairness import metrics as fm

    assert MAX_RECALL_GAP == 0.05
    assert fm.MAX_RECALL_GAP == MAX_RECALL_GAP


@pytest.mark.skipif(not FAIRNESS.exists(), reason="fairness report not produced yet")
def test_gate_verdict_follows_from_the_measurements():
    report = json.loads(FAIRNESS.read_text())
    for attribute, verdict in report["gate_after"].items():
        s = report["after"][attribute]
        if verdict["rule"].startswith("chi-square"):
            expected = s["heterogeneity_p"] >= MIN_P_VALUE
        else:
            expected = s["recall_gap"] <= MAX_RECALL_GAP
        assert verdict["pass"] == expected, f"gate verdict for {attribute} does not match its numbers"
    assert report["all_gates_pass"] == all(v["pass"] for v in report["gate_after"].values())


@pytest.mark.skipif(not (FAIRNESS.exists() and METRICS.exists()), reason="reports not produced yet")
def test_model_failing_the_gate_is_not_in_production():
    report = json.loads(FAIRNESS.read_text())
    stage = json.loads(METRICS.read_text())["registry"]["stage"]
    if not report["all_gates_pass"]:
        assert stage != "Production", "a model that fails the fairness gate is registered in Production"


def test_promotion_refused_when_any_gate_fails():
    pytest.importorskip("mlflow")
    from src.models.train import ReleaseBlocked, promote

    failing = {"all_gates_pass": False,
               "gate_after": {"gender": {"pass": True}, "imd_band": {"pass": False}}}
    with pytest.raises(ReleaseBlocked, match="imd_band"):
        promote(1, failing)
