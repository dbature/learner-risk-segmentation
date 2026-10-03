"""Fairness gate. A release is blocked if recall parity exceeds tolerance.

Skips until the model pipeline writes a report, which happens in Module 4.
The tolerance is fixed here now so it cannot be quietly relaxed later.
"""
import json
import pathlib

import pytest

MAX_RECALL_GAP = 0.05
REPORT = pathlib.Path("reports/fairness.json")


@pytest.mark.skipif(not REPORT.exists(), reason="fairness report not produced until Module 4")
def test_recall_parity_within_tolerance():
    report = json.loads(REPORT.read_text())
    for attribute, gap in report["recall_gap"].items():
        assert gap <= MAX_RECALL_GAP, (
            f"recall gap on {attribute} is {gap:.3f}, tolerance is {MAX_RECALL_GAP}"
        )


def test_tolerance_is_not_relaxed():
    """Guards the commitment itself. Module 1 set 5 percentage points."""
    assert MAX_RECALL_GAP == 0.05
