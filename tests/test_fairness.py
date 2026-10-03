"""Fairness gate. A release is blocked if recall parity exceeds the tolerance."""
import json
import pathlib

MAX_GAP = 0.05


def test_recall_parity_within_tolerance():
    report = json.loads(pathlib.Path("reports/fairness.json").read_text())
    for attribute, gap in report["recall_gap"].items():
        assert gap <= MAX_GAP, f"recall gap on {attribute} is {gap:.3f}, tolerance {MAX_GAP}"
