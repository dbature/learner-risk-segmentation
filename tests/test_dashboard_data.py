"""The public dashboard files: aggregates only, and consistent with the Module 4 reports."""
import json
import pathlib

import numpy as np
import pytest

DATA = pathlib.Path("dashboard/data")
REPORTS = pathlib.Path("reports")
pytestmark = pytest.mark.skipif(not (DATA / "fairness.json").exists(), reason="dashboard data not built yet")


def load(name, folder=DATA):
    return json.loads((folder / f"{name}.json").read_text())


def test_no_identifiers_or_protected_attributes_in_examples():
    ex = load("examples")
    assert set(ex) == {"A", "B", "C"}
    banned = {"learner_key", "id_student", "gender", "age_band", "disability", "imd_band", "region"}
    for v in ex.values():
        assert not banned & set(v["features"])


def test_histograms_cover_the_whole_test_set_once_per_attribute():
    f, n = load("fairness"), load("summary")["test"]["learners"]
    for att, groups in f["histograms"].items():
        assert sum(sum(v["0"]) + sum(v["1"]) for v in groups.values()) == n, att


def test_histograms_reproduce_the_reported_recall_gap():
    f, s = load("fairness"), load("summary")["test"]
    report = load("fairness", REPORTS)
    for att, groups in f["histograms"].items():
        total = sum(np.array(v["0"]) + np.array(v["1"]) for v in groups.values())
        above = np.cumsum(total[::-1])[::-1]
        i = int(np.argmin(np.abs(above - (s["tp"] + s["fp"]))))
        rec = {g: sum(v["1"][i:]) / sum(v["1"]) for g, v in groups.items()
               if sum(v["1"]) >= report["min_positives_gated"]}
        assert abs((max(rec.values()) - min(rec.values())) - report["before"][att]["recall_gap"]) < 0.005, att


def test_summary_matches_the_model_report():
    s, m = load("summary")["test"], load("metrics", REPORTS)
    t = m["candidates"][m["selected"]]["test"]
    assert s["recall"] == pytest.approx(t["recall"], abs=1e-6)
    assert s["precision"] == pytest.approx(t["precision"], abs=1e-6)


def test_segment_summary_is_complete():
    seg = load("segments")
    assert [r["segment"] for r in seg["summary"]] == ["Steady", "Slipping", "Late starter", "Disengaged"]
    assert sum(r["learners"] for r in seg["summary"]) == load("summary")["test"]["learners"]
