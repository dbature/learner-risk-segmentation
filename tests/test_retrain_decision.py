"""Retraining decision rule, on synthetic numbers."""
from src.models.retrain import decide

CHAMP = {"roc_auc": 0.760, "recall": 0.448}


def test_equal_challenger_that_passes_the_gate_is_eligible_for_review():
    d = decide(CHAMP, {"roc_auc": 0.762, "recall": 0.450}, gate_pass=True)
    assert d["register_as_challenger"] and d["eligible_for_release_review"]


def test_worse_challenger_is_not_registered():
    d = decide(CHAMP, {"roc_auc": 0.740, "recall": 0.450}, gate_pass=True)
    assert not d["register_as_challenger"] and not d["eligible_for_release_review"]


def test_challenger_failing_fairness_is_registered_but_not_eligible():
    d = decide(CHAMP, {"roc_auc": 0.765, "recall": 0.455}, gate_pass=False)
    assert d["register_as_challenger"] and not d["eligible_for_release_review"]
    assert "fairness gate fails" in d["reasons"]
