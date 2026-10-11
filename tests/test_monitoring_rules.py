"""Monitoring rules on synthetic numbers: no dataset needed."""
import numpy as np
import pytest

pytest.importorskip("scipy")
from src.monitoring.build import ALERTS, batch_alerts, psi  # noqa: E402


def batch(**kw):
    b = {"roc_auc": 0.76, "recall": 0.45, "flag_rate": 0.25, "score_psi": 0.02,
         "data_drift": {"share_drifted": 0.1},
         "fairness": {"mitigated": {"gender": {"recall_gap": 0.01, "chi2_p": 0.6},
                                    "age_band": {"recall_gap": 0.02, "chi2_p": 0.5},
                                    "disability": {"recall_gap": 0.02, "chi2_p": 0.5},
                                    "imd_band": {"chi2_p": 0.4}}}}
    b.update(kw)
    return b


def test_healthy_batch_raises_nothing():
    assert batch_alerts(batch(), validation_recall=0.45) == []


def test_auc_below_floor_is_high():
    alerts = batch_alerts(batch(roc_auc=ALERTS["auc_floor"] - 0.01), 0.45)
    assert any(a["type"] == "performance" and a["severity"] == "high" for a in alerts)


def test_noisy_gap_is_low_but_detectable_gap_is_high():
    f = batch()["fairness"]
    f["mitigated"]["disability"] = {"recall_gap": 0.08, "chi2_p": 0.3}
    assert [a["severity"] for a in batch_alerts(batch(fairness=f), 0.45)] == ["low"]
    f["mitigated"]["disability"] = {"recall_gap": 0.08, "chi2_p": 0.01}
    assert [a["severity"] for a in batch_alerts(batch(fairness=f), 0.45)] == ["high"]


def test_psi_is_zero_for_identical_and_large_for_shifted():
    rng = np.random.default_rng(0)
    x = rng.beta(2, 3, 5000)
    assert psi(x, x) == pytest.approx(0, abs=1e-9)
    assert psi(x, np.clip(x + 0.3, 0, 1)) > ALERTS["psi_alert"]
