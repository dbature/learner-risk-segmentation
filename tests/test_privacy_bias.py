"""Anonymisation, PII guard and representation bias suite."""
import pandas as pd
import pytest
from faker import Faker

from src.governance.anonymise import (
    PrivacyError,
    get_salt,
    k_anonymity,
    pii_guard,
    pseudonymise,
)


def test_salt_is_required(monkeypatch):
    monkeypatch.delenv("PIPELINE_SALT")
    with pytest.raises(PrivacyError):
        get_salt()


def test_pseudonym_is_stable_keyed_and_irreversible_without_salt():
    ids = pd.Series([11391, 11391, 28400])
    a = pseudonymise(ids, b"salt-one-0123456789")
    b = pseudonymise(ids, b"salt-two-0123456789")
    assert a.iloc[0] == a.iloc[1]  # same learner, same key: grouped splits still work
    assert a.iloc[0] != a.iloc[2]
    assert a.iloc[0] != b.iloc[0]  # a different secret gives unlinkable keys
    assert a.str.fullmatch(r"[0-9a-f]{16}").all()
    assert "11391" not in "".join(a)


def test_pii_guard_catches_faker_generated_identifiers():
    """OULAD has no direct identifiers. Faker builds the ones a future source
    might carry, to prove the guard would stop them before publication."""
    fake = Faker()
    Faker.seed(42)
    df = pd.DataFrame({
        "code_module": ["AAA", "BBB", "CCC"],
        "full_name": [fake.name() for _ in range(3)],
        "contact": [fake.email() for _ in range(3)],
        "notes": [fake.phone_number() for _ in range(3)],
        "region": ["Scotland", "Wales", "Ireland"],
    })
    flagged = set(pii_guard(df))
    assert {"full_name", "contact"} <= flagged
    assert "code_module" not in flagged and "region" not in flagged


def test_clean_frame_passes_pii_guard():
    df = pd.DataFrame({"code_module": ["AAA"], "region": ["Scotland"], "studied_credits": [60]})
    assert pii_guard(df) == []


def test_k_anonymity_counts_people_not_rows():
    """One learner on two presentations must not form a class of two."""
    df = pd.DataFrame({
        "id_student": [1, 1, 2, 3, 4],
        "gender": ["F", "F", "M", "M", "M"],
        "region": ["Wales", "Wales", "Wales", "Wales", "Wales"],
    })
    out = k_anonymity(df, ["gender", "region"], k=2)
    assert out["min_k"] == 1
    assert out["people_below_k"] == 1


def test_representation_suite_flags_small_and_dropped_groups(tmp_path):
    from src.fairness.representation import attribute_report

    df = pd.DataFrame({
        "imd_band": ["0-10%"] * 10 + ["90-100%"] * 10 + [None] * 2,
        "in_population": [1] * 5 + [0] * 5 + [1] * 10 + [1] * 2,
        "label_withdrew_after_30": [1, 0, 0, 0, 0] + [0] * 5 + [0] * 10 + [0, 0],
        "label_non_completion": [1] * 5 + [0] * 5 + [0] * 10 + [0, 0],
        "no_vle_activity": [0] * 22,
        "mean_score_by_30": [50.0] * 22,
    })
    report = attribute_report(df, "imd_band", min_group_size=3, min_retention_ratio=0.8)
    groups = {g["group"]: g for g in report["groups"]}
    assert "Missing" in groups  # missing IMD kept as its own group
    assert report["retention_ratio_min_over_max"] == 0.5
    assert any("retention ratio" in f for f in report["flags"])
    assert any("Missing" in f for f in report["flags"])
