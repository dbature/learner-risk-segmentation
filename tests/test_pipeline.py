"""End-to-end stage tests on the miniature raw layer from conftest.tiny_raw.

These run the real stage functions in order (everything except the
Great Expectations and Prefect layers, which are exercised by the full run)
and check the rules that matter most: no learner lost in a join, the day-30
population, no future information in the VLE features, quarantine, and no
identifier in anything published.
"""
import json

import pandas as pd
import pytest

from src.config import default_paths
from src.data.clean import clean_all
from src.data.ingest import ingest_all
from src.data.integrate import integrate_all
from src.features.build_features import FEATURE_COLUMNS, FORBIDDEN_FEATURES, LABEL_COLUMNS
from src.features.early_window import aggregate_with_duckdb, build_early_window
from src.governance.anonymise import get_salt, promote, publish
from src.governance.audit import configure


@pytest.fixture
def run(tiny_raw):
    p = default_paths(tiny_raw).make()
    configure(p.logs)
    ingest_all(p.raw, p.interim)
    build_early_window(p.raw, p.interim)
    clean_all(p.interim, p.quarantine)
    integrated = integrate_all(p.interim, p.quarantine)
    staged = publish(integrated, p.processed / "_staging", get_salt())
    published = promote({k: str(v) for k, v in staged.items()}, p.processed)
    return p, pd.read_parquet(integrated), published


def test_duckdb_window_excludes_future_activity(tiny_raw):
    agg, n = aggregate_with_duckdb(tiny_raw / "data" / "raw" / "studentVle.csv")
    assert n == 7
    by = agg.set_index("id_student")
    assert 5 not in by.index  # only clicked on day 40
    one = by.loc[1]
    assert (one.clicks_pre_start, one.clicks_0_29, one.clicks_wk1, one.clicks_wk4) == (3, 12, 10, 2)
    assert (one.active_days_0_29, one.distinct_sites_0_29, one.first_active_day) == (3, 2, 0)


def test_no_registration_lost_in_integration(run):
    _, integrated, _ = run
    assert len(integrated) == 5
    assert integrated[["code_module", "code_presentation", "id_student"]].duplicated().sum() == 0


def test_population_and_labels(run):
    _, df, _ = run
    s = df.set_index("id_student")
    assert s.loc[2, "left_by_day_30"] == 1 and s.loc[2, "in_population"] == 0
    assert s.loc[3, "label_withdrew_after_30"] == 1 and s.loc[3, "in_population"] == 1
    assert s.loc[4, "label_conflict"] == 1 and s.loc[4, "in_population"] == 0
    assert s.loc[5, "label_non_completion"] == 1 and s.loc[5, "label_withdrew_after_30"] == 0


def test_never_active_is_minus_one_not_day_zero(run):
    _, df, _ = run
    s = df.set_index("id_student")
    assert s.loc[5, "no_vle_activity"] == 1
    assert s.loc[5, "first_active_day"] == -1
    assert s.loc[5, "clicks_0_29"] == 0


def test_assessment_features_respect_day_30(run):
    _, df, _ = run
    s = df.set_index("id_student")
    assert s.loc[1, "n_due_by_30"] == 1  # the exam and the day-60 TMA are excluded
    assert s.loc[1, "n_submitted_by_30"] == 1 and s.loc[1, "mean_score_by_30"] == 78
    assert s.loc[3, "n_submitted_by_30"] == 0  # submitted on day 35, too late to count
    assert pd.isna(s.loc[3, "mean_score_by_30"])


def test_missing_imd_kept_missing_and_label_fixed(run):
    _, df, _ = run
    s = df.set_index("id_student")
    assert pd.isna(s.loc[5, "imd_band"])
    assert s.loc[2, "imd_band"] == "10-20%"
    assert s.loc[5, "date_registration_missing"] == 1


def test_published_tables_carry_no_identifier_or_leaky_column(run):
    p, _, published = run
    model_ready = pd.read_parquet(published["model_ready"])
    assert list(model_ready.columns) == ["learner_key", "code_module", "code_presentation",
                                         *FEATURE_COLUMNS, *LABEL_COLUMNS]
    assert not (FORBIDDEN_FEATURES - set(LABEL_COLUMNS)) & set(model_ready.columns)
    assert sorted(model_ready["learner_key"].str.len().unique()) == [16]
    for path in published.values():
        assert "id_student" not in pd.read_parquet(path).columns
    assert not (p.processed / "_staging").exists()


def test_audit_log_has_metadata_only(run):
    p, _, _ = run
    events = [json.loads(line) for line in (p.logs / "privacy_audit.jsonl").read_text().splitlines()]
    assert {e["stage"] for e in events} >= {"ingest", "aggregate_vle", "clean", "integrate", "anonymise"}
    assert all(len(e["sha256"]) == 64 for e in events if e["action"] == "write" and e["sha256"])
    text = (p.logs / "privacy_audit.jsonl").read_text()
    assert "East Anglian Region" not in text  # no learner values, only metadata
    integrate_event = next(e for e in events if e["stage"] == "integrate")
    assert "imd_band" in integrate_event["sensitive_columns"]
    assert integrate_event["direct_identifiers"] == ["id_student"]
