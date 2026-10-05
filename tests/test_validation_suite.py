"""The Great Expectations model_ready suite must reject a leaky table.

Skipped in CI, which installs only the light dependency set. Runs locally and
in the Docker image, where great-expectations is installed.
"""
import pytest

gx = pytest.importorskip("great_expectations")


def _validate(tmp_path, df):
    from src.data.validate import _batch_request, _context, define_model_ready_suite

    ctx = _context(tmp_path)
    v = ctx.get_validator(batch_request=_batch_request(ctx, "probe", df),
                          create_expectation_suite_with_name="probe")
    define_model_ready_suite(v, min_rows=1, max_rows=10)
    return v.validate()


def _good_frame():
    import pandas as pd

    from src.features.build_features import FEATURE_COLUMNS, LABEL_COLUMNS

    row = {c: 0 for c in FEATURE_COLUMNS}
    row.update(studied_credits=60, date_registration=-20, submit_rate_by_30=1.0,
               mean_score_by_30=70.0, first_active_day=0)
    rows = []
    for i, label in enumerate([0, 0, 0, 0, 1]):
        r = dict(row, learner_key=f"{i:016x}", code_module="AAA", code_presentation="2013J")
        r.update({LABEL_COLUMNS[0]: label, LABEL_COLUMNS[1]: label})
        rows.append(r)
    cols = ["learner_key", "code_module", "code_presentation", *FEATURE_COLUMNS, *LABEL_COLUMNS]
    return pd.DataFrame(rows)[cols]


def test_clean_table_passes(tmp_path):
    assert _validate(tmp_path, _good_frame()).success


def test_leaky_column_fails_the_gate(tmp_path):
    df = _good_frame()
    df["date_unregistration"] = 45
    result = _validate(tmp_path, df)
    assert not result.success
    failed = {r.expectation_config.expectation_type for r in result.results if not r.success}
    assert "expect_table_columns_to_match_ordered_list" in failed
