"""Stage 7: validation with Great Expectations.

Two suites, run as checkpoints, each a gate the flow cannot pass if it fails:

integrated_registrations
    The full joined table, before anything is published. Row count equals the
    source registration count (so no join dropped or duplicated learners), the
    composite key is unique, no raw '?' survived, categorical fields hold only
    known labels, numeric fields sit inside the ranges measured in profiling.

model_ready
    The analytical table handed to Module 4. Its column list must match
    exactly, which makes the leakage rule enforceable in the data itself and
    not only in a unit test: date_unregistration, final_result and id_student
    cannot be present. The pseudonymous key must have the expected format,
    labels must be binary, and the withdrawal base rate must stay inside a
    band, so a silent change upstream shows up as a failed expectation.

The suites are defined in code and saved to gx/expectations so they are
reviewed like any other change. Data Docs are rebuilt on every run.
"""
from __future__ import annotations

import json
from pathlib import Path

import great_expectations as gx
import pandas as pd

from src.config import ROOT
from src.data.clean import AGE_ORDER, EDUCATION_ORDER, FINAL_RESULTS, IMD_ORDER
from src.features.build_features import (
    CONTEXT_COLUMNS,
    FEATURE_COLUMNS,
    FORBIDDEN_FEATURES,
    LABEL_COLUMNS,
)
from src.governance.audit import audit

STAGE = "validate"
DATASOURCE = "pipeline_frames"
SENTINEL_REGEX = r"^\s*\?\s*$"


class ValidationFailed(RuntimeError):
    pass


def _context(project_root: Path):
    from great_expectations.data_context.types.base import ProgressBarsConfig

    ctx = gx.get_context(mode="file", project_root_dir=str(project_root))
    pb = ctx.variables.progress_bars
    globally = pb.get("globally") if isinstance(pb, dict) else getattr(pb, "globally", None)
    if globally is not False:
        ctx.variables.progress_bars = ProgressBarsConfig(globally=False)
        ctx.variables.save_config()
    return ctx


def _batch_request(ctx, asset_name: str, df: pd.DataFrame):
    ds = ctx.sources.add_or_update_pandas(DATASOURCE)
    try:
        asset = ds.get_asset(asset_name)
    except LookupError:
        asset = ds.add_dataframe_asset(asset_name)
    return asset.build_batch_request(dataframe=df)


def _shared_feature_expectations(v) -> None:
    v.expect_column_values_to_be_between("studied_credits", min_value=30, max_value=655)
    v.expect_column_values_to_be_between("num_of_prev_attempts", min_value=0, max_value=6)
    v.expect_column_values_to_be_between("date_registration", min_value=-400, max_value=200)
    v.expect_column_values_to_not_be_null("date_registration", mostly=0.99)
    for c in ["clicks_pre_start", "clicks_0_29", "clicks_wk1", "clicks_wk2",
              "clicks_wk3", "clicks_wk4", "distinct_sites_0_29",
              "n_due_by_30", "n_submitted_by_30", "n_banked_by_30"]:
        v.expect_column_values_to_be_between(c, min_value=0)
        v.expect_column_values_to_not_be_null(c)
    v.expect_column_values_to_be_between("active_days_0_29", min_value=0, max_value=30)
    v.expect_column_values_to_be_between("first_active_day", min_value=-1, max_value=29)
    v.expect_column_values_to_be_in_set("no_vle_activity", [0, 1])
    v.expect_column_values_to_be_between("submit_rate_by_30", min_value=0, max_value=1)
    v.expect_column_values_to_be_between("mean_score_by_30", min_value=0, max_value=100)


def define_integrated_suite(v, expected_rows: int) -> None:
    v.expect_table_row_count_to_equal(expected_rows)
    v.expect_compound_columns_to_be_unique(["code_module", "code_presentation", "id_student"])
    for c in ["code_module", "code_presentation", "id_student", "final_result"]:
        v.expect_column_values_to_not_be_null(c)
    v.expect_column_values_to_be_in_set("final_result", FINAL_RESULTS)
    v.expect_column_values_to_be_in_set("imd_band", IMD_ORDER)
    v.expect_column_values_to_not_be_null("imd_band", mostly=0.95)
    v.expect_column_values_to_be_in_set("age_band", AGE_ORDER)
    v.expect_column_values_to_be_in_set("highest_education", EDUCATION_ORDER)
    v.expect_column_values_to_be_in_set("gender", ["M", "F"])
    v.expect_column_values_to_be_in_set("disability", ["Y", "N"])
    for c in ["code_module", "code_presentation", "gender", "region", "disability"]:
        v.expect_column_values_to_not_match_regex(c, SENTINEL_REGEX)
    _shared_feature_expectations(v)
    for c in ["in_population", "left_by_day_30", "label_conflict"] + LABEL_COLUMNS:
        v.expect_column_values_to_be_in_set(c, [0, 1])
    v.expect_column_sum_to_be_between("label_conflict", min_value=0, max_value=200)


def define_model_ready_suite(v, min_rows: int, max_rows: int) -> None:
    # Forbidden names are filtered out of the expected list independently, so
    # even if someone adds date_unregistration to FEATURE_COLUMNS by mistake,
    # this expectation still fails on the data.
    allowed = [c for c in FEATURE_COLUMNS if c not in FORBIDDEN_FEATURES]
    expected = ["learner_key"] + CONTEXT_COLUMNS + allowed + LABEL_COLUMNS
    v.expect_table_columns_to_match_ordered_list(expected)
    v.expect_table_row_count_to_be_between(min_value=min_rows, max_value=max_rows)
    v.expect_column_values_to_not_be_null("learner_key")
    v.expect_column_values_to_match_regex("learner_key", r"^[0-9a-f]{16}$")
    v.expect_compound_columns_to_be_unique(["learner_key", "code_module", "code_presentation"])
    _shared_feature_expectations(v)
    for c in LABEL_COLUMNS:
        v.expect_column_values_to_not_be_null(c)
        v.expect_column_values_to_be_in_set(c, [0, 1])
    v.expect_column_mean_to_be_between("label_withdrew_after_30", min_value=0.15, max_value=0.21)


def _run(ctx, name: str, df: pd.DataFrame, define) -> dict:
    br = _batch_request(ctx, name, df)
    ctx.add_or_update_expectation_suite(expectation_suite_name=name)
    v = ctx.get_validator(batch_request=br, expectation_suite_name=name)
    define(v)
    v.save_expectation_suite(discard_failed_expectations=False)
    cp = ctx.add_or_update_checkpoint(
        name=f"{name}_checkpoint",
        validations=[{"batch_request": br, "expectation_suite_name": name}],
    )
    result = cp.run(run_name=name)
    stats = next(iter(result.run_results.values()))["validation_result"].statistics
    failed = [
        r.expectation_config.expectation_type
        + (f"({r.expectation_config.kwargs['column']})" if "column" in r.expectation_config.kwargs else "")
        for r in next(iter(result.run_results.values()))["validation_result"].results
        if not r.success
    ]
    return {
        "suite": name,
        "success": bool(result.success),
        "evaluated": int(stats["evaluated_expectations"]),
        "successful": int(stats["successful_expectations"]),
        "failed": failed,
    }


def validate_all(integrated_path: Path, model_ready_path: Path, reports_dir: Path,
                 params: dict, project_root: Path = ROOT) -> dict:
    ctx = _context(project_root)
    v_cfg = params["validation"]
    integrated = pd.read_parquet(integrated_path)
    model_ready = pd.read_parquet(model_ready_path)

    results = [
        _run(ctx, "integrated_registrations", integrated,
             lambda v: define_integrated_suite(v, v_cfg["expected_registrations"])),
        _run(ctx, "model_ready", model_ready,
             lambda v: define_model_ready_suite(v, v_cfg["model_ready_min_rows"],
                                                v_cfg["model_ready_max_rows"])),
    ]
    ctx.build_data_docs()

    summary = {"success": all(r["success"] for r in results), "suites": results}
    out = Path(reports_dir) / "validation_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for r, path, frame in [(results[0], integrated_path, integrated),
                           (results[1], model_ready_path, model_ready)]:
        audit(STAGE, "validate", r["suite"], path=path, rows=len(frame), columns=frame.columns,
              note=f"{r['successful']}/{r['evaluated']} expectations passed")
    if not summary["success"]:
        failed = {r["suite"]: r["failed"] for r in results if not r["success"]}
        raise ValidationFailed(f"Great Expectations gate failed: {failed}")
    return summary
