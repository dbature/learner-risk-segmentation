"""Orchestration DAG for the learner risk data pipeline (Prefect 3).

    ingest ────────┐
                   ├─> clean ─> integrate ─> anonymise ─> validate (GX) ─> promote
    aggregate_vle ─┘                                            └─> bias_check ─> privacy_report

anonymise writes to a staging folder. Only if both Great Expectations suites
pass does promote move the files into data/processed (write, audit, publish).

ingest and aggregate_vle have no dependency on each other and run
concurrently. Every later task waits on the output paths of the one before,
so the graph Prefect records is the graph drawn above. Tasks pass file paths,
not DataFrames, so each stage's output is inspectable, hashed in the audit
log, and reproducible on its own with `python -m src.<module>`.

Run locally:   python -m flows.oulad_pipeline
Run in Docker: see README, section "Run the pipeline".
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

from prefect import flow, get_run_logger, task

from src.config import default_paths, load_params
from src.data.clean import clean_all
from src.data.ingest import ingest_all
from src.data.integrate import integrate_all
from src.data.validate import validate_all
from src.fairness.representation import run_bias_suite
from src.features.early_window import build_early_window
from src.governance.anonymise import get_salt, privacy_report, promote, publish
from src.governance.audit import audit, configure, run_id

# Library notices that are not about this pipeline's data.
warnings.filterwarnings("ignore", message=".*result_format.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*parseString.*")


@task(name="ingest", retries=2, retry_delay_seconds=5)
def t_ingest(raw: str, interim: str) -> dict[str, str]:
    return {k: str(v) for k, v in ingest_all(Path(raw), Path(interim)).items()}


@task(name="aggregate_vle", retries=1)
def t_aggregate_vle(raw: str, interim: str, window: int) -> str:
    return str(build_early_window(Path(raw), Path(interim), window))


@task(name="clean")
def t_clean(interim: str, quarantine: str, _ingested: dict) -> dict[str, str]:
    return {k: str(v) for k, v in clean_all(Path(interim), Path(quarantine)).items()}


@task(name="integrate")
def t_integrate(interim: str, quarantine: str, _cleaned: dict, _vle: str) -> str:
    return str(integrate_all(Path(interim), Path(quarantine)))


@task(name="anonymise")
def t_anonymise(integrated: str, processed: str) -> dict[str, str]:
    return {k: str(v) for k, v in publish(Path(integrated), Path(processed), get_salt()).items()}


@task(name="promote")
def t_promote(staged: dict, processed: str, _validated: dict) -> dict[str, str]:
    return {k: str(v) for k, v in promote(staged, Path(processed)).items()}


@task(name="validate_gx")
def t_validate(integrated: str, model_ready: str, reports: str, params: dict) -> dict:
    return validate_all(Path(integrated), Path(model_ready), Path(reports), params)


@task(name="bias_check")
def t_bias(integrated: str, reports: str, params: dict, _validated: dict) -> dict:
    b = params["bias"]
    summary = run_bias_suite(Path(integrated), Path(reports), b["attributes"],
                             b["min_group_size"], b["min_retention_ratio"])
    if b.get("fail_on_flag") and summary["flag_count"]:
        raise RuntimeError(f"{summary['flag_count']} representation bias flags raised")
    return summary


@task(name="privacy_report")
def t_privacy(integrated: str, reports: str, _bias: dict) -> dict:
    report = privacy_report(Path(integrated))
    out = Path(reports) / "privacy_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    audit("privacy_report", "validate", "integrated", path=out,
          note=f"min_k={report['k_anonymity_all_quasi_identifiers']['min_k']}")
    return report


@flow(name="oulad-learner-risk-pipeline", log_prints=True)
def pipeline(base_dir: str | None = None) -> dict:
    log = get_run_logger()
    params = load_params()
    p = default_paths(Path(base_dir) if base_dir else None).make()
    configure(p.logs)
    get_salt()  # fail fast, before any data is read, if the secret is missing
    log.info("run_id=%s", run_id())

    ingested = t_ingest.submit(str(p.raw), str(p.interim))
    vle = t_aggregate_vle.submit(str(p.raw), str(p.interim), params["early_window_days"])
    cleaned = t_clean.submit(str(p.interim), str(p.quarantine), ingested)
    integrated = t_integrate.submit(str(p.interim), str(p.quarantine), cleaned, vle)
    staged = t_anonymise.submit(integrated, str(p.processed / "_staging"))
    validated = t_validate.submit(integrated, staged.result()["model_ready"],
                                  str(p.reports), params)
    published = t_promote.submit(staged, str(p.processed), validated)
    bias = t_bias.submit(integrated, str(p.reports), params, validated)
    privacy = t_privacy.submit(integrated, str(p.reports), bias)

    result = {
        "run_id": run_id(),
        "outputs": published.result(),
        "validation": validated.result(),
        "bias_flags": bias.result()["flag_count"],
        "privacy": privacy.result()["k_anonymity_all_quasi_identifiers"],
    }
    log.info("validation success=%s; bias flags=%s", result["validation"]["success"],
             result["bias_flags"])
    return result


if __name__ == "__main__":
    pipeline()
