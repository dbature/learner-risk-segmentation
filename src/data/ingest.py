"""Stage 1: ingest.

Reads the six small OULAD tables exactly once, with the two rules that the
Module 2 profiling showed are not optional:

* the files are latin-1 and fail to parse as UTF-8
* missing values are the literal string '?', never a blank

Only '?' is treated as missing. pandas' default list of null markers is
switched off so that a legitimate value such as "NA" could never be silently
nulled. Each table is checked against its expected columns and written to the
interim layer as Parquet, with every read and write in the audit log.

studentVle.csv is not read here. At 10.6M rows it is aggregated out of core
by src/features/early_window.py.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.governance.audit import audit

STAGE = "ingest"
ENCODING = "latin-1"
MISSING = ["?"]

SCHEMA: dict[str, list[str]] = {
    "courses": ["code_module", "code_presentation", "module_presentation_length"],
    "assessments": [
        "code_module", "code_presentation", "id_assessment",
        "assessment_type", "date", "weight",
    ],
    "studentInfo": [
        "code_module", "code_presentation", "id_student", "gender", "region",
        "highest_education", "imd_band", "age_band", "num_of_prev_attempts",
        "studied_credits", "disability", "final_result",
    ],
    "studentRegistration": [
        "code_module", "code_presentation", "id_student",
        "date_registration", "date_unregistration",
    ],
    "studentAssessment": [
        "id_assessment", "id_student", "date_submitted", "is_banked", "score",
    ],
    "vle": [
        "id_site", "code_module", "code_presentation",
        "activity_type", "week_from", "week_to",
    ],
}

# Nullable integer types keep "missing" distinct from zero after parsing.
DTYPES: dict[str, dict[str, str]] = {
    "courses": {"module_presentation_length": "Int64"},
    "assessments": {"id_assessment": "Int64", "date": "Int64", "weight": "float64"},
    "studentInfo": {
        "id_student": "Int64", "num_of_prev_attempts": "Int64", "studied_credits": "Int64",
    },
    "studentRegistration": {
        "id_student": "Int64", "date_registration": "Int64", "date_unregistration": "Int64",
    },
    "studentAssessment": {
        "id_assessment": "Int64", "id_student": "Int64", "date_submitted": "Int64",
        "is_banked": "Int64", "score": "float64",
    },
    "vle": {"id_site": "Int64", "week_from": "Int64", "week_to": "Int64"},
}


class SchemaError(ValueError):
    """Raised when a source file does not have the columns the pipeline expects."""


def read_oulad_csv(path: Path, table: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        encoding=ENCODING,
        na_values=MISSING,
        keep_default_na=False,
        dtype=str,
    )
    expected = SCHEMA[table]
    if list(df.columns) != expected:
        raise SchemaError(f"{table}: expected columns {expected}, found {list(df.columns)}")
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].str.strip()
    for column, dtype in DTYPES.get(table, {}).items():
        df[column] = pd.to_numeric(df[column]).astype(dtype)
    return df


def ingest_all(raw_dir: Path, interim_dir: Path) -> dict[str, Path]:
    raw_dir, interim_dir = Path(raw_dir), Path(interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for table in SCHEMA:
        src = raw_dir / f"{table}.csv"
        df = read_oulad_csv(src, table)
        audit(STAGE, "read", table, path=src, rows=len(df), columns=df.columns,
              note=f"encoding={ENCODING}; '?' parsed as null")
        out = interim_dir / f"{table}.parquet"
        df.to_parquet(out, index=False)
        audit(STAGE, "write", table, path=out, rows=len(df), columns=df.columns)
        outputs[table] = out
    return outputs


if __name__ == "__main__":
    from src.config import default_paths
    from src.governance.audit import configure

    p = default_paths().make()
    configure(p.logs)
    for name, path in ingest_all(p.raw, p.interim).items():
        print(f"{name:20s} -> {path}")
