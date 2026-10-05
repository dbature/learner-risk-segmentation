"""Stage 3: clean.

Each rule here exists because profiling found the problem in the real files:

* imd_band has one label written "10-20" without the percent sign, so a
  string sort puts it in the wrong place. It is normalised and the bands are
  given an explicit order.
* Every table is checked for duplicate keys and exact duplicate rows. The
  OULAD files currently have none, but the check stays: a re-extract that
  duplicates rows would otherwise double-count learners silently.
* final_result and date_unregistration disagree for 102 registrations:
  93 are "Withdrawn" with no unregistration date, 9 are "Fail" with one.
  Withdrawal timing cannot be known for those rows, so they are quarantined
  with a reason rather than relabelled by guesswork.

Missing values are not imputed here. A missing imd_band is kept missing
because the 1,111 learners without one behave differently from every band;
imputing the median band would hide a group, not repair one.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import KEY
from src.governance.audit import audit

STAGE = "clean"

IMD_ORDER = [
    "0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
    "50-60%", "60-70%", "70-80%", "80-90%", "90-100%",
]
AGE_ORDER = ["0-35", "35-55", "55<="]
EDUCATION_ORDER = [
    "No Formal quals", "Lower Than A Level", "A Level or Equivalent",
    "HE Qualification", "Post Graduate Qualification",
]
FINAL_RESULTS = ["Distinction", "Pass", "Fail", "Withdrawn"]

TABLE_KEYS = {
    "courses": ["code_module", "code_presentation"],
    "assessments": ["id_assessment"],
    "studentInfo": KEY,
    "studentRegistration": KEY,
    "studentAssessment": ["id_assessment", "id_student"],
    "vle": ["id_site"],
}


class DataQualityError(ValueError):
    """Raised when a rule that should never break does break."""


def normalise_imd_band(series: pd.Series) -> pd.Series:
    fixed = series.replace({"10-20": "10-20%"})
    unknown = set(fixed.dropna()) - set(IMD_ORDER)
    if unknown:
        raise DataQualityError(f"unexpected imd_band labels: {sorted(unknown)}")
    return pd.Categorical(fixed, categories=IMD_ORDER, ordered=True)


def ordered(series: pd.Series, order: list[str], name: str) -> pd.Categorical:
    unknown = set(series.dropna()) - set(order)
    if unknown:
        raise DataQualityError(f"unexpected {name} labels: {sorted(unknown)}")
    return pd.Categorical(series, categories=order, ordered=True)


def check_duplicates(df: pd.DataFrame, table: str) -> dict[str, int]:
    exact = int(df.duplicated().sum())
    key = int(df.duplicated(TABLE_KEYS[table]).sum())
    if key > exact:
        raise DataQualityError(
            f"{table}: {key - exact} rows share a key but differ in content; "
            "this needs a human decision, not a silent drop"
        )
    return {"exact_duplicates": exact, "key_duplicates": key}


def label_conflicts(info: pd.DataFrame, registration: pd.DataFrame) -> pd.DataFrame:
    """Registrations whose outcome and unregistration date contradict each other."""
    m = info[KEY + ["final_result"]].merge(
        registration[KEY + ["date_unregistration"]], on=KEY, validate="1:1"
    )
    has_date = m["date_unregistration"].notna()
    withdrawn = m["final_result"].eq("Withdrawn")
    conflict = (withdrawn & ~has_date) | (~withdrawn & has_date)
    out = m.loc[conflict, KEY].copy()
    out["reason"] = (
        m.loc[conflict, "final_result"]
        .eq("Withdrawn")
        .map({True: "withdrawn_without_unregistration_date",
              False: "unregistration_date_but_not_withdrawn"})
    )
    return out.reset_index(drop=True)


def clean_all(interim_dir: Path, quarantine_dir: Path) -> dict[str, Path]:
    interim_dir, quarantine_dir = Path(interim_dir), Path(quarantine_dir)
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    tables = {t: pd.read_parquet(interim_dir / f"{t}.parquet") for t in TABLE_KEYS}
    report: dict[str, dict[str, int]] = {}

    for name, df in tables.items():
        report[name] = check_duplicates(df, name)
        tables[name] = df.drop_duplicates().reset_index(drop=True)

    info = tables["studentInfo"]
    info["imd_band"] = normalise_imd_band(info["imd_band"])
    info["age_band"] = ordered(info["age_band"], AGE_ORDER, "age_band")
    info["highest_education"] = ordered(info["highest_education"], EDUCATION_ORDER, "highest_education")
    info["final_result"] = ordered(info["final_result"], FINAL_RESULTS, "final_result")
    audit(STAGE, "transform", "studentInfo", rows=len(info),
          columns=["imd_band", "age_band", "highest_education", "final_result"],
          note="imd_band '10-20' normalised to '10-20%'; ordered categoricals applied")

    conflicts = label_conflicts(info, tables["studentRegistration"])
    q_path = quarantine_dir / "label_conflicts.parquet"
    conflicts.to_parquet(q_path, index=False)
    audit(STAGE, "quarantine", "label_conflicts", path=q_path, rows=len(conflicts),
          columns=conflicts.columns,
          note="; ".join(f"{k}={v}" for k, v in conflicts["reason"].value_counts().items()))

    outputs: dict[str, Path] = {}
    for name, df in tables.items():
        out = interim_dir / f"{name}_clean.parquet"
        df.to_parquet(out, index=False)
        audit(STAGE, "write", name, path=out, rows=len(df), columns=df.columns,
              note=f"duplicates found: {report[name]}")
        outputs[name] = out
    outputs["label_conflicts"] = q_path
    return outputs


if __name__ == "__main__":
    from src.config import default_paths
    from src.governance.audit import configure

    p = default_paths().make()
    configure(p.logs)
    print(clean_all(p.interim, p.quarantine))
