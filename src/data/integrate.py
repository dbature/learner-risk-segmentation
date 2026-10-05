"""Stage 5: integrate.

Joins registration, demographics, assessment features and VLE features into
one row per registration, then defines who the model is allowed to learn from.

Every join is a LEFT join from studentRegistration with validate="1:1", and the
row count is asserted unchanged. An inner join would silently drop the 3,778
registrations with no VLE record, 80.9% of whom eventually withdraw.

Prediction population: registrations still active on day 30, meaning
date_unregistration is missing or later than day 30. 5,127 registrations (half
of all withdrawals) end on or before day 30. Their early silence records an
exit that already happened, so including them made a near-useless signal look
nine times stronger than it is. They are kept in a separate early-leaver table
for the registration-time segment.

Two labels are produced so Module 4 can choose the target on evidence:
  label_withdrew_after_30  unregistered after day 30
  label_non_completion     final_result is Withdrawn or Fail
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import KEY, SENSITIVE_ATTRIBUTES
from src.features.build_features import (
    WINDOW_DAYS,
    assessment_features,
    vle_features,
)
from src.governance.audit import audit

STAGE = "integrate"


class IntegrationError(ValueError):
    pass


def _left(base: pd.DataFrame, other: pd.DataFrame, name: str) -> pd.DataFrame:
    out = base.merge(other, on=KEY, how="left", validate="1:1")
    if len(out) != len(base):
        raise IntegrationError(f"join with {name} changed row count {len(base)} -> {len(out)}")
    return out


def integrate(
    registration: pd.DataFrame,
    info: pd.DataFrame,
    assessments: pd.DataFrame,
    student_assessment: pd.DataFrame,
    vle_window: pd.DataFrame,
    conflicts: pd.DataFrame,
    window: int = WINDOW_DAYS,
) -> pd.DataFrame:
    base = registration.copy()
    base = _left(base, info, "studentInfo")
    a = assessment_features(assessments, student_assessment, registration, window)
    base = _left(base, a, "assessment_features")
    v = vle_features(vle_window, registration)
    base = _left(base, v, "vle_features")

    base["date_registration_missing"] = base["date_registration"].isna().astype("int64")

    unreg = base["date_unregistration"]
    base["left_by_day_30"] = (unreg.notna() & (unreg <= window)).astype("int64")
    base["label_withdrew_after_30"] = (unreg.notna() & (unreg > window)).astype("int64")
    base["label_non_completion"] = base["final_result"].isin(["Withdrawn", "Fail"]).astype("int64")

    flagged = conflicts[KEY].assign(label_conflict=1)
    base = _left(base, flagged, "label_conflicts")
    base["label_conflict"] = base["label_conflict"].fillna(0).astype("int64")

    base["in_population"] = (
        base["left_by_day_30"].eq(0) & base["label_conflict"].eq(0)
    ).astype("int64")
    return base


def integrate_all(interim_dir: Path, quarantine_dir: Path) -> Path:
    interim_dir = Path(interim_dir)
    t = {
        n: pd.read_parquet(interim_dir / f"{n}_clean.parquet")
        for n in ["studentRegistration", "studentInfo", "assessments", "studentAssessment"]
    }
    vle = pd.read_parquet(interim_dir / "vle_early_window.parquet")
    conflicts = pd.read_parquet(Path(quarantine_dir) / "label_conflicts.parquet")
    df = integrate(
        t["studentRegistration"], t["studentInfo"], t["assessments"],
        t["studentAssessment"], vle, conflicts,
    )
    out = interim_dir / "integrated.parquet"
    df.to_parquet(out, index=False)
    audit(
        STAGE, "write", "integrated", path=out, rows=len(df), columns=df.columns,
        note=(
            f"left joins from studentRegistration; in_population={int(df['in_population'].sum())}; "
            f"left_by_day_30={int(df['left_by_day_30'].sum())}; "
            f"label_conflict={int(df['label_conflict'].sum())}; "
            f"sensitive attributes present: {', '.join(SENSITIVE_ATTRIBUTES)}"
        ),
    )
    return out


if __name__ == "__main__":
    from src.config import default_paths
    from src.governance.audit import configure

    p = default_paths().make()
    configure(p.logs)
    print(integrate_all(p.interim, p.quarantine))
