"""Stage 6: anonymise and publish.

OULAD carries no names, emails, phone numbers or addresses. Its direct
identifier is id_student, already a pseudonymous number assigned by the Open
University. Two risks remain, and this stage handles both:

1. Linkability. id_student is the same number the source system uses, so
   anyone holding the source can re-link records. It is replaced with a keyed
   HMAC-SHA256 pseudonym. The key (PIPELINE_SALT) lives in the environment,
   never in the repository, so the pseudonym cannot be reversed or recomputed
   without it. The same learner gets the same key across presentations, which
   keeps the grouped train and test split possible.

2. Re-identification through quasi-identifiers. Gender, region, age band,
   disability, IMD band and education together can single a person out even
   with no name present (Sweeney, 2002). k-anonymity is measured on that
   combination and reported. The sensitive attributes are then split into a
   separate, access-restricted file used only for fairness auditing, so the
   analytical table carries none of them.

A PII guard also fails the run if a column that looks like a direct identifier
(name, email, phone, address, postcode, date of birth, IP) ever appears in
the source, or if any text value matches an email or phone pattern.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
from pathlib import Path

import pandas as pd

from src.config import SENSITIVE_ATTRIBUTES
from src.features.build_features import (
    CONTEXT_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    REGISTRATION_FEATURES,
)
from src.governance.audit import audit

STAGE = "anonymise"
SALT_ENV = "PIPELINE_SALT"
MIN_SALT_LENGTH = 16

PII_NAME_PATTERN = re.compile(
    r"(^|_)(first_?name|last_?name|surname|full_?name|name|email|e_?mail|phone|mobile|"
    r"address|street|postcode|post_?code|zip|dob|date_of_birth|birth_?date|ip_?address|ssn|nin)($|_)",
    re.IGNORECASE,
)
EMAIL_PATTERN = re.compile(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"(\+?\d[\d\s().-]{8,}\d)")

QUASI_IDENTIFIERS = ["gender", "region", "age_band", "disability", "imd_band", "highest_education"]


class PrivacyError(RuntimeError):
    pass


def get_salt() -> bytes:
    salt = os.environ.get(SALT_ENV, "")
    if len(salt) < MIN_SALT_LENGTH:
        raise PrivacyError(
            f"{SALT_ENV} must be set to a secret of at least {MIN_SALT_LENGTH} characters. "
            "It is never stored in the repository."
        )
    return salt.encode("utf-8")


def pseudonymise(ids: pd.Series, salt: bytes) -> pd.Series:
    def one(v) -> str:
        return hmac.new(salt, str(int(v)).encode("utf-8"), hashlib.sha256).hexdigest()[:16]

    return ids.map(one)


def pii_guard(df: pd.DataFrame, allowed: set[str] | None = None) -> list[str]:
    """Return the names of columns that look like direct identifiers."""
    allowed = allowed or set()
    flagged = [c for c in df.columns if c not in allowed and PII_NAME_PATTERN.search(c)]
    for c in df.select_dtypes(include=["object", "string"]).columns:
        if c in allowed or c in flagged:
            continue
        sample = df[c].dropna().astype(str)
        if sample.str.contains(EMAIL_PATTERN).any() or sample.str.fullmatch(PHONE_PATTERN).any():
            flagged.append(c)
    return flagged


def k_anonymity(df: pd.DataFrame, quasi: list[str], k: int = 5,
                person: str = "id_student") -> dict:
    """k counts distinct people, not rows: a learner registered on two
    presentations must not be allowed to make up two members of one class."""
    keyed = df[quasi + [person]].copy()
    keyed[quasi] = keyed[quasi].astype("string").fillna("<missing>")
    groups = keyed.drop_duplicates().groupby(quasi, observed=True)[person].nunique()
    below = groups[groups < k]
    n_people = int(keyed[person].nunique())
    return {
        "quasi_identifiers": quasi,
        "k_threshold": k,
        "min_k": int(groups.min()),
        "equivalence_classes": int(len(groups)),
        "classes_below_k": int(len(below)),
        "people_below_k": int(below.sum()),
        "people": n_people,
        "share_people_below_k": round(float(below.sum() / n_people), 4),
    }


def publish(integrated_path: Path, processed_dir: Path, salt: bytes) -> dict[str, Path]:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(integrated_path)

    flagged = pii_guard(df)
    if flagged:
        raise PrivacyError(f"possible direct identifiers in source: {flagged}")

    df["learner_key"] = pseudonymise(df["id_student"], salt)
    audit(STAGE, "pseudonymise", "integrated", rows=len(df), columns=["id_student", "learner_key"],
          note="HMAC-SHA256 with secret salt from environment, truncated to 16 hex characters")

    ident = ["learner_key"] + CONTEXT_COLUMNS
    pop = df[df["in_population"].eq(1)]

    outputs = {
        "model_ready": (pop[ident + FEATURE_COLUMNS + LABEL_COLUMNS], "analyst"),
        "audit_attributes": (pop[ident + SENSITIVE_ATTRIBUTES], "fairness_auditor"),
        "early_leavers": (
            df[df["left_by_day_30"].eq(1)][ident + REGISTRATION_FEATURES + ["date_unregistration"]],
            "analyst",
        ),
    }
    paths: dict[str, Path] = {}
    for name, (frame, tier) in outputs.items():
        frame = frame.reset_index(drop=True)
        if "id_student" in frame.columns:
            raise PrivacyError(f"{name} would publish id_student")
        out = processed_dir / f"{name}.parquet"
        frame.to_parquet(out, index=False)
        audit(STAGE, "write", name, path=out, rows=len(frame), columns=frame.columns,
              note=f"access tier: {tier}")
        paths[name] = out
    return paths


def privacy_report(integrated_path: Path) -> dict:
    df = pd.read_parquet(integrated_path)
    pop = df[df["in_population"].eq(1)]
    return {
        "direct_identifier_columns_in_source": pii_guard(df),
        "k_anonymity_all_quasi_identifiers": k_anonymity(pop, QUASI_IDENTIFIERS),
        "k_anonymity_without_region": k_anonymity(pop, [q for q in QUASI_IDENTIFIERS if q != "region"]),
    }


def promote(staged: dict[str, str], processed_dir: Path) -> dict[str, Path]:
    """Move validated files from staging into the published layer.

    Write, audit, publish: nothing reaches data/processed until the Great
    Expectations gate has passed on the staged copy.
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    final: dict[str, Path] = {}
    for name, src in staged.items():
        src = Path(src)
        dst = processed_dir / src.name
        src.replace(dst)
        audit(STAGE, "write", name, path=dst, note="promoted from staging after validation passed")
        final[name] = dst
        try:
            src.parent.rmdir()
        except OSError:
            pass
    return final
