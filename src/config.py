"""Paths and parameters shared by every pipeline stage.

Every stage reads its settings from params.yaml so that a change to a
threshold is a reviewed commit, not an edit buried in code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Composite key of a registration. One learner can hold several.
KEY = ["code_module", "code_presentation", "id_student"]

# Fields that identify a person directly. Never written past the interim layer.
DIRECT_IDENTIFIERS = ["id_student"]

# Protected or quasi-identifying attributes. Kept out of the feature set and
# released only in the restricted audit file, for fairness measurement.
SENSITIVE_ATTRIBUTES = [
    "gender",
    "disability",
    "imd_band",
    "age_band",
    "region",
    "highest_education",
]


@dataclass(frozen=True)
class Paths:
    raw: Path
    interim: Path
    processed: Path
    quarantine: Path
    reports: Path
    logs: Path

    def make(self) -> "Paths":
        for p in (self.interim, self.processed, self.quarantine, self.reports, self.logs):
            p.mkdir(parents=True, exist_ok=True)
        return self


def load_params(path: Path | None = None) -> dict:
    path = path or ROOT / "params.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def default_paths(base: Path | None = None) -> Paths:
    base = Path(base) if base else ROOT
    return Paths(
        raw=base / "data" / "raw",
        interim=base / "data" / "interim",
        processed=base / "data" / "processed",
        quarantine=base / "data" / "quarantine",
        reports=base / "reports",
        logs=base / "logs",
    )
