"""Privacy audit log.

Every read, write and transformation of learner data is recorded as one JSON
line in logs/privacy_audit.jsonl: who ran it, which run, which stage, which
file, how many rows, which columns, the file's SHA-256, and whether sensitive
attributes or direct identifiers were touched.

The log records metadata only. It never contains a learner value, so the
audit trail cannot itself become a privacy leak.
"""
from __future__ import annotations

import datetime as dt
import getpass
import hashlib
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Iterable

from src.config import DIRECT_IDENTIFIERS, SENSITIVE_ATTRIBUTES

LOGGER_NAME = "privacy_audit"
LOG_FILE = "privacy_audit.jsonl"

_RUN_ID = os.environ.get("PIPELINE_RUN_ID") or uuid.uuid4().hex[:12]


class _JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


def configure(log_dir: Path) -> logging.Logger:
    """Point the audit log at log_dir. One run writes to exactly one file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    target = str((log_dir / LOG_FILE).resolve())
    for h in list(logger.handlers):
        if getattr(h, "baseFilename", None) != target:
            logger.removeHandler(h)
            h.close()
    if not logger.handlers:
        handler = logging.FileHandler(target, encoding="utf-8")
        handler.setFormatter(_JsonLineFormatter())
        logger.addHandler(handler)
    return logger


def run_id() -> str:
    return _RUN_ID


def actor() -> str:
    explicit = os.environ.get("PIPELINE_ACTOR")
    if explicit:
        return explicit
    try:
        return getpass.getuser()
    except Exception:  # containers without a passwd entry
        return "unknown"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _display(path: Path | None) -> str | None:
    if not path:
        return None
    try:
        return Path(path).resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def audit(
    stage: str,
    action: str,
    dataset: str,
    *,
    path: Path | None = None,
    rows: int | None = None,
    columns: Iterable[str] | None = None,
    note: str | None = None,
) -> dict:
    """Write one audit event and return it.

    action is one of: read, write, transform, validate, quarantine, pseudonymise.
    """
    cols = sorted(columns) if columns is not None else []
    event = {
        "ts_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "run_id": _RUN_ID,
        "actor": actor(),
        "stage": stage,
        "action": action,
        "dataset": dataset,
        "path": _display(path),
        "sha256": file_sha256(path) if path and Path(path).is_file() else None,
        "rows": rows,
        "n_columns": len(cols) if columns is not None else None,
        "sensitive_columns": sorted(set(cols) & set(SENSITIVE_ATTRIBUTES)),
        "direct_identifiers": sorted(set(cols) & set(DIRECT_IDENTIFIERS)),
        "note": note,
    }
    logging.getLogger(LOGGER_NAME).info(json.dumps(event))
    return event
