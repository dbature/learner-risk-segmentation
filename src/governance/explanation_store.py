"""Explanation store: an append-only audit record of every prediction.

Each call to /predict writes one JSON line holding what is needed to answer
"why was this learner flagged, by which model, under which rule?" months
later: a random id, the UTC time, the model's file hash and registry
version, a hash of the request, the day-30 inputs (no names or learner
keys exist in the API schema), the score, the threshold and rule applied,
the flag, and every feature's contribution to the score.

The store is append-only: records are never edited. An auditor retrieves a
record by its id; there is deliberately no endpoint that lists other
callers' records. In production the file would sit on durable, access
controlled storage with the retention period set in docs/governance.md; the
public demo writes to the container's local disk, which is cleared when the
container restarts.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

_LOCK = threading.Lock()
SCHEMA_VERSION = 1


def store_path() -> Path:
    return Path(os.environ.get("EXPLANATION_STORE", "reports/explanation_store/predictions.jsonl"))


def request_hash(inputs: dict) -> str:
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest()


def append(inputs: dict, score: float, threshold: float, rule: str, flagged: bool,
           contributions: dict, model: dict) -> str:
    rec_id = uuid.uuid4().hex
    record = {
        "id": rec_id, "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model, "request_sha256": request_hash(inputs), "inputs": inputs,
        "risk_score": round(float(score), 6), "threshold": round(float(threshold), 6),
        "decision_rule": rule, "flagged": bool(flagged),
        "contributions": {k: round(float(v), 6) for k, v in contributions.items()},
    }
    path = store_path()
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    return rec_id


def get(rec_id: str) -> dict | None:
    path = store_path()
    if not path.exists() or len(rec_id) != 32:
        return None
    with path.open(encoding="utf-8") as f:
        for line in f:
            if rec_id in line:
                rec = json.loads(line)
                if rec.get("id") == rec_id:
                    return rec
    return None


def count() -> int:
    path = store_path()
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for _ in f)
