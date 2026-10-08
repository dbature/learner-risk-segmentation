"""API demo: sends the three example learners to /predict and prints the answers.

    uvicorn src.serving.api:app --port 8000      # in one terminal
    python scripts/api_demo.py                    # in another
    python scripts/api_demo.py --in-process       # no server needed (FastAPI TestClient)

The same examples appear in the stakeholder deck and the dashboard, so the
three outputs can be checked against both.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIELDS = ["code_module", "num_of_prev_attempts", "studied_credits", "date_registration", "n_due_by_30",
          "n_submitted_by_30", "n_banked_by_30", "mean_score_by_30", "clicks_pre_start", "clicks_0_29",
          "clicks_wk1", "clicks_wk2", "clicks_wk3", "clicks_wk4", "active_days_0_29", "distinct_sites_0_29",
          "first_active_day"]
INTS = set(FIELDS) - {"code_module", "date_registration", "mean_score_by_30"}


def payloads() -> dict:
    ex = json.loads((ROOT / "dashboard" / "data" / "examples.json").read_text())
    out = {}
    for k, v in ex.items():
        f = v["features"]
        out[k] = {c: (int(f[c]) if c in INTS else f[c]) for c in FIELDS}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--in-process", action="store_true")
    args = ap.parse_args()
    if args.in_process:
        from fastapi.testclient import TestClient

        from src.serving.api import app
        client, base = TestClient(app), ""
    else:
        import httpx
        client, base = httpx.Client(timeout=30), args.url.rstrip("/")
    print("GET /health ->", client.get(f"{base}/health").json())
    for name, body in payloads().items():
        r = client.post(f"{base}/predict", json=body)
        r.raise_for_status()
        res = r.json()
        reasons = "; ".join(f"{f['factor']} ({f['direction']})" for f in res["top_factors"])
        print(f"\nLearner {name}: risk {res['risk_score']:.4f}, flagged {res['flagged']} "
              f"(threshold {res['threshold']:.4f}, {res['decision_rule']})\n  top factors: {reasons}")


if __name__ == "__main__":
    main()
