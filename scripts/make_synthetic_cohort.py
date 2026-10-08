"""Writes dashboard/sample_cohort.csv: 60 entirely synthetic learners for the
coach caseload demo. No row comes from, or is derived from, a real learner;
values are drawn from simple rules with a fixed seed, and every reference
starts with SYN- so the file cannot be mistaken for real data.

    python scripts/make_synthetic_cohort.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
PATTERNS = {"steady": [90, 85, 95, 90], "slipping": [60, 25, 15, 5], "late": [2, 15, 55, 30], "quiet": [3, 1, 0, 0]}
WEEK_DAYS = [7, 7, 7, 9]


def learner(i: int, rng: np.random.Generator) -> dict:
    pattern = rng.choice(list(PATTERNS), p=[0.4, 0.25, 0.15, 0.2])
    weeks = [int(max(0, round(b * rng.lognormal(0, 0.45)))) for b in PATTERNS[pattern]]
    total = sum(weeks)
    max_days = sum(min(d, w) for d, w in zip(WEEK_DAYS, weeks))
    active_weeks = sum(w > 0 for w in weeks)
    days = int(min(max_days, max(active_weeks, round(total / rng.uniform(8, 20))))) if total else 0
    first = next((7 * k + int(rng.integers(0, 3)) for k, w in enumerate(weeks) if w > 0), -1)
    due = int(rng.choice([0, 1, 1, 2]))
    keen = {"steady": 0.95, "slipping": 0.7, "late": 0.6, "quiet": 0.25}[pattern]
    submitted = int(rng.binomial(due, keen)) if due else 0
    return {
        "learner_ref": f"SYN-{i:03d}", "code_module": str(rng.choice(MODULES)),
        "num_of_prev_attempts": int(rng.choice([0, 0, 0, 0, 1])), "studied_credits": int(rng.choice([30, 60, 60, 90, 120])),
        "date_registration": int(rng.integers(-180, -5)), "n_due_by_30": due, "n_submitted_by_30": submitted,
        "n_banked_by_30": 0, "mean_score_by_30": round(float(rng.normal(72, 14)), 1) if submitted else None,
        "clicks_pre_start": int(rng.integers(0, 200)) if pattern != "quiet" else int(rng.integers(0, 15)),
        "clicks_0_29": total, "clicks_wk1": weeks[0], "clicks_wk2": weeks[1], "clicks_wk3": weeks[2],
        "clicks_wk4": weeks[3], "active_days_0_29": days,
        "distinct_sites_0_29": int(min(total, rng.integers(1, 45))) if total else 0, "first_active_day": first,
    }


def main(n: int = 60, seed: int = 2026) -> Path:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame([learner(i + 1, rng) for i in range(n)])
    df["mean_score_by_30"] = df["mean_score_by_30"].clip(0, 100)
    path = ROOT / "dashboard" / "sample_cohort.csv"
    df.to_csv(path, index=False)
    print(f"wrote {len(df)} synthetic learners to {path}")
    return path


if __name__ == "__main__":
    main()
