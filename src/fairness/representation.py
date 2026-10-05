"""Stage 8: representation bias suite.

Checks the prepared data, before any model exists, for three ways a group can
be misrepresented (Suresh and Guttag, 2021):

1. Size. A group too small to estimate a recall for cannot be audited later.
   Groups below min_group_size in the prediction population are flagged.
2. Retention. Defining the day-30 population drops early leavers. If that
   filter removes one group much faster than another, the model will learn
   from a population skewed against that group. The ratio of the lowest to
   the highest group retention rate is checked against min_retention_ratio,
   in the spirit of the four-fifths rule.
3. Signal coverage. If one group is far more likely to have no VLE activity
   or no early assessment score, the model sees less about that group.

Group statistics are computed with Fairlearn's MetricFrame (Bird et al., 2020).
Missing IMD band is reported as its own group rather than dropped.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, count

from src.governance.audit import audit

STAGE = "bias_check"
MISSING_LABEL = "Missing"


def _mean_of(y_true, y_pred, values):  # noqa: ARG001 - MetricFrame signature
    return float(np.mean(values))


def _rate(y_true, y_pred):  # noqa: ARG001
    return float(np.mean(y_true))


def _groups(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(MISSING_LABEL)


def attribute_report(df: pd.DataFrame, attribute: str, min_group_size: int,
                     min_retention_ratio: float) -> dict:
    groups_all = _groups(df[attribute])
    retention = MetricFrame(
        metrics={"registrations": count, "retained_to_day_30": _rate},
        y_true=df["in_population"].to_numpy(),
        y_pred=df["in_population"].to_numpy(),
        sensitive_features=groups_all,
    )

    pop = df[df["in_population"].eq(1)]
    groups_pop = _groups(pop[attribute])
    y = pop["label_withdrew_after_30"].to_numpy()
    mf = MetricFrame(
        metrics={
            "population_n": count,
            "withdrew_after_30_rate": _rate,
            "non_completion_rate": _mean_of,
            "no_vle_activity_rate": _mean_of,
            "no_early_score_rate": _mean_of,
        },
        y_true=y,
        y_pred=y,
        sensitive_features=groups_pop,
        sample_params={
            "non_completion_rate": {"values": pop["label_non_completion"].to_numpy()},
            "no_vle_activity_rate": {"values": pop["no_vle_activity"].to_numpy()},
            "no_early_score_rate": {"values": pop["mean_score_by_30"].isna().astype(int).to_numpy()},
        },
    )

    table = retention.by_group.join(mf.by_group, how="outer")
    for c in ["registrations", "population_n"]:
        table[c] = table[c].fillna(0).astype(int)
    table["share_of_registrations"] = table["registrations"] / table["registrations"].sum()
    table["share_of_population"] = table["population_n"] / table["population_n"].sum()
    table = table.reset_index().rename(columns={table.index.name or "sensitive_feature_0": "group"})

    retention_ratio = float(retention.ratio(method="between_groups")["retained_to_day_30"])
    small = table.loc[table["population_n"] < min_group_size, "group"].tolist()

    flags = []
    if small:
        flags.append(f"groups below {min_group_size} learners in the population: {small}")
    if retention_ratio < min_retention_ratio:
        flags.append(
            f"retention ratio {retention_ratio:.3f} is below {min_retention_ratio}: "
            "the day-30 filter removes some groups disproportionately"
        )

    return {
        "attribute": attribute,
        "retention_ratio_min_over_max": round(retention_ratio, 4),
        "withdrew_after_30_rate_gap": round(float(mf.difference(method="between_groups")["withdrew_after_30_rate"]), 4),
        "no_vle_activity_rate_gap": round(float(mf.difference(method="between_groups")["no_vle_activity_rate"]), 4),
        "flags": flags,
        "groups": [
            {k: (round(float(v), 4) if isinstance(v, (float, np.floating))
                 else int(v) if isinstance(v, (int, np.integer)) else v)
             for k, v in row.items()}
            for row in table.to_dict(orient="records")
        ],
    }


def run_bias_suite(integrated_path: Path, reports_dir: Path, attributes: list[str],
                   min_group_size: int, min_retention_ratio: float) -> dict:
    df = pd.read_parquet(integrated_path)
    df = df[df["label_conflict"].eq(0)]
    with warnings.catch_warnings():
        # fairlearn 0.10 triggers a pandas groupby deprecation notice; harmless here
        warnings.simplefilter("ignore", DeprecationWarning)
        reports = [attribute_report(df, a, min_group_size, min_retention_ratio) for a in attributes]
    summary = {
        "population_n": int(df["in_population"].sum()),
        "registrations_n": int(len(df)),
        "attributes": reports,
        "flag_count": sum(len(r["flags"]) for r in reports),
    }
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / "representation_bias.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = [dict(attribute=r["attribute"], **g) for r in reports for g in r["groups"]]
    pd.DataFrame(rows).to_csv(reports_dir / "representation_bias.csv", index=False)
    audit(STAGE, "validate", "integrated", path=out, rows=len(df), columns=attributes,
          note=f"{summary['flag_count']} representation flags raised")
    return summary
