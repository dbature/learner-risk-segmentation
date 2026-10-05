"""Stage 2: early-window aggregation of studentVle.

studentVle.csv holds 10.6M daily click rows (433 MB). It is aggregated with
DuckDB, which streams the CSV instead of loading it into memory, down to one
row per registration covering only what was observable before day 30.

The WHERE clause matters. A registration gets a row only if it had VLE
activity before day 30. Using "has any VLE row at all" instead would leak
the future: a learner whose first click is on day 100 would look engaged at
day 30.

studentVle carries only module codes and integers, so DuckDB's UTF-8 CSV
reader is safe here even though the other OULAD tables need latin-1.

If studentVle.csv is not present but the pre-aggregated file produced by
make_vle_early_agg.py is, that file is used and the audit log says so.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.governance.audit import audit

STAGE = "aggregate_vle"
WINDOW_DAYS = 30

AGG_COLUMNS = [
    "code_module", "code_presentation", "id_student",
    "clicks_pre_start", "clicks_0_29",
    "clicks_wk1", "clicks_wk2", "clicks_wk3", "clicks_wk4",
    "active_days_0_29", "distinct_sites_0_29", "first_active_day",
]


def _sql(window: int) -> str:
    last = window - 1
    return f"""
        SELECT
            code_module, code_presentation, id_student,
            SUM(CASE WHEN date < 0 THEN sum_click ELSE 0 END)                 AS clicks_pre_start,
            SUM(CASE WHEN date BETWEEN 0 AND {last} THEN sum_click ELSE 0 END) AS clicks_0_29,
            SUM(CASE WHEN date BETWEEN 0 AND 6 THEN sum_click ELSE 0 END)     AS clicks_wk1,
            SUM(CASE WHEN date BETWEEN 7 AND 13 THEN sum_click ELSE 0 END)    AS clicks_wk2,
            SUM(CASE WHEN date BETWEEN 14 AND 20 THEN sum_click ELSE 0 END)   AS clicks_wk3,
            SUM(CASE WHEN date BETWEEN 21 AND {last} THEN sum_click ELSE 0 END) AS clicks_wk4,
            COUNT(DISTINCT CASE WHEN date BETWEEN 0 AND {last} THEN date END)    AS active_days_0_29,
            COUNT(DISTINCT CASE WHEN date BETWEEN 0 AND {last} THEN id_site END) AS distinct_sites_0_29,
            COALESCE(MIN(CASE WHEN date BETWEEN 0 AND {last} THEN date END), -1) AS first_active_day
        FROM read_csv(?, header = true, delim = ',',
                      columns = {{'code_module': 'VARCHAR', 'code_presentation': 'VARCHAR',
                                  'id_student': 'BIGINT', 'id_site': 'BIGINT',
                                  'date': 'INTEGER', 'sum_click': 'BIGINT'}})
        WHERE date < {window}
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
    """


def aggregate_with_duckdb(vle_csv: Path, window: int = WINDOW_DAYS) -> tuple[pd.DataFrame, int]:
    """Return the aggregate and the number of source rows scanned."""
    import duckdb

    con = duckdb.connect()
    try:
        df = con.execute(_sql(window), [str(vle_csv)]).df()
        (n_rows,) = con.execute(
            "SELECT count(*) FROM read_csv(?, header = true)", [str(vle_csv)]
        ).fetchone()
    finally:
        con.close()
    return _coerce(df), int(n_rows)


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    df = df[AGG_COLUMNS].copy()
    df["id_student"] = df["id_student"].astype("Int64")
    for column in AGG_COLUMNS[3:]:
        df[column] = df[column].astype("int64")
    return df


def build_early_window(raw_dir: Path, interim_dir: Path, window: int = WINDOW_DAYS) -> Path:
    raw_dir, interim_dir = Path(raw_dir), Path(interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)
    full = raw_dir / "studentVle.csv"
    pre = raw_dir / "vle_early_agg.csv"
    if full.exists():
        df, n_rows = aggregate_with_duckdb(full, window)
        audit(STAGE, "read", "studentVle", path=full, rows=n_rows,
              columns=["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"],
              note=f"streamed with DuckDB; rows with date >= {window} excluded")
    elif pre.exists():
        df = _coerce(pd.read_csv(pre, encoding="latin-1"))
        audit(STAGE, "read", "vle_early_agg", path=pre, rows=len(df), columns=df.columns,
              note="studentVle.csv absent; used the pre-aggregated file from make_vle_early_agg.py")
    else:
        raise FileNotFoundError(f"neither {full} nor {pre} exists")
    out = interim_dir / "vle_early_window.parquet"
    df.to_parquet(out, index=False)
    audit(STAGE, "write", "vle_early_window", path=out, rows=len(df), columns=df.columns)
    return out


if __name__ == "__main__":
    from src.config import default_paths
    from src.governance.audit import configure

    p = default_paths().make()
    configure(p.logs)
    print(build_early_window(p.raw, p.interim))
