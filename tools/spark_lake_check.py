"""Cross-check the Parquet lake with two independent engines: per-season row and game counts from Spark and DuckDB must
agree with each other, and their totals with pyarrow's own row count.

    python tools/spark_lake_check.py [--tables play_by_play player_boxscores team_boxscores]

Needs the optional packages (pip install -r requirements-bigdata.txt) and Java 17 or 21 for Spark. Read-only: Spark
only reads the lake and nothing here writes. Exit status 0 only when every table agrees everywhere.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.ingest import datalake  # noqa: E402

DEFAULT_TABLES = ("play_by_play", "player_boxscores", "team_boxscores")


def compare_counts(duck: dict, spark: dict, arrow_total: int) -> dict:
    """``duck`` and ``spark`` map season -> (rows, distinct games). Returns the per-season comparison and a verdict."""
    seasons = sorted(set(duck) | set(spark))
    rows = []
    for s in seasons:
        d, p = duck.get(s), spark.get(s)
        rows.append({"season_year": s, "duckdb": d, "spark": p, "ok": d is not None and d == p})
    totals = {"duckdb": sum(v[0] for v in duck.values()), "spark": sum(v[0] for v in spark.values()),
              "arrow": arrow_total}
    totals_ok = len(set(totals.values())) == 1
    return {"seasons": rows, "totals": totals, "totals_ok": totals_ok,
            "ok": totals_ok and all(r["ok"] for r in rows)}


def duck_counts(con, table: str) -> dict:
    q = (f'SELECT season_year, count(*), count(DISTINCT game_id) FROM "{table}" GROUP BY 1 ORDER BY 1')
    return {s: (int(n), int(g)) for s, n, g in con.execute(q).fetchall()}


def spark_counts(spark, table: str, root=None) -> dict:
    from pyspark.sql import functions as F
    df = datalake.spark_read(spark, table, root)
    out = df.groupBy("season_year").agg(F.count(F.lit(1)).alias("n"), F.countDistinct("game_id").alias("g")).collect()
    return {r["season_year"]: (int(r["n"]), int(r["g"])) for r in out}


def check_table(table: str, con, spark, root=None) -> dict:
    res = compare_counts(duck_counts(con, table), spark_counts(spark, table, root),
                         datalake.dataset(table, root).count_rows())
    res["table"] = table
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", nargs="+", default=list(DEFAULT_TABLES))
    ap.add_argument("--lake-root", type=Path, default=None)
    ap.add_argument("--memory", default=None, help="Spark driver memory (default $NBA_SPARK_MEMORY or 4g)")
    a = ap.parse_args(argv)

    root = a.lake_root or datalake.lake_root()
    have = datalake.tables(root)
    missing = [t for t in a.tables if t not in have]
    if missing:
        print(f"lake tables not found under {root}: {missing} (have: {have})", file=sys.stderr)
        return 2
    con = datalake.duck(root)
    spark = datalake.get_spark(memory=a.memory)
    ok = True
    try:
        for table in a.tables:
            print(f"checking {table} ...", flush=True)
            r = check_table(table, con, spark, root)
            ok &= r["ok"]
            print(f"  {'AGREE' if r['ok'] else 'DISAGREE'}: totals {r['totals']}, {len(r['seasons'])} seasons")
            for row in r["seasons"]:
                if not row["ok"]:
                    print(f"    - {row['season_year']}: duckdb={row['duckdb']} spark={row['spark']}")
    finally:
        spark.stop()
    print("ALL ENGINES AGREE" if ok else "ENGINES DISAGREE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
