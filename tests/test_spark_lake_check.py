"""Spark/DuckDB cross-check (tools/spark_lake_check.py). The comparison logic always runs; the real Spark run needs
PySpark + Java and is opt-in (RUN_SPARK=1) because starting a JVM takes ~15 s, like the RUN_PIT leakage check."""
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import spark_lake_check as slc  # noqa: E402
from src.ingest import datalake  # noqa: E402


def test_compare_counts_agree():
    duck = {"2019-20": (10, 2), "2020-21": (5, 1)}
    res = slc.compare_counts(duck, dict(duck), arrow_total=15)
    assert res["ok"] and res["totals_ok"] and all(r["ok"] for r in res["seasons"])


def test_compare_counts_flags_a_season_difference_a_missing_season_and_a_total_difference():
    duck = {"2019-20": (10, 2), "2020-21": (5, 1)}
    res = slc.compare_counts(duck, {"2019-20": (10, 2), "2020-21": (4, 1)}, arrow_total=15)
    assert not res["ok"] and [r["season_year"] for r in res["seasons"] if not r["ok"]] == ["2020-21"]
    res = slc.compare_counts(duck, {"2019-20": (10, 2)}, arrow_total=15)
    assert not res["ok"] and res["seasons"][1]["spark"] is None
    res = slc.compare_counts(duck, dict(duck), arrow_total=14)  # engines agree with each other but not with arrow
    assert not res["ok"] and not res["totals_ok"] and all(r["ok"] for r in res["seasons"])
    assert slc.compare_counts({}, {}, 0)["ok"]


def test_get_spark_explains_a_missing_java(monkeypatch):
    pytest.importorskip("pyspark")
    monkeypatch.setattr(datalake.shutil, "which", lambda _: None)
    monkeypatch.delenv("JAVA_HOME", raising=False)
    with pytest.raises(RuntimeError, match="Java 17 or 21"):
        datalake.get_spark()


def test_spark_and_duckdb_agree_on_a_synthetic_lake(tmp_path, monkeypatch):
    pytest.importorskip("pyspark")
    pytest.importorskip("duckdb")
    if os.environ.get("RUN_SPARK") != "1":
        pytest.skip("set RUN_SPARK=1 to start a local Spark session (needs Java 17/21)")
    if not (shutil.which("java") or os.environ.get("JAVA_HOME")):
        pytest.skip("no Java on PATH / JAVA_HOME")
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path))
    rows = []
    for season, yy in [("1996-97", "96"), ("2019-20", "19"), ("2025-26", "25")]:
        for g in (1, 2):
            for i in range(4):
                rows.append({"game_id": f"002{yy}{g:05d}", "season_year": season, "action_number": i})
    datalake.write_table("play_by_play", pd.DataFrame(rows))
    spark = datalake.get_spark(memory="1g", cores=2)
    try:
        sdf = datalake.spark_read(spark, "play_by_play")
        assert dict(sdf.dtypes)["season_year"] == "string"
        assert dict(sdf.dtypes)["game_id"] == "string"
        assert sdf.where("game_id = '0029600001'").count() == 4  # leading zeros survive through Spark
        res = slc.check_table("play_by_play", datalake.duck(), spark)
    finally:
        spark.stop()
    assert res["ok"], res
    assert res["totals"] == {"duckdb": 24, "spark": 24, "arrow": 24}
    assert {r["season_year"]: r["spark"] for r in res["seasons"]} == {
        "1996-97": (8, 2), "2019-20": (8, 2), "2025-26": (8, 2)}
