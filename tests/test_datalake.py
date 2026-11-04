"""Parquet lake layout (src/ingest/datalake.py) on tiny synthetic frames: no network, no real data, no DuckDB needed
(the DuckDB parity test skips itself when duckdb is not installed)."""
import sys
import types
import warnings
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from src.ingest import datalake


@pytest.fixture(autouse=True)
def lake_env(tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path))
    return tmp_path


def pbp(n_per_season=3):
    rows = []
    for season, yy in [("1996-97", "96"), ("2019-20", "19"), ("2025-26", "25")]:
        for i in range(1, n_per_season + 1):
            rows.append({"game_id": f"002{yy}{i:05d}", "season_year": season, "action_number": i,
                         "description": f"play {i}", "score_home": float(i)})
    return pd.DataFrame(rows)


def test_lake_root_follows_nba_data_dir(lake_env, monkeypatch):
    assert datalake.lake_root() == lake_env.resolve() / "lake"
    monkeypatch.delenv("NBA_DATA_DIR")
    assert datalake.lake_root() == datalake.REPO.resolve() / "data" / "lake"


def test_game_id_leading_zeros_round_trip():
    df = pbp()
    datalake.write_table("play_by_play", df)
    back = datalake.read_pandas("play_by_play")
    assert pd.api.types.is_string_dtype(back["game_id"])
    assert set(back["game_id"]) == set(df["game_id"])
    assert back["game_id"].str.len().eq(10).all()
    assert "0029600001" in set(back["game_id"])


def test_integer_game_id_is_refused():
    df = pbp()
    df["game_id"] = df["game_id"].astype("int64")  # zeros already lost
    with pytest.raises(ValueError, match="game_id must be a string"):
        datalake.write_table("play_by_play", df)


def test_partitioned_tables_get_one_directory_per_season():
    base = datalake.write_table("play_by_play", pbp())
    assert sorted(p.name for p in base.iterdir()) == ["season_year=1996-97", "season_year=2019-20", "season_year=2025-26"]
    back = datalake.read_pandas("play_by_play")
    assert pd.api.types.is_string_dtype(back["season_year"])
    assert sorted(back["season_year"].unique()) == ["1996-97", "2019-20", "2025-26"]


def test_other_tables_are_flat():
    odds = pd.DataFrame({"game_id": ["0021900001", "0021900002"], "decimal_home": [1.5, 2.5]})
    base = datalake.write_table("game_odds", odds)
    assert not [p for p in base.iterdir() if p.is_dir()]
    assert len(list(base.glob("*.parquet"))) == 1


def test_partitioned_table_requires_season_year():
    with pytest.raises(ValueError, match="season_year"):
        datalake.write_table("play_by_play", pbp().drop(columns="season_year"))
    bad = pbp()
    bad.loc[0, "season_year"] = None
    with pytest.raises(ValueError, match="nulls"):
        datalake.write_table("play_by_play", bad)


def test_filters_and_columns():
    datalake.write_table("play_by_play", pbp())
    one = datalake.read_pandas("play_by_play", filters=[("season_year", "==", "2019-20")])
    assert set(one["season_year"]) == {"2019-20"} and len(one) == 3
    cols = datalake.read_pandas("play_by_play", columns=["game_id", "action_number"],
                                filters=[("season_year", "in", ["1996-97", "2025-26"]), ("action_number", ">", 1)])
    assert list(cols.columns) == ["game_id", "action_number"]
    assert len(cols) == 4 and cols["action_number"].min() == 2
    either = datalake.read_pandas("play_by_play", filters=[[("season_year", "==", "1996-97")],
                                                           [("game_id", "==", "0021900001")]])
    assert len(either) == 4


def test_append_adds_and_overwrite_replaces():
    datalake.write_table("play_by_play", pbp(2))
    datalake.write_table("play_by_play", pbp(2), mode="append")
    assert len(datalake.read_pandas("play_by_play")) == 12
    datalake.write_table("play_by_play", pbp(1))
    assert len(datalake.read_pandas("play_by_play")) == 3
    with pytest.raises(ValueError, match="mode"):
        datalake.write_table("play_by_play", pbp(1), mode="merge")


def test_overwrite_only_touches_its_own_table():
    datalake.write_table("play_by_play", pbp())
    datalake.write_table("game_odds", pd.DataFrame({"game_id": ["0021900001"], "decimal_home": [1.5]}))
    datalake.write_table("game_odds", pd.DataFrame({"game_id": ["0021900002"], "decimal_home": [2.0]}))
    assert len(datalake.read_pandas("play_by_play")) == 9
    assert datalake.tables() == ["game_odds", "play_by_play"]


def test_zstd_compression():
    import pyarrow.parquet as pq
    base = datalake.write_table("play_by_play", pbp())
    f = next(base.rglob("*.parquet"))
    assert pq.ParquetFile(f).metadata.row_group(0).column(0).compression == "ZSTD"


def test_missing_table_and_bad_names():
    with pytest.raises(FileNotFoundError, match="no lake table"):
        datalake.dataset("play_by_play")
    with pytest.raises(ValueError, match="invalid table name"):
        datalake.table_path("../escape")


def test_dataset_is_lazy_and_countable():
    datalake.write_table("play_by_play", pbp())
    assert datalake.dataset("play_by_play").count_rows() == 9


def test_duckdb_count_parity():
    pytest.importorskip("duckdb")
    datalake.write_table("play_by_play", pbp())
    datalake.write_table("game_odds", pd.DataFrame({"game_id": ["0021900001", "0021900002"], "decimal_home": [1.5, 2.5]}))
    con = datalake.duck()
    assert con.execute("SELECT count(*) FROM play_by_play").fetchone()[0] == len(datalake.read_pandas("play_by_play")) == 9
    assert con.execute("SELECT count(*) FROM game_odds").fetchone()[0] == 2
    per_season = dict(con.execute("SELECT season_year, count(*) FROM play_by_play GROUP BY 1").fetchall())
    assert per_season == {"1996-97": 3, "2019-20": 3, "2025-26": 3}
    # leading zeros survive through DuckDB too
    assert "0029600001" in {r[0] for r in con.execute("SELECT game_id FROM play_by_play").fetchall()}


# ---- append refuses a different schema -----------------------------------------------------------------------------

def test_append_rejects_new_missing_and_retyped_columns():
    datalake.write_table("play_by_play", pbp())
    with pytest.raises(ValueError, match=r"new columns \['extra'\]"):
        datalake.write_table("play_by_play", pbp().assign(extra=1), mode="append")
    with pytest.raises(ValueError, match=r"missing columns \['description'\]"):
        datalake.write_table("play_by_play", pbp().drop(columns="description"), mode="append")
    with pytest.raises(ValueError, match="action_number: table has int64, data has double"):
        datalake.write_table("play_by_play", pbp().assign(action_number=1.5), mode="append")
    assert len(datalake.read_pandas("play_by_play")) == 9  # nothing was appended


def test_append_accepts_reordered_columns_and_either_string_flavour():
    schema = pa.schema([("game_id", pa.string()), ("season_year", pa.string()), ("action_number", pa.int64()),
                        ("description", pa.string()), ("score_home", pa.float64())])
    datalake.write_table("play_by_play", pa.Table.from_pandas(pbp(), schema=schema, preserve_index=False))
    datalake.write_table("play_by_play", pbp()[list(reversed(pbp().columns))], mode="append")  # pandas str, new order
    assert len(datalake.read_pandas("play_by_play")) == 18


def test_first_append_to_a_new_table_is_fine():
    datalake.write_table("play_by_play", pbp(), mode="append")
    assert len(datalake.read_pandas("play_by_play")) == 9


# ---- overwrite is atomic -------------------------------------------------------------------------------------------

def test_failed_overwrite_keeps_the_old_table_and_leaves_no_staging(monkeypatch):
    datalake.write_table("play_by_play", pbp(2))
    real = datalake.ds.write_dataset

    def write_then_die(*a, **k):
        real(*a, **k)  # the new files are on disk when the "crash" happens
        raise OSError("disk full")
    with monkeypatch.context() as m:
        m.setattr(datalake.ds, "write_dataset", write_then_die)
        with pytest.raises(OSError, match="disk full"):
            datalake.write_table("play_by_play", pbp(1))
    assert len(datalake.read_pandas("play_by_play")) == 6
    assert not [p for p in datalake.lake_root().iterdir() if p.name.startswith(".")]


def test_overwrite_replaces_and_leaves_nothing_behind():
    datalake.write_table("play_by_play", pbp(2))
    datalake.write_table("play_by_play", pbp(1))
    assert len(datalake.read_pandas("play_by_play")) == 3
    assert [p.name for p in datalake.lake_root().iterdir()] == ["play_by_play"]


def test_replace_dir_restores_the_old_directory_if_the_move_fails(tmp_path):
    dst = tmp_path / "t"
    dst.mkdir()
    (dst / "keep.txt").write_text("x")
    with pytest.raises(FileNotFoundError):
        datalake.replace_dir(tmp_path / "missing_src", dst)
    assert (dst / "keep.txt").read_text() == "x" and [p.name for p in tmp_path.iterdir()] == ["t"]


# ---- Spark helpers (no JVM: a fake pyspark stands in) --------------------------------------------------------------

class _Reader:
    def __init__(self):
        self.calls = []

    def option(self, k, v):
        self.calls.append((k, v))
        return self

    def parquet(self, path):
        self.calls.append(("parquet", path))
        return "DF"


def test_spark_read_and_table_uri_accept_a_relative_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    datalake.write_table("game_odds", pd.DataFrame({"game_id": ["0021900001"]}), root=Path("rel_lake"))
    uri = datalake.table_uri("game_odds", Path("rel_lake"))
    assert uri == (tmp_path / "rel_lake" / "game_odds").as_uri() and uri.startswith("file:///")
    spark = types.SimpleNamespace(read=_Reader())
    assert datalake.spark_read(spark, "game_odds", root=Path("rel_lake")) == "DF"
    assert spark.read.calls == [("basePath", uri), ("parquet", uri)]


def _fake_pyspark(monkeypatch, active=None):
    built = []

    class Conf:
        def __init__(self, d):
            self.d = d

        def get(self, k, default=None):
            return self.d.get(k, default)

    class Session:
        def __init__(self, master, conf):
            self.sparkContext = types.SimpleNamespace(master=master, getConf=lambda: Conf(conf))

    class Builder:
        def __init__(self):
            self.conf, self.m = {}, None

        def master(self, m):
            self.m = m
            return self

        def appName(self, _):
            return self

        def config(self, k, v):
            self.conf[k] = v
            return self

        def getOrCreate(self):
            built.append((self.m, dict(self.conf)))
            return Session(self.m, self.conf)

    class SparkSession:
        builder = Builder()

        @staticmethod
        def getActiveSession():
            return active

    sql = types.ModuleType("pyspark.sql")
    sql.SparkSession = SparkSession
    pkg = types.ModuleType("pyspark")
    pkg.sql = sql
    monkeypatch.setitem(sys.modules, "pyspark", pkg)
    monkeypatch.setitem(sys.modules, "pyspark.sql", sql)
    return built, Session


def _java_home(tmp_path, monkeypatch):
    home = tmp_path / "jdk"
    (home / "bin").mkdir(parents=True)
    (home / "bin" / "java").write_text("")
    monkeypatch.setenv("JAVA_HOME", str(home))


def test_get_spark_rejects_a_stale_java_home_even_if_java_is_on_path(tmp_path, monkeypatch):
    _fake_pyspark(monkeypatch)
    monkeypatch.setenv("JAVA_HOME", str(tmp_path / "removed-jdk"))
    monkeypatch.setattr(datalake.shutil, "which", lambda _: "/usr/bin/java")
    with pytest.raises(RuntimeError, match="removed-jdk"):
        datalake.get_spark()


def test_get_spark_builds_local_mode_with_requested_or_env_defaults(tmp_path, monkeypatch):
    built, _ = _fake_pyspark(monkeypatch)
    _java_home(tmp_path, monkeypatch)
    datalake.get_spark(memory="2g", cores=2)
    assert built[-1][0] == "local[2]" and built[-1][1]["spark.driver.memory"] == "2g"
    assert built[-1][1]["spark.sql.sources.partitionColumnTypeInference.enabled"] == "false"
    monkeypatch.setenv("NBA_SPARK_MEMORY", "3g")
    monkeypatch.setenv("NBA_SPARK_CORES", "3")
    datalake.get_spark()
    assert built[-1][0] == "local[3]" and built[-1][1]["spark.driver.memory"] == "3g"


def test_get_spark_warns_when_an_active_session_ignores_the_request(tmp_path, monkeypatch):
    _java_home(tmp_path, monkeypatch)
    _, Session = _fake_pyspark(monkeypatch)
    active = Session("local[8]", {"spark.driver.memory": "1g"})
    built, _ = _fake_pyspark(monkeypatch, active=active)
    with pytest.warns(UserWarning, match="NOT applied"):
        assert datalake.get_spark(memory="2g", cores=2) is active
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no request, or the same request, means no warning
        assert datalake.get_spark() is active
        assert datalake.get_spark(memory="1g", cores=8) is active
    assert built == []  # never builds a second session


# ---- verification gate ---------------------------------------------------------------------------------------------

def _report(tables, age=0):
    import json
    import os
    import time
    path = datalake.converter_report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"tables": tables}))
    t = time.time() + 60 - age
    os.utime(path, (t, t))


def test_verification_status_reasons():
    datalake.write_table("game_odds", pd.DataFrame({"game_id": ["0021900001"], "v": [1]}))
    assert datalake.verification_status("game_odds") == (False, "no converter report; run tools/sqlite_to_parquet.py")
    _report({})
    assert "no entry" in datalake.verification_status("game_odds")[1]
    _report({"game_odds": {"convert": {}}})
    assert "not verified" in datalake.verification_status("game_odds")[1]
    _report({"game_odds": {"verify": {"ok": False}}})
    assert "verification failed" in datalake.verification_status("game_odds")[1]
    _report({"game_odds": {"error": "boom", "verify": {"ok": True}}})
    assert "conversion failed" in datalake.verification_status("game_odds")[1]
    _report({"game_odds": {"verify": {"ok": True}}})
    assert datalake.verification_status("game_odds") == (True, "")
    _report({"game_odds": {"verify": {"ok": True}}}, age=10_000)
    assert "written to after" in datalake.verification_status("game_odds")[1]


def test_prefer_lake_policy():
    assert datalake.prefer_lake("game_odds", csv_exists=True) == (False, "")  # no lake table at all
    datalake.write_table("game_odds", pd.DataFrame({"game_id": ["0021900001"], "v": [1]}))
    use, msg = datalake.prefer_lake("game_odds", csv_exists=True)
    assert use is False and "reading the CSV" in msg
    use, msg = datalake.prefer_lake("game_odds", csv_exists=False)
    assert use is True and "no CSV fallback" in msg
    _report({"game_odds": {"verify": {"ok": True}}})
    assert datalake.prefer_lake("game_odds", csv_exists=True) == (True, "")


def test_windows_without_winutils_is_detected(monkeypatch, tmp_path):
    monkeypatch.delenv("HADOOP_HOME", raising=False)
    monkeypatch.delenv("hadoop.home.dir", raising=False)
    monkeypatch.setattr(datalake.sys, "platform", "linux")
    assert not datalake._windows_hadoop_missing()  # Linux/macOS need nothing
    monkeypatch.setattr(datalake.sys, "platform", "win32")
    assert datalake._windows_hadoop_missing()
    monkeypatch.setenv("HADOOP_HOME", str(tmp_path))
    assert datalake._windows_hadoop_missing()  # folder set but no bin/winutils.exe
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "winutils.exe").write_bytes(b"")
    assert not datalake._windows_hadoop_missing()
