"""SQLite -> Parquet converter (tools/sqlite_to_parquet.py) on tiny synthetic databases: no real data, no DuckDB."""
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import sqlite_to_parquet as s2p  # noqa: E402
from src.ingest import datalake  # noqa: E402


@pytest.fixture(autouse=True)
def lake_env(tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path / "data"))
    return tmp_path


def make_db(path, table, columns, rows):
    """columns: [(name, declared_type)], rows: list of tuples."""
    con = sqlite3.connect(path)
    con.execute(f'CREATE TABLE "{table}" ({", ".join(f"{n} {t}" for n, t in columns)})')
    con.executemany(f'INSERT INTO "{table}" VALUES ({", ".join("?" * len(columns))})', rows)
    con.commit()
    con.close()
    return Path(path)


PBP_COLS = [("game_id", "TEXT"), ("action_number", "NUMERIC"), ("description", "TEXT"), ("shot_distance", "NUMERIC")]


def pbp_rows(n):
    ids = ["0029600001", "0021900001", "0022500001", "0029900001", "0020000001"]
    return [(ids[i % len(ids)], i, f"play {i}" if i % 4 else None, (i * 1.5) if i % 3 else None) for i in range(n)]


def games_index(path, extra=()):
    rows = [("0029600001", "1996-97"), ("0021900001", "2019-20"), ("0022500001", "2025-26"),
            ("0029900001", "1999-00"), ("0020000001", "2000-01")] + list(extra)
    return make_db(path, "games_index", [("game_id", "TEXT"), ("season_year", "TEXT")], rows)


# ---- season derivation -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("gid,label", [("0029600001", "1996-97"), ("0029900001", "1999-00"), ("0020000001", "2000-01"),
                                      ("0021900001", "2019-20"), ("0022500001", "2025-26"),
                                      ("0042000101", "2020-21"), ("0011900001", "2019-20")])
def test_season_label_from_game_id(gid, label):
    assert s2p.season_from_game_id(gid) == label


@pytest.mark.parametrize("bad", ["20000001", "002000000X", "", "00200000011"])
def test_season_label_rejects_bad_ids(bad):
    with pytest.raises(ValueError):
        s2p.season_from_game_id(bad)


def test_pbp_season_year_is_derived_and_partitioned(tmp_path):
    db = make_db(tmp_path / "pbp.sqlite", "play_by_play", PBP_COLS, pbp_rows(40))
    s2p.convert(db, "play_by_play", log=lambda *_: None)
    base = datalake.table_path("play_by_play")
    assert sorted(p.name for p in base.iterdir()) == [f"season_year={s}" for s in
                                                      ["1996-97", "1999-00", "2000-01", "2019-20", "2025-26"]]
    back = datalake.read_pandas("play_by_play")
    assert (back["game_id"].map(s2p.season_from_game_id) == back["season_year"]).all()
    assert s2p.verify(db, "play_by_play", log=lambda *_: None).ok


def test_check_seasons_matches_and_catches_a_wrong_index(tmp_path):
    db = make_db(tmp_path / "pbp.sqlite", "play_by_play", PBP_COLS, pbp_rows(40))
    s2p.convert(db, "play_by_play", log=lambda *_: None)
    good = s2p.check_seasons(games_index(tmp_path / "gi.sqlite"), log=lambda *_: None)
    assert good.ok and good.games_index_mismatches == 0 and good.tables["play_by_play"]["mismatches"] == 0
    games_index(tmp_path / "gi2.sqlite")
    con = sqlite3.connect(tmp_path / "gi2.sqlite")
    con.execute("UPDATE games_index SET season_year='2018-19' WHERE game_id='0021900001'")
    con.commit()
    con.close()
    bad = s2p.check_seasons(tmp_path / "gi2.sqlite", log=lambda *_: None)
    assert not bad.ok and bad.games_index_mismatches == 1 and bad.tables["play_by_play"]["mismatches"] == 1
    assert bad.games_index_examples[0] == ["0021900001", "2018-19", "2019-20"]


def test_check_seasons_reports_games_missing_from_the_index(tmp_path):
    db = make_db(tmp_path / "pbp.sqlite", "play_by_play", PBP_COLS, pbp_rows(40))
    s2p.convert(db, "play_by_play", log=lambda *_: None)
    gi = make_db(tmp_path / "gi.sqlite", "games_index", [("game_id", "TEXT"), ("season_year", "TEXT")],
                 [("0021900001", "2019-20")])
    rep = s2p.check_seasons(gi, log=lambda *_: None)
    assert rep.ok and rep.tables["play_by_play"]["unmatched"] == 4


# ---- chunk boundaries --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n,chunk", [(25, 7), (21, 7), (7, 7), (1, 7), (25, 1), (25, 100), (25, 24), (25, 25)])
def test_chunk_boundaries_keep_every_row(tmp_path, n, chunk):
    db = make_db(tmp_path / "pbp.sqlite", "play_by_play", PBP_COLS, pbp_rows(n))
    res = s2p.convert(db, "play_by_play", chunk_rows=chunk, log=lambda *_: None)
    assert res.rows == n and res.chunks == -(-n // chunk)
    back = datalake.read_pandas("play_by_play").sort_values("action_number").reset_index(drop=True)
    assert back["action_number"].tolist() == list(range(n))
    assert s2p.verify(db, "play_by_play", log=lambda *_: None).ok


def test_max_cells_caps_the_chunk_size(tmp_path):
    cols = [("game_id", "TEXT")] + [(f"c{i}", "NUMERIC") for i in range(9)]
    db = make_db(tmp_path / "wide.sqlite", "player_boxscores", cols + [("season_year", "TEXT")],
                 [("0021900001",) + tuple(range(9)) + ("2019-20",) for _ in range(30)])
    res = s2p.convert(db, "player_boxscores", chunk_rows=1000, max_cells=55, log=lambda *_: None)  # 11 cols -> 5 rows
    assert res.chunk_rows == 5 and res.chunks == 6 and res.rows == 30


# ---- leading zeros and game ids ----------------------------------------------------------------------------------

def test_leading_zeros_survive(tmp_path):
    db = make_db(tmp_path / "x.sqlite", "game_odds", [("game_id", "TEXT"), ("decimal_home", "NUMERIC")],
                 [("0020000001", 1.5), ("0020000002", 2.5)])
    s2p.convert(db, "game_odds", log=lambda *_: None)
    assert sorted(datalake.read_pandas("game_odds")["game_id"]) == ["0020000001", "0020000002"]


def test_numeric_game_id_is_refused(tmp_path):
    db = make_db(tmp_path / "x.sqlite", "game_odds", [("game_id", "NUMERIC"), ("decimal_home", "NUMERIC")],
                 [(20000001, 1.5)])
    with pytest.raises(s2p.ConversionError, match="game_id is stored as a number"):
        s2p.convert(db, "game_odds", log=lambda *_: None)


def test_short_game_id_is_refused_and_nothing_is_written(tmp_path):
    db = make_db(tmp_path / "x.sqlite", "game_odds", [("game_id", "TEXT"), ("decimal_home", "NUMERIC")],
                 [("0020000001", 1.5), ("20000002", 2.5)])
    with pytest.raises(s2p.ConversionError, match="10-digit"):
        s2p.convert(db, "game_odds", log=lambda *_: None)
    assert not datalake.table_path("game_odds").exists()


# ---- column typing -----------------------------------------------------------------------------------------------

def test_mixed_int_and_real_become_float64_and_int_only_stays_int64_with_nulls(tmp_path):
    cols = [("game_id", "TEXT"), ("min", "NUMERIC"), ("pts", "NUMERIC"), ("empty_num", "NUMERIC"),
            ("empty_txt", "TEXT")]
    db = make_db(tmp_path / "x.sqlite", "game_odds", cols,
                 [("0020000001", 12, 10, None, None), ("0020000002", 7.5, None, None, None),
                  ("0020000003", 3, 4, None, None)])
    s2p.convert(db, "game_odds", log=lambda *_: None)
    schema = datalake.dataset("game_odds").schema
    assert schema.field("min").type == pa.float64()
    assert schema.field("pts").type == pa.int64()  # nulls do not turn an int column into float
    assert schema.field("empty_num").type == pa.float64()
    assert schema.field("empty_txt").type == pa.string()
    assert s2p.verify(db, "game_odds", log=lambda *_: None).ok


def test_text_mixed_with_numbers_fails_and_names_the_column(tmp_path):
    db = make_db(tmp_path / "x.sqlite", "game_odds",
                 [("game_id", "TEXT"), ("good", "NUMERIC"), ("messy", "NUMERIC")],
                 [("0020000001", 1, 5), ("0020000002", 2, "n/a"), ("0020000003", 3, 7.5)])
    with pytest.raises(s2p.ConversionError, match=r"game_odds\.messy: mixes text"):
        s2p.convert(db, "game_odds", log=lambda *_: None)
    assert not datalake.table_path("game_odds").exists()
    assert not [p for p in datalake.lake_root().glob(".staging-*")]


def test_blob_column_is_refused(tmp_path):
    db = make_db(tmp_path / "x.sqlite", "game_odds", [("game_id", "TEXT"), ("b", "BLOB")], [("0020000001", b"\x00")])
    with pytest.raises(s2p.ConversionError, match=r"game_odds\.b: holds 1 blob"):
        s2p.convert(db, "game_odds", log=lambda *_: None)


# ---- verify catches tampering ------------------------------------------------------------------------------------

def convert_small(tmp_path):
    db = make_db(tmp_path / "x.sqlite", "game_odds",
                 [("game_id", "TEXT"), ("odds_date", "TEXT"), ("decimal_home", "NUMERIC"), ("n", "NUMERIC")],
                 [(f"00200000{i:02d}", f"2020-01-{i + 1:02d}", 1.5 + i, i if i % 2 else None) for i in range(10)])
    s2p.convert(db, "game_odds", log=lambda *_: None)
    assert s2p.verify(db, "game_odds", log=lambda *_: None).ok
    return db, next(datalake.table_path("game_odds").glob("*.parquet"))


def rewrite(path, fn):
    pq.write_table(fn(pq.read_table(path)), path)


def test_verify_catches_a_changed_number(tmp_path):
    db, f = convert_small(tmp_path)

    def bump(t):
        vals = t.column("decimal_home").to_pylist()
        vals[3] += 0.01
        return t.set_column(t.schema.get_field_index("decimal_home"), "decimal_home", pa.array(vals, pa.float64()))
    rewrite(f, bump)
    rep = s2p.verify(db, "game_odds", log=lambda *_: None)
    assert not rep.ok and any("game_odds.decimal_home: sum" in m for m in rep.failures)


def test_verify_catches_a_dropped_row(tmp_path):
    db, f = convert_small(tmp_path)
    rewrite(f, lambda t: t.slice(0, t.num_rows - 1))
    rep = s2p.verify(db, "game_odds", log=lambda *_: None)
    assert not rep.ok and any(m.startswith("row count: sqlite=10 lake=9") for m in rep.failures)


def test_verify_catches_a_nulled_value_and_a_trimmed_string(tmp_path):
    db, f = convert_small(tmp_path)

    def tamper(t):
        n = t.column("n").to_pylist()
        n[1] = None
        d = t.column("odds_date").to_pylist()
        d[2] = d[2][:-1]
        t = t.set_column(t.schema.get_field_index("n"), "n", pa.array(n, pa.int64()))
        return t.set_column(t.schema.get_field_index("odds_date"), "odds_date", pa.array(d, pa.string()))
    rewrite(f, tamper)
    msgs = s2p.verify(db, "game_odds", log=lambda *_: None).failures
    assert any(m.startswith("game_odds.n: non-null count") for m in msgs)
    assert any(m.startswith("game_odds.odds_date: text length sum") for m in msgs)


def test_verify_catches_a_dropped_column(tmp_path):
    db, f = convert_small(tmp_path)
    rewrite(f, lambda t: t.drop_columns(["n"]))
    rep = s2p.verify(db, "game_odds", log=lambda *_: None)
    assert not rep.ok and any("columns missing from the lake" in m and "'n'" in m for m in rep.failures)


# ---- safety ------------------------------------------------------------------------------------------------------

def test_source_sqlite_is_never_modified(tmp_path):
    db = make_db(tmp_path / "pbp.sqlite", "play_by_play", PBP_COLS, pbp_rows(30))
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    s2p.convert(db, "play_by_play", chunk_rows=4, log=lambda *_: None)
    s2p.verify(db, "play_by_play", log=lambda *_: None)
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
    assert not list(tmp_path.glob("pbp.sqlite-*"))  # no journal / wal left behind


def test_existing_lake_table_needs_overwrite_and_failed_rerun_keeps_it(tmp_path):
    db = make_db(tmp_path / "x.sqlite", "game_odds", [("game_id", "TEXT"), ("v", "NUMERIC")],
                 [("0020000001", 1), ("0020000002", 2)])
    s2p.convert(db, "game_odds", log=lambda *_: None)
    with pytest.raises(s2p.ConversionError, match="already exists"):
        s2p.convert(db, "game_odds", log=lambda *_: None)
    con = sqlite3.connect(db)
    con.execute("INSERT INTO game_odds VALUES ('0020000003', 'oops')")  # now mixed
    con.commit()
    con.close()
    with pytest.raises(s2p.ConversionError, match="mixes text"):
        s2p.convert(db, "game_odds", overwrite=True, log=lambda *_: None)
    assert len(datalake.read_pandas("game_odds")) == 2  # the good lake table survived the failed rerun
    con = sqlite3.connect(db)
    con.execute("DELETE FROM game_odds WHERE v='oops'")
    con.execute("INSERT INTO game_odds VALUES ('0020000003', 3)")
    con.commit()
    con.close()
    s2p.convert(db, "game_odds", overwrite=True, log=lambda *_: None)
    assert len(datalake.read_pandas("game_odds")) == 3
    assert datalake.tables() == ["game_odds"]


def test_missing_table_and_missing_file(tmp_path):
    db = make_db(tmp_path / "x.sqlite", "game_odds", [("game_id", "TEXT")], [("0020000001",)])
    with pytest.raises(s2p.ConversionError, match="not found"):
        s2p.convert(db, "team_boxscores", log=lambda *_: None)
    with pytest.raises(s2p.ConversionError, match="not found"):
        s2p.convert(tmp_path / "nope.sqlite", "game_odds", log=lambda *_: None)


# ---- CLI ---------------------------------------------------------------------------------------------------------

def test_cli_converts_verifies_and_writes_a_report(tmp_path, capsys):
    sql = tmp_path / "data" / "sql"
    sql.mkdir(parents=True)
    make_db(sql / s2p.PBP_DB, "play_by_play", PBP_COLS, pbp_rows(40))
    con = sqlite3.connect(sql / s2p.MAIN_DB)
    con.execute("CREATE TABLE games_index (game_id TEXT, season_year TEXT)")
    con.executemany("INSERT INTO games_index VALUES (?, ?)", [("0029600001", "1996-97"), ("0021900001", "2019-20"),
                                                             ("0022500001", "2025-26"), ("0029900001", "1999-00"),
                                                             ("0020000001", "2000-01")])
    con.commit()
    con.close()
    assert s2p.main(["--tables", "games_index", "play_by_play", "--chunk-rows", "9"]) == 0
    out = capsys.readouterr().out
    assert "OK" in out and "lake size" in out
    report = json.loads((datalake.lake_root() / "_reports" / "sqlite_to_parquet.json").read_text())
    assert report["ok"] and report["tables"]["play_by_play"]["convert"]["rows"] == 40
    assert report["seasons"]["ok"] and report["lake_bytes"]["TOTAL"] > 0
    assert datalake.tables() == ["games_index", "play_by_play"]  # _reports is not a table
    assert s2p.main(["--tables", "play_by_play"]) == 1  # existing lake table, no --overwrite
    assert s2p.main(["--tables", "play_by_play", "--verify-only"]) == 0


def test_verify_is_exact_for_integer_sums_above_2_pow_53(tmp_path):
    n = 5_000
    rows = [(f"002190{i % 9000 + 1:04d}", 2_000_000_000_000 + i * 7 + 1) for i in range(n)]
    assert sum(r[1] for r in rows) > 2 ** 53
    db = make_db(tmp_path / "x.sqlite", "game_odds", [("game_id", "TEXT"), ("big", "NUMERIC")], rows)
    s2p.convert(db, "game_odds", chunk_rows=700, log=lambda *_: None)
    assert s2p.verify(db, "game_odds", log=lambda *_: None).ok
    f = next(datalake.table_path("game_odds").glob("*.parquet"))

    def bump(t):
        vals = t.column("big").to_pylist()
        vals[0] += 1
        return t.set_column(t.schema.get_field_index("big"), "big", pa.array(vals, pa.int64()))
    rewrite(f, bump)
    assert any(m.startswith("game_odds.big: sum") for m in s2p.verify(db, "game_odds", log=lambda *_: None).failures)


# ---- CLI resilience, report merging, season range ------------------------------------------------------------------

GI_ROWS = [("0029600001", "1996-97"), ("0021900001", "2019-20"), ("0022500001", "2025-26"),
           ("0029900001", "1999-00"), ("0020000001", "2000-01")]


def cli_sql_dir(tmp_path, gi_extra=()):
    sql = tmp_path / "data" / "sql"
    sql.mkdir(parents=True, exist_ok=True)
    make_db(sql / s2p.PBP_DB, "play_by_play", PBP_COLS, pbp_rows(40))
    con = sqlite3.connect(sql / s2p.MAIN_DB)
    con.execute("CREATE TABLE games_index (game_id TEXT, season_year TEXT)")
    con.executemany("INSERT INTO games_index VALUES (?, ?)", GI_ROWS + list(gi_extra))
    con.execute("CREATE TABLE game_odds (game_id TEXT, decimal_home NUMERIC)")
    con.executemany("INSERT INTO game_odds VALUES (?, ?)", [("0021900001", 1.5), ("0021900002", 2.5)])
    con.commit()
    con.close()
    return sql


def read_report():
    return json.loads((datalake.lake_root() / "_reports" / "sqlite_to_parquet.json").read_text())


def test_cli_keeps_going_after_an_os_error_and_still_writes_the_report(tmp_path, monkeypatch, capsys):
    cli_sql_dir(tmp_path)
    real = s2p.convert

    def flaky(src, table, **kw):
        if table == "games_index":
            raise PermissionError("file in use by another process")
        return real(src, table, **kw)
    monkeypatch.setattr(s2p, "convert", flaky)
    assert s2p.main(["--tables", "games_index", "game_odds", "play_by_play"]) == 1
    rep = read_report()
    assert "PermissionError" in rep["tables"]["games_index"]["error"]
    assert rep["tables"]["game_odds"]["verify"]["ok"] and rep["tables"]["play_by_play"]["verify"]["ok"]
    assert rep["ok"] is False
    assert datalake.tables() == ["game_odds", "play_by_play"]


def test_cli_survives_a_corrupt_sqlite_file(tmp_path, capsys):
    sql = cli_sql_dir(tmp_path)
    (sql / s2p.PBP_DB).write_bytes(b"this is not a database" * 100)
    assert s2p.main(["--tables", "game_odds", "play_by_play"]) == 1
    rep = read_report()
    assert "DatabaseError" in rep["tables"]["play_by_play"]["error"]
    assert rep["tables"]["game_odds"]["verify"]["ok"]


def test_season_check_counts_an_unplaceable_games_index_id_instead_of_crashing(tmp_path, capsys):
    cli_sql_dir(tmp_path, gi_extra=[("0028500001", "1985-86")])
    assert s2p.main(["--tables", "play_by_play"]) == 1
    seasons = read_report()["seasons"]
    assert seasons["ok"] is False and seasons["games_index_mismatches"] == 1
    assert seasons["games_index_examples"] == [["0028500001", "1985-86", "(unplaceable id)"]]


def test_report_merges_tables_and_clears_stale_verification(tmp_path, capsys):
    cli_sql_dir(tmp_path)
    assert s2p.main(["--tables", "game_odds"]) == 0
    assert s2p.main(["--tables", "games_index"]) == 0
    rep = read_report()
    assert set(rep["tables"]) == {"game_odds", "games_index"}  # the second run kept the first run's result
    assert s2p.main(["--tables", "game_odds", "--verify-only"]) == 0
    entry = read_report()["tables"]["game_odds"]
    assert "convert" in entry and entry["verify"]["ok"]  # verify-only keeps the earlier convert record
    assert s2p.main(["--tables", "game_odds", "--skip-verify", "--overwrite"]) == 0
    entry = read_report()["tables"]["game_odds"]
    assert "convert" in entry and "verify" not in entry  # a re-convert makes the old verification stale


@pytest.mark.parametrize("gid,label", [("0029600001", "1996-97"), ("0024900001", "2049-50")])
def test_season_range_edges_are_accepted(gid, label):
    assert s2p.season_from_game_id(gid) == label


@pytest.mark.parametrize("gid", ["0025000001", "0028500001", "0029500001"])
def test_ids_outside_1996_2049_are_refused(gid):
    with pytest.raises(ValueError, match="cannot place"):
        s2p.season_from_game_id(gid)


def test_converter_refuses_unplaceable_ids_and_names_the_table(tmp_path):
    rows = pbp_rows(10) + [("0028500001", 99, "old", None)]
    db = make_db(tmp_path / "pbp.sqlite", "play_by_play", PBP_COLS, rows)
    with pytest.raises(s2p.ConversionError, match=r"play_by_play: cannot derive season_year: game id '0028500001'"):
        s2p.convert(db, "play_by_play", log=lambda *_: None)
    assert not datalake.table_path("play_by_play").exists()
