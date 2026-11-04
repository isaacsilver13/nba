"""Parquet lake layout (src/ingest/datalake.py) on tiny synthetic frames: no network, no real data, no DuckDB needed
(the DuckDB parity test skips itself when duckdb is not installed)."""
import pandas as pd
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
