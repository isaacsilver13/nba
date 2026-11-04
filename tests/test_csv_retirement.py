"""Retiring data/csv/ as a source: the CSV-vs-lake comparison gate, and the tools that now read the lake
(tools/backfill_boxscores.py, tools/encompass_check.py) with the CSV as fallback. Synthetic data only."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TOOLS = str(Path(__file__).resolve().parent.parent / "tools")
sys.path.insert(0, TOOLS)
import backfill_boxscores as bf  # noqa: E402
import compare_csv_to_lake as cmp  # noqa: E402
import encompass_check as enc  # noqa: E402
from src.ingest import datalake  # noqa: E402


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "data" / "csv").mkdir(parents=True)
    return tmp_path / "data"


def odds_df():
    return pd.DataFrame({
        "game_id": ["0020000001", "0020000001", "0021900002", "0021900002"],
        "game_date": ["2020-01-01T00:00:00Z"] * 4,
        "odds_date": ["open", "close", "2020-01-01T09:00:00Z", None],
        "decimal_home": [1.5, 1.6, 2.5, 3],
        "decimal_away": [2.5, 2.4, None, 1.5],
        "moneyline_home": ["-200", "", "+150", None],
        "n": pd.array([1, 2, None, 4], dtype="Int64"),
    })


def write_pair(data_dir, name="game_odds", df=None, csv_df=None):
    df = odds_df() if df is None else df
    datalake.write_table(name, df)
    (csv_df if csv_df is not None else df).to_csv(data_dir / "csv" / f"{name}.csv", index=False)


def compare(data_dir, name="game_odds"):
    return cmp.compare_one(data_dir / "csv" / f"{name}.csv", log=lambda *_: None)


# ---- the comparison gate -----------------------------------------------------------------------------------------

def test_identical_csv_and_lake(data_dir):
    write_pair(data_dir)
    r = compare(data_dir)
    assert r.identical, r.reasons
    assert r.csv_rows == r.lake_rows == 4


def test_empty_string_in_lake_equals_empty_csv_field(data_dir):
    df = odds_df()
    datalake.write_table("game_odds", df)  # moneyline_home has '' and None
    df.to_csv(data_dir / "csv" / "game_odds.csv", index=False)  # both become an empty field
    assert compare(data_dir).identical


def test_csv_that_lost_leading_zeros_is_identical_with_a_note(data_dir):
    df = odds_df()
    write_pair(data_dir, df=df, csv_df=df.assign(game_id=df.game_id.astype("int64")))
    r = compare(data_dir)
    assert r.identical, r.reasons
    assert any("leading zeros" in n for n in r.notes)


def test_csv_rows_with_an_unquoted_comma_are_reported_as_malformed(data_dir):
    df = odds_df()
    datalake.write_table("game_odds", df)
    lines = (data_dir / "csv" / "game_odds.csv")
    df.to_csv(lines, index=False)
    text = lines.read_text().splitlines()
    text[2] = text[2].replace("2020-01-01T00:00:00Z", "2020-01-01,T00:00:00Z", 1)  # one unquoted comma: 8 fields not 7
    lines.write_text("\n".join(text) + "\n")
    r = compare(data_dir)
    assert not r.identical
    assert "CSV is malformed: 1 rows" in r.reasons[0] and "{8: 1}" in r.reasons[0]


def test_well_formed_csv_with_quoted_commas_and_newlines_is_not_malformed(data_dir):
    df = odds_df().assign(odds_date=["a, b", 'say "hi"', "line1\nline2", None])
    write_pair(data_dir, df=df)
    r = compare(data_dir)
    assert r.identical, r.reasons


def test_extra_csv_column_is_a_difference(data_dir):
    df = odds_df()
    write_pair(data_dir, df=df, csv_df=df.assign(only_in_csv=1))
    r = compare(data_dir)
    assert not r.identical and any("columns the lake lacks: ['only_in_csv']" in m for m in r.reasons)


def test_extra_lake_column_is_only_a_note(data_dir):
    df = odds_df()
    write_pair(data_dir, df=df, csv_df=df.drop(columns="n"))
    r = compare(data_dir)
    assert r.identical and any("lake has columns the CSV lacks" in n for n in r.notes)


def test_changed_number_extra_row_and_changed_text_are_differences(data_dir):
    df = odds_df()
    changed = df.copy()
    changed.loc[0, "decimal_home"] = 1.51
    write_pair(data_dir, df=df, csv_df=changed)
    assert any(m.startswith("decimal_home: sum") for m in compare(data_dir).reasons)

    write_pair(data_dir, df=df, csv_df=pd.concat([df, df.iloc[[0]]], ignore_index=True))
    r = compare(data_dir)
    assert any("row count: csv=5 lake=4 (CSV has rows the lake lacks)" in m for m in r.reasons)

    text = df.copy()
    text.loc[0, "moneyline_home"] = "-2000"
    write_pair(data_dir, df=df, csv_df=text)
    assert any(m.startswith("moneyline_home: text length sum") for m in compare(data_dir).reasons)


def test_different_game_ids_and_non_numeric_values_are_differences(data_dir):
    df = odds_df()
    other = df.copy()
    other.loc[0, "game_id"] = "0020000009"
    write_pair(data_dir, df=df, csv_df=other)
    assert any(m.startswith("distinct game_id") for m in compare(data_dir).reasons)

    junk = df.copy().astype({"decimal_home": object})
    junk.loc[0, "decimal_home"] = "n/a"
    write_pair(data_dir, df=df, csv_df=junk)
    assert any("not numbers" in m for m in compare(data_dir).reasons)


def test_csv_without_a_lake_table(data_dir):
    odds_df().to_csv(data_dir / "csv" / "kalshi.csv", index=False)
    r = compare(data_dir, "kalshi")
    assert not r.identical and r.table is None and "not in the lake at all" in r.reasons[0]


def test_cli_exit_codes_and_report(data_dir, capsys):
    write_pair(data_dir)
    assert cmp.main([]) == 0
    assert "ALL IDENTICAL" in capsys.readouterr().out
    report = json.loads((datalake.lake_root() / "_reports" / "csv_vs_lake.json").read_text())
    assert report["ok"] and report["results"][0]["table"] == "game_odds"
    df = odds_df()
    write_pair(data_dir, df=df, csv_df=df.assign(only_in_csv=1))
    assert cmp.main([]) == 1
    assert "NOT safely redundant" in capsys.readouterr().out
    assert cmp.main(["--csv-dir", str(data_dir / "nowhere")]) == 2


def test_partitioned_table_compare(data_dir):
    df = player_df()
    write_pair(data_dir, "player_boxscores", df=df)
    r = compare(data_dir, "player_boxscores")
    assert r.identical, r.reasons


# ---- backfill_boxscores: lake and CSV give the same output --------------------------------------------------------

def player_df():
    rng = np.random.default_rng(0)
    rows = []
    for gid, season, home, away in [("0021900002", "2019-20", "BOS", "NYK"), ("0021900003", "2019-20", "LAL", "DEN"),
                                    ("0021900007", "2019-20", "MIA", "CHI")]:
        for team in (home, away):
            for p in range(3):
                rows.append({
                    "season_year": season, "team_abbreviation": team, "team_name": f"{team} team", "game_id": gid,
                    "player_name": None if p == 2 else f"First{p} Last{p}", "team_id": 1610612700 + len(rows),
                    "player_id": 200000 + len(rows), "min": [31.5, 12.0, 0.25][p], "fgm": int(rng.integers(0, 10)),
                    "fga": int(rng.integers(5, 20)), "fg_pct": float(rng.random()), "fg3m": 1, "fg3a": 2,
                    "fg3_pct": 0.5, "ftm": 2, "fta": 2, "ft_pct": 1.0, "oreb": 1, "dreb": 3, "reb": 4, "ast": 2,
                    "stl": 1, "blk": 0, "tov": 1, "pf": 2, "pts": 12,
                    "plus_minus": None if p == 1 else float(rng.integers(-15, 15)),
                })
    return pd.DataFrame(rows)


def mark_verified(*tables, ok=True, error=None, age=0):
    """Write the converter report the way tools/sqlite_to_parquet.py does, newer than any lake file."""
    path = datalake.converter_report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"error": error} if error else {"verify": {"ok": ok}}
    path.write_text(json.dumps({"tables": {t: dict(entry) for t in tables}}))
    t = time.time() + 60 - age
    os.utime(path, (t, t))


def setup_backfill(tmp_path, monkeypatch, name, with_lake):
    d = tmp_path / name
    (d / "processed").mkdir(parents=True)
    (d / "csv").mkdir()
    pd.DataFrame({"GAME_ID": ["0021900001", "0021900002", "0021900003", "0021900007", "0021900009"],
                  "home": ["ATL", "BOS", "LAL", "MIA", "UTA"], "away": ["HOU", "NYK", "DEN", "BKN", "SAC"]}
                 ).to_csv(d / "processed" / "games.csv", index=False)
    pd.DataFrame({"GAME_ID": [21900001], "team": ["ATL"], "PLAYER_ID": [1], "points": [10]}
                 ).to_csv(d / "processed" / "all_boxscores.csv", index=False)
    df = player_df()
    df.to_csv(d / "csv" / "player_boxscores.csv", index=False)
    monkeypatch.setenv("NBA_DATA_DIR", str(d))
    if with_lake:
        datalake.write_table("player_boxscores", df)
    monkeypatch.setattr(bf, "GAMES", d / "processed" / "games.csv")
    monkeypatch.setattr(bf, "BOX", d / "processed" / "all_boxscores.csv")
    monkeypatch.setattr(bf, "SRC", d / "csv" / "player_boxscores.csv")
    return d / "processed" / "all_boxscores.csv"


def test_backfill_from_lake_matches_backfill_from_csv(tmp_path, monkeypatch, capsys):
    out_csv = setup_backfill(tmp_path, monkeypatch, "a", with_lake=True)
    bf.main(["--source", "csv"])
    assert "available in player_boxscores.csv: 2 games" in capsys.readouterr().out  # 0021900007 team codes mismatch
    out_lake = setup_backfill(tmp_path, monkeypatch, "b", with_lake=True)
    bf.main(["--source", "lake"])
    assert "available in lake/player_boxscores: 2 games" in capsys.readouterr().out
    a, b = out_csv.read_bytes(), out_lake.read_bytes()
    assert a == b
    out = pd.read_csv(out_lake)
    assert len(out) == 1 + 12 and set(out["endpoint_family"].dropna()) == {"csv_backfill"}
    assert out.loc[out["PLAYER_ID"] == 200000, "minutes"].item() == "31:30"


def test_backfill_second_run_adds_nothing_from_either_source(tmp_path, monkeypatch, capsys):
    out = setup_backfill(tmp_path, monkeypatch, "c", with_lake=True)
    bf.main(["--source", "lake"])
    first = out.read_bytes()
    capsys.readouterr()
    bf.main(["--source", "lake"])
    assert out.read_bytes() == first


def test_backfill_auto_uses_the_lake_only_when_it_is_verified(tmp_path, monkeypatch, capsys):
    setup_backfill(tmp_path, monkeypatch, "d", with_lake=True)
    bf.main(["--dry-run"])  # lake exists but nothing verified it
    out = capsys.readouterr().out
    assert "available in player_boxscores.csv" in out and "not using lake table 'player_boxscores'" in out
    mark_verified("player_boxscores")
    bf.main(["--dry-run"])
    out = capsys.readouterr().out
    assert "available in lake/player_boxscores" in out and "not using" not in out
    setup_backfill(tmp_path, monkeypatch, "e", with_lake=False)
    bf.main(["--dry-run"])
    assert "available in player_boxscores.csv" in capsys.readouterr().out


@pytest.mark.parametrize("report", [
    dict(ok=False),                          # verification failed
    dict(error="PermissionError: locked"),   # conversion failed
    dict(age=10_000),                        # the lake was written to after the report
])
def test_backfill_auto_falls_back_to_csv_for_a_bad_or_stale_verification(tmp_path, monkeypatch, capsys, report):
    setup_backfill(tmp_path, monkeypatch, "g", with_lake=True)
    mark_verified("player_boxscores", **report)
    bf.main(["--dry-run"])
    assert "available in player_boxscores.csv" in capsys.readouterr().out


def test_backfill_auto_uses_an_unverified_lake_with_a_warning_when_there_is_no_csv(tmp_path, monkeypatch, capsys):
    setup_backfill(tmp_path, monkeypatch, "h", with_lake=True)
    bf.SRC.unlink()
    bf.main(["--dry-run"])
    out = capsys.readouterr().out
    assert "warning: lake table 'player_boxscores' is not verified" in out and "available in lake/player_boxscores" in out


def test_backfill_lake_source_without_a_lake_fails_loudly(tmp_path, monkeypatch):
    setup_backfill(tmp_path, monkeypatch, "f", with_lake=False)
    with pytest.raises(FileNotFoundError, match="no lake table"):
        bf.main(["--source", "lake"])


def test_backfill_reads_only_the_columns_it_uses():
    assert set(bf.SRC_COLUMNS) >= set(bf.RENAME) | {"game_id", "min", "player_name"}
    assert len(bf.SRC_COLUMNS) == len(set(bf.SRC_COLUMNS))


# ---- encompass_check: odds from lake or CSV ------------------------------------------------------------------------

def test_load_odds_uses_a_verified_lake_else_csv_and_explicit_path_wins(data_dir, monkeypatch, capsys):
    monkeypatch.setattr(enc, "DATA", data_dir)  # module constant, fixed at import like the other tools
    assert enc.load_odds() == (None, None)
    csv_df = odds_df().assign(decimal_home=9.0)  # marker so we can tell the sources apart
    csv_df.to_csv(data_dir / "csv" / "game_odds.csv", index=False)
    g, label = enc.load_odds()
    assert label == "game_odds.csv" and (g["decimal_home"] == 9.0).all() and g["game_id"].iloc[0] == "0020000001"
    datalake.write_table("game_odds", odds_df())
    g, label = enc.load_odds()  # lake exists but is unverified: still the CSV
    assert label == "game_odds.csv" and "not using lake table 'game_odds'" in capsys.readouterr().out
    mark_verified("game_odds")
    g, label = enc.load_odds()
    assert label == "lake/game_odds" and g["decimal_home"].iloc[0] == 1.5 and g["game_id"].iloc[0] == "0020000001"
    g, label = enc.load_odds(data_dir / "csv" / "game_odds.csv")
    assert label == "game_odds.csv" and (g["decimal_home"] == 9.0).all()
    assert enc.load_odds(data_dir / "csv" / "missing.csv") == (None, None)


def test_load_odds_uses_an_unverified_lake_when_there_is_no_csv(data_dir, monkeypatch, capsys):
    monkeypatch.setattr(enc, "DATA", data_dir)
    datalake.write_table("game_odds", odds_df())
    g, label = enc.load_odds()
    assert label == "lake/game_odds" and "no CSV fallback" in capsys.readouterr().out
