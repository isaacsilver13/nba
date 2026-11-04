"""Backfill box scores for games the ingestion skipped, from the player_boxscores table (Parquet lake, or the old
data/csv/player_boxscores.csv when no lake exists; see --source).

all_boxscores.csv has no rows for 176 games that exist in the games table: 87 where |margin| == the spread (the old
push filter dropped them before the box scores were fetched; 34 of them were not pushes at all) and 89 early
2019-20 games. data/csv/player_boxscores.csv has every one of them. On the 8,439 player-games the two sources share,
points, attempts, turnovers, rebounds, minutes and team codes agree 100%.

What the backfilled rows are: players who played, with core box score columns only. They have no DNP/inactive rows
(so the pregame absence features read 0 for these games) and no advanced / tracking columns (team_boxscores_v3.csv has
no rows for them; step 3a keeps those metrics NaN instead of 0).

Idempotent: only games missing from all_boxscores.csv are added; a second run does nothing.

    python tools/backfill_boxscores.py [--dry-run] [--source auto|lake|csv]

--source auto (default) reads the lake's player_boxscores only once tools/sqlite_to_parquet.py has verified it (and it
has not been written to since); otherwise it reads the CSV, and says which. --source lake / csv force one.
Run tools/compare_csv_to_lake.py for proof that the two sources are the same data.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.ingest import datalake  # noqa: E402

DATA = Path(os.environ.get("NBA_DATA_DIR", REPO / "data"))
GAMES = DATA / "processed" / "nba_games_with_game_id_processed.csv"
BOX = DATA / "processed" / "all_boxscores.csv"
SRC = DATA / "csv" / "player_boxscores.csv"

# csv column -> all_boxscores (BoxScore V3) column
RENAME = {
    "team_id": "TEAM_ID", "team_abbreviation": "team", "team_name": "teamName", "player_id": "PLAYER_ID",
    "fgm": "fieldGoalsMade", "fga": "fieldGoalsAttempted", "fg_pct": "fieldGoalsPercentage",
    "fg3m": "threePointersMade", "fg3a": "threePointersAttempted", "fg3_pct": "threePointersPercentage",
    "ftm": "freeThrowsMade", "fta": "freeThrowsAttempted", "ft_pct": "freeThrowsPercentage",
    "oreb": "reboundsOffensive", "dreb": "reboundsDefensive", "reb": "reboundsTotal", "ast": "assists",
    "stl": "steals", "blk": "blocks", "tov": "turnovers", "pf": "foulsPersonal", "pts": "points",
    "plus_minus": "plusMinusPoints",
}


# the only player_boxscores columns this tool uses (the lake lets us read just these instead of all 207)
SRC_COLUMNS = list(dict.fromkeys(["game_id", "player_id", "team_abbreviation", "player_name", "min", *RENAME]))


def load_source(source, game_ids):
    """Source player rows (``SRC_COLUMNS``; the lake path is already limited to ``game_ids``) and a label for the log.
    ``auto`` uses the lake only when its table is verified (datalake.prefer_lake), otherwise the CSV."""
    if source not in ("auto", "lake", "csv"):
        raise ValueError(f"source must be auto, lake or csv, got {source!r}")
    use_lake = source == "lake"
    if source == "auto":
        use_lake, message = datalake.prefer_lake("player_boxscores", csv_exists=SRC.is_file())
        if message:
            print(message)
    if use_lake:
        df = datalake.read_pandas("player_boxscores", columns=SRC_COLUMNS,
                                  filters=[("game_id", "in", sorted(game_ids))])
        return df, "lake/player_boxscores"
    # round_trip: the default parser truncates floats to 15 digits, so the csv and lake sources would differ
    return pd.read_csv(SRC, dtype={"game_id": str}, float_precision="round_trip"), SRC.name


def _mmss(minutes):
    total = float(minutes)
    m, s = int(total), int(round((total - int(total)) * 60))
    if s == 60:
        m, s = m + 1, 0
    return f"{m}:{s:02d}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--source", choices=["auto", "lake", "csv"], default="auto")
    a = ap.parse_args(argv)

    games = pd.read_csv(GAMES, dtype={"GAME_ID": str})
    games["GAME_ID"] = games["GAME_ID"].str.zfill(10)
    box = pd.read_csv(BOX, low_memory=False, float_precision="round_trip")  # rewritten below: don't lose digits
    have = set(box["GAME_ID"].astype(str).str.replace(".0", "", regex=False).str.zfill(10))
    missing = games[~games["GAME_ID"].isin(have)]
    print(f"games without box scores: {len(missing)}")
    if missing.empty:
        return

    src, label = load_source(a.source, set(missing["GAME_ID"]))
    src["GAME_ID"] = src["game_id"].str.zfill(10)
    src = src[src["GAME_ID"].isin(set(missing["GAME_ID"]))].drop_duplicates(["GAME_ID", "player_id"]).copy()

    # both teams of every game must match the games table, otherwise skip that game
    teams = missing.set_index("GAME_ID")[["home", "away"]]
    ok = src.groupby("GAME_ID")["team_abbreviation"].apply(
        lambda t: set(t.str.upper()) == set(teams.loc[t.name].str.upper()))
    bad = list(ok[~ok].index)
    src = src[~src["GAME_ID"].isin(bad)]
    print(f"available in {label}: {src['GAME_ID'].nunique()} games ({len(src):,} player rows); "
          f"skipped for team-code mismatch: {len(bad)}; not in source: {len(missing) - src['GAME_ID'].nunique() - len(bad)}")
    if a.dry_run or src.empty:
        return

    rows = src.rename(columns=RENAME).copy()
    rows["GAME_ID"] = rows["GAME_ID"].astype(int)
    rows["minutes"] = src["min"].map(_mmss)
    name = src["player_name"].fillna("").str.split(" ", n=1, expand=True)
    rows["firstName"], rows["familyName"] = name[0], (name[1] if name.shape[1] > 1 else "")
    rows["comment"] = ""
    rows["endpoint_family"] = "csv_backfill"
    keep = list(RENAME.values()) + ["GAME_ID", "minutes", "firstName", "familyName", "comment", "endpoint_family"]
    rows = rows[list(dict.fromkeys(keep))]

    shutil.copyfile(BOX, BOX.with_name("all_boxscores.before_backfill.csv"))
    out = pd.concat([box, rows], ignore_index=True)
    out.to_csv(BOX, index=False)
    print(f"added {len(rows):,} rows for {rows['GAME_ID'].nunique()} games -> {len(out):,} total "
          f"(original kept as {BOX.with_name('all_boxscores.before_backfill.csv').name})")


if __name__ == "__main__":
    main()
