"""Backfill box scores for games the ingestion skipped, from data/csv/player_boxscores.csv.

all_boxscores.csv has no rows for 176 games that exist in the games table: 87 where |margin| == the spread (the old
push filter dropped them before the box scores were fetched; 34 of them were not pushes at all) and 89 early
2019-20 games. data/csv/player_boxscores.csv has every one of them. On the 8,439 player-games the two sources share,
points, attempts, turnovers, rebounds, minutes and team codes agree 100%.

What the backfilled rows are: players who played, with core box score columns only. They have no DNP/inactive rows
(so the pregame absence features read 0 for these games) and no advanced / tracking columns (team_boxscores_v3.csv has
no rows for them; step 3a keeps those metrics NaN instead of 0).

Idempotent: only games missing from all_boxscores.csv are added; a second run does nothing.

    python tools/backfill_boxscores.py [--dry-run]
"""
import argparse
import os
import shutil
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
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


def _mmss(minutes):
    total = float(minutes)
    m, s = int(total), int(round((total - int(total)) * 60))
    if s == 60:
        m, s = m + 1, 0
    return f"{m}:{s:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    games = pd.read_csv(GAMES, dtype={"GAME_ID": str})
    games["GAME_ID"] = games["GAME_ID"].str.zfill(10)
    box = pd.read_csv(BOX, low_memory=False)
    have = set(box["GAME_ID"].astype(str).str.replace(".0", "", regex=False).str.zfill(10))
    missing = games[~games["GAME_ID"].isin(have)]
    print(f"games without box scores: {len(missing)}")
    if missing.empty:
        return

    src = pd.read_csv(SRC, dtype={"game_id": str})
    src["GAME_ID"] = src["game_id"].str.zfill(10)
    src = src[src["GAME_ID"].isin(set(missing["GAME_ID"]))].drop_duplicates(["GAME_ID", "player_id"]).copy()

    # both teams of every game must match the games table, otherwise skip that game
    teams = missing.set_index("GAME_ID")[["home", "away"]]
    ok = src.groupby("GAME_ID")["team_abbreviation"].apply(
        lambda t: set(t.str.upper()) == set(teams.loc[t.name].str.upper()))
    bad = list(ok[~ok].index)
    src = src[~src["GAME_ID"].isin(bad)]
    print(f"available in {SRC.name}: {src['GAME_ID'].nunique()} games ({len(src):,} player rows); "
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
