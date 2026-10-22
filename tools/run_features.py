"""Rebuild the feature table, or build point-in-time rows for one day's unplayed slate.

    python tools/run_features.py                                  # full rebuild -> data/processed/df_model_3.csv
    python tools/run_features.py --slate 2026-10-21 --out x.csv   # pregame rows for that date's games (no writes)
"""
import argparse
import io
import sys, time, os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
DATA = Path(os.environ.get("NBA_DATA_DIR", os.path.join(ROOT, "data")))
from ingest.dataPrep.config import get_config
from ingest.dataPrep.steps import (
    aggregate_player_to_team as s3a, calculate_elo_and_rest as s2, engineer_advanced_features as s4,
    calculate_travel_fatigue as s5, engineer_rolling_features as s6, finalize_feature_selection as s6a,
)

# Same-day absences come from the game's own box score (official inactives, published ~30-90 min before tip). A 9:30am
# slate cannot see them, so slate rows leave them NaN rather than guessing. prev_game_minutes_out is the PREVIOUS
# game's list and is reproduced exactly.
SAME_DAY_UNKNOWN = ["pregame_missing_top1", "pregame_missing_top2", "pregame_minutes_out", "pregame_usage_out"]


def run_features(cfg=None, with_team_stats=True, inputs=None):
    """Steps 3a (team/opp sums), 2, 4, 5, 6, 6a. Step 1/3 (downloads, need nba_api) are not run.

    `inputs` seeds the in-memory chain (games_with_ids, player_boxscores, team_opp_stats) instead of the CSVs."""
    cfg = cfg or get_config(); out = dict(inputs or {})
    t = time.time()
    steps = [("2", s2), ("4", s4), ("5", s5), ("6", s6), ("6a", s6a)]
    if with_team_stats:
        steps.insert(0, ("3a", s3a))
    for name, mod in steps:
        out.update(mod.run(cfg, out)); print(f"step {name} done {time.time()-t:.0f}s", flush=True)
    return out


def _gid(s):
    return pd.to_numeric(s.astype(str).str.replace(".0", "", regex=False), errors="coerce").astype("Int64")


def slate_rows(date, schedule=None, data_dir=DATA):
    """Pregame feature rows for `date`'s games, from data strictly before `date`.

    schedule: DataFrame GAME_ID, home, away (NBA tricodes); default = that date's games in the processed games table
    (past dates). Every game before `date` is kept as ingested; `date`'s games get no score, no line and a one-row
    box score stub per team, so the unchanged step code computes their lagged features exactly as in a full build
    (the stub's own values are never read: every feature is shifted one game). Nothing is written to disk."""
    D = pd.Timestamp(date).normalize()
    games = pd.read_csv(data_dir / "processed" / "nba_games_with_game_id.csv", parse_dates=["date"])
    gdate = games.set_index(_gid(games["GAME_ID"]))["date"]
    if schedule is None:
        schedule = games.loc[games["date"] == D, ["GAME_ID", "home", "away"]]
    schedule = schedule.assign(GAME_ID=_gid(schedule["GAME_ID"])).reset_index(drop=True)
    if schedule.empty:
        return pd.DataFrame()
    past = games[games["date"] < D]
    last = past.loc[past["date"].idxmax()]
    # season = ending year, like the games table; a long gap means a new season (date rule fails for the 2020 bubble)
    season = last["season"] if (D - last["date"]).days < 120 else D.year + (D.month >= 7)
    slate = schedule.assign(date=D, season=season, regular=True, score_home=np.nan, score_away=np.nan,
                            spread=np.nan, whos_favored=np.nan, id_spread=1)  # id_spread only gates step 2's filter
    games = pd.concat([past, slate[[c for c in games.columns if c in slate.columns]]], ignore_index=True)

    box = pd.read_csv(data_dir / "processed" / "all_boxscores.csv", low_memory=False)
    box = box[_gid(box["GAME_ID"]).map(gdate) < D]
    team_ids = box.drop_duplicates("team", keep="last").set_index("team")["TEAM_ID"]
    unknown = set(slate["home"]) | set(slate["away"])
    if unknown - set(team_ids.index):
        raise ValueError(f"teams with no box score history: {sorted(unknown - set(team_ids.index))}")
    stub = pd.concat([slate[["GAME_ID", "home"]].rename(columns={"home": "team"}),
                      slate[["GAME_ID", "away"]].rename(columns={"away": "team"})], ignore_index=True)
    stub = stub.assign(GAME_ID=stub["GAME_ID"].astype("int64"), TEAM_ID=stub["team"].map(team_ids),
                       PLAYER_ID=-1 - np.arange(len(stub)), minutes="1:00")  # "played", so the team-game row exists
    box = pd.concat([box, stub], ignore_index=True)

    opp = pd.read_csv(data_dir / "processed" / "team opp stats.csv", dtype={"GAME_ID": str})
    opp = opp[_gid(opp["GAME_ID"]).map(gdate) < D]
    opp = pd.concat([opp, stub[["GAME_ID", "team"]].assign(GAME_ID=stub["GAME_ID"].astype(str).str.zfill(10))],
                    ignore_index=True)

    cfg = get_config({"cache": {"enabled": False},
                      "steps": {"step5": {"paths": {"arena_locations": str(data_dir / "raw" / "arena_locations.csv")}}}})
    out = run_features(cfg, with_team_stats=False,
                       inputs={"games_with_ids": games, "player_boxscores": box, "team_opp_stats": opp})
    df = out["df_model_3"]
    df = df[pd.to_datetime(df["date_x"]).dt.normalize() == D]
    # The model trains on df_model_3.csv as parsed back from CSV, which can sit 1 ulp from the in-memory float.
    # LightGBM thresholds sit on observed values, so that ulp moved predictions by up to 0.2 points: round-trip too.
    df = pd.read_csv(io.StringIO(df.to_csv(index=False)), parse_dates=["date_x"])
    df["GAME_ID"] = _gid(df["GAME_ID"])
    df[SAME_DAY_UNKNOWN] = np.nan
    df = df.drop(columns=[c for c in ("home", "away") if c in df]).merge(slate[["GAME_ID", "home", "away"]], on="GAME_ID")
    missing = set(slate["GAME_ID"]) - set(df["GAME_ID"])
    if missing:
        raise RuntimeError(f"slate games dropped by the feature steps: {sorted(missing)}")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--slate", help="YYYY-MM-DD: build that date's pregame rows instead of the full table")
    ap.add_argument("--out", help="CSV path for --slate rows (default: print a summary)")
    a = ap.parse_args()
    if a.slate:
        rows = slate_rows(a.slate)
        if a.out:
            rows.to_csv(a.out, index=False)
        print(f"{len(rows)} slate rows for {a.slate}", rows[["GAME_ID", "home", "away", "elo_diff"]].to_string(), sep="\n")
    else:
        cfg = get_config(); cfg["cache"]["read_on_hit"] = False
        run_features(cfg)
