"""Add lineup stints for ESPN-sourced games (current season) to lineup_stints.csv, from ESPN play-by-play.

    NBA_DATA_DIR=<data dir> python tools/extend_lineup_stints.py                 # games with no audit row yet
    NBA_DATA_DIR=<data dir> python tools/extend_lineup_stints.py --retry-failed  # also games the archive could not rebuild
    NBA_DATA_DIR=<data dir> python tools/extend_lineup_stints.py --validate      # compare ESPN vs archive stints, write nothing

Targets the repo's ESPN-sourced game ids (002<yy>9<nnnn>, see tools/ingest_daily.py) because stats.nba.com / cdn.nba.com
refuse this PC and the nba_data archive stops at 2025-26. Summaries come from tools/ingest_daily.summary (cached under
data/cache/espn once a game is final). Rows replace any earlier rows for the same game; games that cannot be rebuilt
are written to the audit table with a reason and excluded, as in build_lineup_stints.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
import ingest_daily as idl  # noqa: E402
from ingest.dataPrep.steps.build_lineup_stints import build_stints  # noqa: E402
from ingest.dataPrep.steps.espn_pbp import plays_to_pbp  # noqa: E402

DATA = Path(os.environ.get("NBA_DATA_DIR", ROOT / "data"))
PROC = DATA / "processed"
BOX_COLS = ["GAME_ID", "TEAM_ID", "team", "PLAYER_ID", "firstName", "familyName", "nameI", "position", "minutes"]


def espn_games(games, box):
    """ESPN-sourced games (id pattern 002<yy>9<nnnn>) with their ESPN event ids, from the scoreboard by date."""
    g = games[(games.GAME_ID % 100000) >= 90000].copy()
    g["date"] = pd.to_datetime(g["date"])
    frames = [idl.scoreboard(day.date(), final_only=True)[["espn_id", "date", "home", "away"]]
              for day in sorted(g["date"].unique())]
    sb = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["espn_id", "date", "home", "away"])
    return g.merge(sb, on=["date", "home", "away"], how="left", validate="1:1")


def athlete_map(summary, rows, team_ids):
    """ESPN athlete id -> PLAYER_ID for one game. all_boxscores.csv does not keep the ESPN id, but ingest_daily built its
    firstName/familyName from the same ESPN displayName (split at the first space), so the two sides meet on
    (team, first, last); ids at or above ESPN_ID_OFFSET are the ESPN id plus the offset by construction."""
    out = {}
    for team in summary["boxscore"]["players"]:
        tid = team_ids[idl.tricode(team["team"]["abbreviation"])]
        mine = rows[rows.TEAM_ID == tid]
        for a in team["statistics"][0]["athletes"]:
            ath = a["athlete"]
            if "id" not in ath:
                continue
            first, _, last = ath["displayName"].partition(" ")
            hit = mine[(mine.firstName == first) & (mine.familyName == last)].PLAYER_ID
            if len(hit) == 1:
                out[int(ath["id"])] = int(hit.iloc[0])
            elif int(ath["id"]) + idl.ESPN_ID_OFFSET in set(mine.PLAYER_ID):
                out[int(ath["id"])] = int(ath["id"]) + idl.ESPN_ID_OFFSET
    return out


def pbp_frame(games, box):
    """Concatenated stint-builder input for the given ESPN games; returns (pbp, list of (game_id, error))."""
    team_ids = box.drop_duplicates("team").set_index("team")["TEAM_ID"].to_dict()
    frames, skipped = [], []
    for g in games.itertuples():
        if pd.isna(g.espn_id):
            skipped.append((int(g.GAME_ID), "no ESPN event for this game"))
            continue
        rows = box[box.GAME_ID == g.GAME_ID]
        try:
            summ = idl.summary(g.espn_id)
            to_pid = athlete_map(summ, rows, team_ids)
            frames.append(plays_to_pbp(summ, int(g.GAME_ID), idl.tricode, team_ids, to_pid))
        except Exception as e:  # noqa: BLE001 - one bad summary must not stop the batch
            skipped.append((int(g.GAME_ID), f"ESPN plays unreadable: {type(e).__name__}"))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), skipped


def validate(games, box, stints):
    """Rebuild archive-built 2025-26 games from ESPN and compare per-lineup seconds and points."""
    both = games[games.GAME_ID.isin(set(stints.GAME_ID))]
    pbp, skipped = pbp_frame(both, box)
    new, audit = build_stints(pbp, box)
    key = ["GAME_ID", "TEAM_ID", "lineup_key"]
    cols = ["seconds", "pts_for", "pts_against"]
    a = stints[stints.GAME_ID.isin(set(new.GAME_ID))].groupby(key)[cols].sum()
    b = new.groupby(key)[cols].sum()
    j = a.join(b, how="outer", lsuffix="_archive", rsuffix="_espn").fillna(0)
    game_gap = j.groupby(level=0).apply(lambda d: (d.seconds_archive - d.seconds_espn).abs().sum() / 2)
    print(f"[validate] archive games with ESPN plays: {len(both)}; rebuilt from ESPN: {int(audit.ok.sum())}; "
          f"skipped {len(skipped)}")
    print(f"[validate] games rebuilt both ways: {game_gap.shape[0]}; per-game lineup-seconds disagreement "
          f"(sum |archive - ESPN| / 2): median {game_gap.median():.1f}s, 90th pct {game_gap.quantile(.9):.1f}s, "
          f"max {game_gap.max():.1f}s")
    print(f"[validate] points mismatches: {(j.pts_for_archive != j.pts_for_espn).mean():.3%} of team-lineups")
    print(audit[~audit.ok].reason.str.replace(r"'.*'|\[.*\]|\d+", "#", regex=True).value_counts().head(6).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    games = pd.read_csv(PROC / "nba_games_with_game_id.csv")
    games = games[games.regular.astype(bool)]
    box = pd.read_csv(PROC / "all_boxscores.csv", usecols=BOX_COLS, low_memory=False)
    stints = pd.read_csv(PROC / "lineup_stints.csv")
    audit = pd.read_csv(PROC / "lineup_stint_audit.csv")
    targets = espn_games(games, box)
    if args.validate:
        return validate(targets, box, stints)

    done = set(audit.GAME_ID) if not args.retry_failed else set(audit.GAME_ID[audit.ok.astype(bool)])
    todo = targets[~targets.GAME_ID.isin(done)]
    print(f"[extend] {len(todo)} ESPN-sourced games to build ({len(targets)} total)")
    pbp, skipped = pbp_frame(todo, box)
    new, new_audit = build_stints(pbp, box) if len(pbp) else (pd.DataFrame(), pd.DataFrame())
    extra = pd.DataFrame([{"GAME_ID": g, "ok": False, "reason": r, "max_min_gap_sec": np.nan} for g, r in skipped])
    new_audit = pd.concat([new_audit, extra], ignore_index=True)
    rebuilt = set(new_audit.GAME_ID)
    stints = pd.concat([stints[~stints.GAME_ID.isin(rebuilt)], new], ignore_index=True)
    audit = pd.concat([audit[~audit.GAME_ID.isin(rebuilt)], new_audit], ignore_index=True)
    stints.to_csv(PROC / "lineup_stints.csv", index=False)
    audit.to_csv(PROC / "lineup_stint_audit.csv", index=False)
    print(f"[extend] built {int(new_audit.ok.sum())}/{len(new_audit)}; failures:")
    print(new_audit[~new_audit.ok].reason.str.replace(r"'.*'|\[.*\]|\d+", "#", regex=True).value_counts().head(6).to_string())


if __name__ == "__main__":
    main()
