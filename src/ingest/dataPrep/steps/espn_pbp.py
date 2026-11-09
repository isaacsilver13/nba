"""Turn an ESPN game summary's plays into the play-by-play frame that build_lineup_stints reads.

stats.nba.com and cdn.nba.com refuse this PC, so current-season play-by-play comes from ESPN (the source
tools/ingest_daily.py already uses for scores and box scores). ESPN names every player by ESPN athlete id, so substitutions
carry both ids and need no name matching: `personId` is the player going out, `inPersonId` the one coming in. Ids are
translated to the repo's PLAYER_ID through the `espn_player_id` column of the same game's box score.

Output columns follow the nba_data V3 layout where the stint builder uses them (actionType, subType, description, period,
clock "PTmmMss.ssS", teamId, personId, location h/v, scoreHome/scoreAway), plus `inPersonId`.
"""
import re

import numpy as np
import pandas as pd

COLUMNS = ["gameId", "period", "clock", "teamId", "personId", "inPersonId", "location", "actionType", "subType",
           "description", "scoreHome", "scoreAway", "playerName", "playerNameI", "teamTricode"]


def _clock(display):
    """ESPN '7:46' or '34.5' -> 'PT07M46.00S'."""
    text = str(display or "0:00")
    minutes, _, seconds = text.rpartition(":")
    return f"PT{int(minutes or 0):02d}M{float(seconds):05.2f}S"


def _kind(play):
    """(actionType, subType) in the NBA V3 vocabulary the stint builder counts."""
    text = re.sub(r"\s+", " ", play.get("type", {}).get("text", "")).strip()
    if text == "Substitution":
        return "Substitution", ""
    if "Free Throw" in text:
        return "Free Throw", ""
    if "Turnover" in text:
        return "Turnover", ""
    if text in ("Offensive Rebound", "Defensive Rebound"):
        return "Rebound", text.split()[0].lower()
    if text in ("End Period", "End Game"):
        return "period", ""
    if play.get("shootingPlay"):
        return ("Made Shot" if play.get("scoringPlay") else "Missed Shot"), ""
    if "Foul" in text:
        return "Foul", ""
    return text or "Other", ""


def plays_to_pbp(summary, game_id, to_code, team_ids, espn_to_pid):
    """One game's plays as a DataFrame in the stint builder's input layout.

    to_code: ESPN team abbreviation -> NBA tricode. team_ids: tricode -> TEAM_ID. espn_to_pid: ESPN athlete id -> PLAYER_ID.
    Unmapped athletes become 0 (so they can never be on court); the stint builder then excludes the game and says why.
    """
    sides = {c["team"]["id"]: (to_code(c["team"]["abbreviation"]), c["homeAway"][0])
             for c in summary["header"]["competitions"][0]["competitors"]}
    pid = lambda a: espn_to_pid.get(int(a), 0)  # noqa: E731
    rows = []
    for pl in summary.get("plays", []):
        action, sub = _kind(pl)
        team = sides.get(str(pl.get("team", {}).get("id")))
        who = [p["athlete"]["id"] for p in pl.get("participants", []) if "athlete" in p]
        actor, entering = (pid(who[1]) if action == "Substitution" and len(who) > 1 else
                           pid(who[0]) if who else 0), np.nan
        if action == "Substitution" and who:
            entering = pid(who[0])
        rows.append({
            "gameId": game_id, "period": pl["period"]["number"], "clock": _clock(pl.get("clock", {}).get("displayValue")),
            "teamId": team_ids[team[0]] if team else 0, "personId": actor,
            "inPersonId": entering if entering else np.nan, "location": team[1] if team else None,
            "actionType": action, "subType": sub, "description": pl.get("text", ""),
            "scoreHome": pl.get("homeScore"), "scoreAway": pl.get("awayScore"),
            "playerName": None, "playerNameI": None, "teamTricode": team[0] if team else None,
        })
    return pd.DataFrame(rows, columns=COLUMNS)
