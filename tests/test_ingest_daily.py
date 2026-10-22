"""ESPN box score -> all_boxscores.csv rows (tools/ingest_daily.py), on a hand-made summary: no network, no data."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import ingest_daily as ing  # noqa: E402

KEYS = ["minutes", "points", "fieldGoalsMade-fieldGoalsAttempted",
        "threePointFieldGoalsMade-threePointFieldGoalsAttempted", "freeThrowsMade-freeThrowsAttempted", "rebounds",
        "assists", "turnovers", "steals", "blocks", "offensiveRebounds", "defensiveRebounds", "fouls", "plusMinus"]


def athlete(aid, name, stats, dnp=False, reason=None, starter=False):
    return {"athlete": {"id": str(aid), "displayName": name, "shortName": name, "jersey": "7",
                        "position": {"abbreviation": "G"}},
            "starter": starter, "didNotPlay": dnp, "reason": reason, "stats": stats}


SUMMARY = {"boxscore": {"players": [{"team": {"abbreviation": "GS"}, "statistics": [{"keys": KEYS, "athletes": [
    athlete(1, "Stephen Curry", ["35", "30", "10-20", "6-12", "4-4", "5", "7", "3", "1", "0", "1", "4", "2", "+9"],
            starter=True),
    athlete(2, "Pat Spencer", [], dnp=True, reason="COACH'S DECISION"),
    athlete(3, "Jimmy Butler III", [], dnp=True, reason="RIGHT KNEE SORENESS"),
    athlete(4, "Two Way", ["--"] + ["0"] * 13),  # listed by ESPN only
]}]}]}}


def test_box_rows(monkeypatch):
    monkeypatch.setattr(ing, "get_json", lambda *a, **k: SUMMARY)
    rows = pd.DataFrame(ing.box_rows("401", 22590001, {"GSW": 1610612744}))
    assert rows["familyName"].tolist() == ["Curry", "Spencer", "Butler III"]
    curry = rows.iloc[0]
    assert (curry["team"], curry["TEAM_ID"], curry["minutes"], curry["position"]) == ("GSW", 1610612744, "35:00", "G")
    assert (curry["fieldGoalsAttempted"], curry["threePointersMade"], curry["plusMinusPoints"]) == (20, 6, 9)
    assert rows["comment"].fillna("").tolist() == ["", "DNP - Coach's Decision", "DNP - Right Knee Soreness"]
    assert rows["minutes"].iloc[1:].isna().all() and (rows["points"].iloc[1:] == 0).all()

    old = pd.DataFrame({"PLAYER_ID": [201939, 5, 6], "firstName": ["Stephen", "Pat", "Pat"],
                        "familyName": ["Curry", "Spencer", "Spencer"], "endpoint_family": "traditional"})
    # unique name -> NBA id; ambiguous or unknown -> offset ESPN id
    assert ing.player_ids(rows, old).tolist() == [201939, ing.ESPN_ID_OFFSET + 2, ing.ESPN_ID_OFFSET + 3]


def test_game_ids_sit_above_real_nba_ids():
    assert ing.game_id_base(2026) == 22590000 and ing.game_id_base(2027) == 22690000
