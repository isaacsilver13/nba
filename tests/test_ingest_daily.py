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


def team(abbr, fg, tp, ft, oreb, dreb, tov, total_tov):
    stats = {"fieldGoalsMade-fieldGoalsAttempted": fg, "threePointFieldGoalsMade-threePointFieldGoalsAttempted": tp,
             "freeThrowsMade-freeThrowsAttempted": ft, "offensiveRebounds": oreb, "defensiveRebounds": dreb,
             "turnovers": tov, "totalTurnovers": total_tov, "fastBreakPoints": "9"}
    return {"team": {"abbreviation": abbr}, "statistics": [{"name": k, "displayValue": v} for k, v in stats.items()]}


def test_v3_rows(monkeypatch):
    # HOU 124 - OKC 125 (2025-10-21), totals as ESPN reports them
    teams = [team("HOU", "43-97", "11-39", "27-31", "16", "36", "22", "25"),
             team("OKC", "44-94", "13-41", "24-29", "12", "37", "15", "15")]
    monkeypatch.setattr(ing, "summary", lambda _: {"boxscore": {"teams": teams}})
    hou, okc = ing.v3_rows("401", 22590001, {"HOU": 1, "OKC": 2})
    est = ((97 + .44 * 31 - 16 + 22) + (94 + .44 * 29 - 12 + 15)) / 2
    poss = ing.POSS_FIT[0] + ing.POSS_FIT[1] * est
    assert (hou["TEAM_ABBREVIATION"], hou["TEAM_ID"], hou["adv_possessions"]) == ("HOU", 1, round(poss))
    assert hou["ff_effective_field_goal_percentage"] == round((43 + 5.5) / 97, 3)
    assert hou["ff_free_throw_attempt_rate"] == round(31 / 97, 3)
    assert hou["ff_team_turnover_percentage"] == round(25 / poss, 3)  # team turnovers count
    assert hou["ff_offensive_rebound_percentage"] == round(ing.OREB_FIT[0] + ing.OREB_FIT[1] * 16 / (16 + 37), 3)
    assert hou["adv_offensive_rating"] == okc["adv_defensive_rating"] == round(100 * 124 / poss, 1)
    assert hou["adv_net_rating"] == -okc["adv_net_rating"]


def test_game_ids_sit_above_real_nba_ids():
    assert ing.game_id_base(2026) == 22590000 and ing.game_id_base(2027) == 22690000
