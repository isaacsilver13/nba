"""ESPN plays -> stint builder input, on a tiny synthetic summary (no network)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ingest.dataPrep.steps.build_lineup_stints import build_game_stints  # noqa: E402
from ingest.dataPrep.steps.espn_pbp import _clock, plays_to_pbp  # noqa: E402

H, A = 100, 200


def _play(seq, period, clock, kind, team, who, text="", home=0, away=0, shooting=False, scoring=False):
    return {"sequenceNumber": str(seq), "type": {"text": kind}, "text": text, "period": {"number": period},
            "clock": {"displayValue": clock}, "team": {"id": team} if team else {}, "homeScore": home, "awayScore": away,
            "participants": [{"athlete": {"id": str(w)}} for w in who], "shootingPlay": shooting, "scoringPlay": scoring}


def _summary():
    comp = [{"team": {"id": "1", "abbreviation": "HOM"}, "homeAway": "home"},
            {"team": {"id": "2", "abbreviation": "AWY"}, "homeAway": "away"}]
    plays = [
        _play(1, 1, "12:00", "Jumpball", None, []),
        _play(2, 1, "11:00", "Jump Shot", "1", [1001], home=3, shooting=True, scoring=True),
        _play(3, 1, "6:00", "Substitution", "1", [1006, 1000], "Home5 enters the game for Home0", home=3),
        _play(4, 1, "5:00", "Layup Shot", "2", [2001], home=3, away=2, shooting=True, scoring=True),
        _play(5, 1, "0:00", "End Period", None, [], home=3, away=2),
    ]
    return {"header": {"competitions": [{"competitors": comp}]}, "plays": plays}


def _roster():
    rows = []
    for t, base in ((H, 1000), (A, 2000)):
        for i in range(7):
            rows.append({"GAME_ID": 1, "TEAM_ID": t, "PLAYER_ID": base + i, "firstName": f"F{t}{'abcdefg'[i]}",
                         "familyName": f"{'Home' if t == H else 'Away'}{'abcdefg'[i]}", "nameI": "x",
                         "position": "G" if i < 5 else None, "minutes": "10:00"})
    return pd.DataFrame(rows)


def test_clock_formats():
    assert _clock("7:46") == "PT07M46.00S" and _clock("34.5") == "PT00M34.50S" and _clock("12:00") == "PT12M00.00S"


def test_espn_game_becomes_stints_with_id_based_substitution():
    espn_to_pid = {str(p): p for p in range(1000, 1007)} | {str(p): p for p in range(2000, 2007)}
    pbp = plays_to_pbp(_summary(), 1, lambda a: {"HOM": "H", "AWY": "A"}[a], {"H": H, "A": A},
                       {int(k): v for k, v in espn_to_pid.items()})
    sub = pbp[pbp.actionType == "Substitution"].iloc[0]
    assert sub.personId == 1000 and sub.inPersonId == 1006  # out, in
    st, err = build_game_stints(pbp, _roster())
    assert err is None
    home = st[st.is_home == 1].reset_index(drop=True)
    assert list(home.seconds) == [360, 360]
    assert (home.pts_for[0], home.pts_against[0], home.pts_for[1], home.pts_against[1]) == (3, 0, 0, 2)
    assert "1006" in home.lineup_key[1].split("-") and "1000" not in home.lineup_key[1].split("-")
