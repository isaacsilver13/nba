"""Synthetic-game checks for the lineup stint builder (no local data needed)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ingest.dataPrep.steps.build_lineup_stints import build_game_stints, build_stints  # noqa: E402

H, A = 100, 200  # team ids


def _name(t, i):
    return f"{'Home' if t == H else 'Away'}{'abcdefg'[i]}"  # letters only: the matcher strips digits


def _roster():
    rows = []
    for t, base in ((H, 10), (A, 20)):
        for i in range(7):
            rows.append({"GAME_ID": 1, "TEAM_ID": t, "PLAYER_ID": base + i, "firstName": f"First{i}",
                         "familyName": _name(t, i), "nameI": f"F. {_name(t, i)}",
                         "position": "G" if i < 5 else None, "minutes": "10:00"})
    return pd.DataFrame(rows)


def _ev(num, clock, typ, team=0, pid=0, desc="", sh=None, sa=None, loc=None, period=1):
    return {"actionNumber": num, "clock": clock, "period": period, "teamId": team, "personId": pid,
            "description": desc, "actionType": typ, "subType": "", "scoreHome": sh, "scoreAway": sa,
            "location": loc, "playerName": None, "playerNameI": None}


def _pbp():
    # H player 11 hits a three at 11:00, H5 replaces H0 (id 10) at 6:00, A scores at 5:00, period ends 0:00
    ev = [
        _ev(1, "PT12M00.00S", "period"),
        _ev(2, "PT11M00.00S", "Made Shot", H, 11, "Homeb 3PT", 3, 0, "h"),
        _ev(3, "PT06M00.00S", "Substitution", H, 10, "SUB: Homef FOR Homea", loc="h"),
        _ev(4, "PT05M00.00S", "Made Shot", A, 21, "Awayb 2PT", 3, 2, "v"),
        _ev(5, "PT00M00.00S", "Missed Shot", A, 22, "MISS Awayc", None, None, "v"),
    ]
    return pd.DataFrame(ev)


def test_substitution_splits_stint_and_credits_points():
    st, err = build_game_stints(_pbp(), _roster())
    assert err is None
    home = st[st.is_home == 1].reset_index(drop=True)
    assert list(home.seconds) == [360, 720 - 360]
    assert "10" in home.lineup_key[0].split("-") and "10" not in home.lineup_key[1].split("-")
    assert "15" in home.lineup_key[1].split("-")
    # three-pointer lands in stint 0, the away two in stint 1; each row is that team's view
    assert (home.pts_for[0], home.pts_against[0]) == (3, 0)
    assert (home.pts_for[1], home.pts_against[1]) == (0, 2)
    away = st[st.is_home == 0].reset_index(drop=True)
    assert (away.pts_for[1], away.pts_against[1]) == (2, 0)


def test_future_event_does_not_change_earlier_stint():
    """Truncating the game after the substitution must leave the first stint's row unchanged."""
    full, _ = build_game_stints(_pbp(), _roster())
    cut, _ = build_game_stints(_pbp().iloc[:4], _roster())
    cols = ["lineup_key", "seconds", "pts_for", "pts_against", "fga_for"]
    assert full[full.is_home == 1].iloc[0][cols].tolist() == cut[cut.is_home == 1].iloc[0][cols].tolist()


def test_unresolvable_substitution_excludes_game_not_patches_it():
    bad = _pbp()
    bad.loc[bad.actionNumber == 3, "description"] = "SUB: Nobody FOR Homea"
    stints, audit = build_stints(bad.assign(gameId=1), _roster())
    assert stints.empty and not audit.ok.iloc[0] and "unresolved" in audit.reason.iloc[0]
