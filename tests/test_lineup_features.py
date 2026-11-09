"""Leakage and arithmetic checks for pre-game lineup features (synthetic, no local data)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ingest.dataPrep.steps.engineer_lineup_features import team_lineup_features  # noqa: E402


def _stint(gid, team, key, start, secs, pf, pa, poss=50):
    # poss comes out as (fga + oreb-free terms): fga_for = fga_against = poss, others 0
    return {"GAME_ID": gid, "team": team, "lineup_key": key, "start_sec": start, "seconds": secs,
            "pts_for": pf, "pts_against": pa, "fga_for": poss, "fta_for": 0, "tov_for": 0, "oreb_for": 0,
            "fga_against": poss, "fta_against": 0, "tov_against": 0, "oreb_against": 0}


def _fixture():
    """Team X plays games 1..4 (same season) then game 5 in a new season; unit A starts, B is the bench five."""
    sched = pd.DataFrame({
        "GAME_ID": [1, 2, 3, 4, 5],
        "date": ["2024-10-22", "2024-10-24", "2024-10-26", "2024-10-28", "2025-10-22"],
        "season": [2025, 2025, 2025, 2025, 2026], "home": ["X"] * 5, "away": ["Y"] * 5,
    })
    rows = []
    for gid in (1, 2, 3, 4, 5):
        rows += [_stint(gid, "X", "A", 0, 2000, 40, 30), _stint(gid, "X", "B", 2000, 880, 20, 25),
                 _stint(gid, "Y", "C", 0, 2880, 55, 60)]
    return pd.DataFrame(rows), sched


def _row(feats, gid, team="X"):
    return feats[(feats.GAME_ID == gid) & (feats.team == team)].iloc[0]


def test_opener_has_no_lineup_features_and_new_season_does_not_inherit():
    stints, sched = _fixture()
    f = team_lineup_features(stints, sched)
    assert _row(f, 1).lu_games == 0 and np.isnan(_row(f, 1).lu_top_unit_net)
    assert _row(f, 5).lu_games == 0 and np.isnan(_row(f, 5).lu_top_unit_share)  # four old-season games ignored
    assert _row(f, 5).lu_days_gap == 359  # gap is still reported across seasons


def test_hand_computed_values():
    stints, sched = _fixture()
    r = _row(team_lineup_features(stints, sched, shrink_poss=50), 3)  # window = games 1, 2
    assert r.lu_games == 2
    assert np.isclose(r.lu_top_unit_share, 4000 / 5760)
    # A: +20 pts over 100 poss -> 20 per 100, shrunk by 100/(100+50)
    assert np.isclose(r.lu_top_unit_net, 20.0 * 100 / 150)
    assert np.isclose(r.lu_starter_unit_share, r.lu_top_unit_share)  # starters were the top unit
    assert np.isclose(r.lu_unit_concentration, (4000 / 5760) ** 2 + (1760 / 5760) ** 2)


def test_own_game_and_later_games_never_reach_a_row():
    stints, sched = _fixture()
    base = _row(team_lineup_features(stints, sched), 3)
    changed = stints.copy()
    own = changed.GAME_ID.eq(3) & changed.team.eq("X")
    changed.loc[own, ["pts_for", "pts_against", "seconds", "fga_for"]] = [999, 0, 1, 9999]
    changed = changed[~changed.GAME_ID.eq(4)]  # and a later game removed outright
    after = _row(team_lineup_features(changed, sched.iloc[:3]), 3)
    pd.testing.assert_series_equal(base, after)


def test_missing_stints_shrink_games_count_not_the_window():
    stints, sched = _fixture()
    thin = stints[~stints.GAME_ID.eq(2)]
    r = _row(team_lineup_features(thin, sched), 4)  # window = games 1,2,3 but game 2 has no stints
    assert r.lu_games == 2
