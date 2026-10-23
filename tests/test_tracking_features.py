"""Leakage and arithmetic checks for pre-game tracking features (synthetic, no local data)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ingest.dataPrep.steps.engineer_tracking_features import team_tracking_features  # noqa: E402


def _p(gid, team, pid, mins, dist=2.0, speed=4.0, touches=40, passes=30, ufga=6, cfga=4, dfgm=3, dfga=5, reb=5):
    return {"GAME_ID": gid, "team": team, "PLAYER_ID": pid, "minutes": f"{mins}:00", "ptrk_speed": speed,
            "ptrk_distance": dist, "ptrk_touches": touches, "ptrk_passes": passes,
            "ptrk_uncontested_field_goals_attempted": ufga, "ptrk_contested_field_goals_attempted": cfga,
            "ptrk_defended_at_rim_field_goals_made": dfgm, "ptrk_defended_at_rim_field_goals_attempted": dfga,
            "ptrk_rebound_chances_total": reb}


def _fixture():
    sched = pd.DataFrame({
        "GAME_ID": [1, 2, 3, 4, 5],
        "date": ["2024-10-22", "2024-10-24", "2024-10-26", "2024-10-28", "2025-10-22"],
        "season": [2025, 2025, 2025, 2025, 2026], "home": ["X"] * 5, "away": ["Y"] * 5,
    })
    rows = []
    for gid in (1, 2, 3, 4, 5):
        rows += [_p(gid, "X", 1, 30), _p(gid, "X", 2, 10, dist=1.0, speed=2.0), _p(gid, "Y", 9, 40)]
    return pd.DataFrame(rows), sched


def _row(feats, gid, team="X"):
    return feats[(feats.GAME_ID == gid) & (feats.team == team)].iloc[0]


def test_opener_and_new_season_are_blank():
    box, sched = _fixture()
    f = team_tracking_features(box, sched)
    assert _row(f, 1).trk_games == 0 and np.isnan(_row(f, 1).trk_dist_per_min)
    assert _row(f, 5).trk_games == 0 and np.isnan(_row(f, 5).trk_speed)


def test_hand_computed_values_are_minutes_weighted():
    box, sched = _fixture()
    r = _row(team_tracking_features(box, sched, shrink_att=0), 3)  # window = games 1, 2; 80 rotation minutes
    assert r.trk_games == 2
    assert np.isclose(r.trk_dist_per_min, (2 * (2.0 + 1.0)) / 80)
    assert np.isclose(r.trk_speed, (2 * (4.0 * 30 + 2.0 * 10)) / 80)
    assert np.isclose(r.trk_uncontested_fga_share, 0.6)
    assert np.isclose(r.trk_rim_fg_allowed, 0.6)  # 3 of 5 at the rim, no shrinkage
    assert np.isclose(r.trk_rim_fga_per_min, (2 * 10) / 80)


def test_small_samples_shrink_toward_the_prior():
    box, sched = _fixture()
    r = _row(team_tracking_features(box, sched, shrink_att=1000, league_rim_fg=0.65), 3)
    assert abs(r.trk_rim_fg_allowed - 0.65) < 0.01


def test_own_game_and_later_games_never_reach_a_row():
    box, sched = _fixture()
    base = _row(team_tracking_features(box, sched), 3)
    changed = box.copy()
    own = changed.GAME_ID.eq(3) & changed.team.eq("X")
    changed.loc[own, ["ptrk_distance", "ptrk_touches", "ptrk_speed"]] = [99.0, 999, 99.0]
    changed = changed[~changed.GAME_ID.eq(4)]
    after = _row(team_tracking_features(changed, sched.iloc[:3]), 3)
    pd.testing.assert_series_equal(base, after)


def test_games_without_tracking_are_not_counted():
    box, sched = _fixture()
    box.loc[box.GAME_ID.eq(2), "ptrk_distance"] = np.nan  # game 2 untracked
    assert _row(team_tracking_features(box, sched), 4).trk_games == 2  # games 1 and 3 only
