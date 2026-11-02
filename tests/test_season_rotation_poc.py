"""Proof-of-concept history features must not learn from the target game."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from season_rotation_poc import add_poc_features, rotation_history  # noqa: E402


def test_rotation_uses_multiple_prior_games_not_rest_game_roster():
    days = pd.date_range("2026-04-08", periods=5)
    games = pd.DataFrame({"GAME_ID": range(1, 6), "date": days,
                          "season": [2026] * 4 + [2027], "home": ["AAA"] * 5,
                          "away": ["BBB"] * 5})
    rows = []
    for gid in range(1, 4):
        rows.extend((gid, "AAA", pid, "30:00", 10) for pid in range(1, 9))
    rows.extend((4, "AAA", pid, "20:00", 2) for pid in range(9, 17))
    box = pd.DataFrame(rows, columns=["GAME_ID", "team", "PLAYER_ID", "minutes", "points"])
    target = rotation_history(games, box).set_index(["GAME_ID", "team"]).loc[(5, "AAA")]
    assert target["rotation_points"] == 7.5
    assert target["same_season_games"] == 0
    assert target["prior_season"] == 2026

    target_rows = pd.DataFrame([(5, "AAA", 999, "48:00", 100)], columns=box.columns)
    changed = rotation_history(games, pd.concat([box, target_rows])).set_index(["GAME_ID", "team"]).loc[(5, "AAA")]
    assert changed["rotation_points"] == target["rotation_points"]


def test_previous_absence_becomes_unknown_across_season():
    summary = pd.DataFrame({"GAME_ID": [5, 5], "team": ["AAA", "BBB"],
                            "rotation_points": [7.5, 5.0], "prior_season": [2027, 2026],
                            "gap_days": [2, 193], "same_season_games": [1, 0],
                            "rotation_coverage": [1.0, 0.9]})
    game = pd.DataFrame({"GAME_ID": [5], "home": ["AAA"], "away": ["BBB"],
                         "season": [2027], "prev_game_minutes_out": [-44.6]})
    got = add_poc_features(game, summary).iloc[0]
    assert np.isnan(got["poc_prev_minutes_out_same_season"])
    assert got["poc_rotation_points_diff"] == 2.5
    assert got["poc_rotation_gap_days_max"] == 193
    assert got["poc_rotation_same_season_games_min"] == 0
    assert got["poc_rotation_coverage_min"] == 0.9


def test_missing_boxscore_does_not_count_as_zero_points():
    games = pd.DataFrame({"GAME_ID": range(1, 7),
                          "date": pd.date_range("2026-04-01", periods=6),
                          "season": [2026] * 6, "home": ["AAA"] * 6,
                          "away": ["BBB"] * 6})
    box = pd.DataFrame([(gid, "AAA", pid, "30:00", 10)
                        for gid in (1, 2, 4, 5) for pid in range(1, 9)],
                       columns=["GAME_ID", "team", "PLAYER_ID", "minutes", "points"])
    target = rotation_history(games, box).set_index(["GAME_ID", "team"]).loc[(6, "AAA")]
    assert target["rotation_coverage"] == 0.8
    assert target["rotation_points"] == 10
