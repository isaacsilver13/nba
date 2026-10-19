from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiment as exp
from config_guard import validate_config


def _config():
    source = Path(exp.__file__).read_text(encoding="utf8")
    return source[source.index("# AGENT-EDITABLE CONFIG"):source.index("# FROZEN")]


def test_generated_config_is_validated_without_execution():
    valid = _config()
    validate_config(valid)
    for replacement in ("__import__('os').system('untrusted')", "(lambda: 10)()",
                        "[x for x in range(10)]", "lgb.__dict__['danger']()"):
        with pytest.raises(ValueError):
            validate_config(valid.replace("TRAIN_SIZE = 2500", "TRAIN_SIZE = " + replacement))
    for extra in ("\nimport os", "\nprint('untrusted')", "\nlgb = 1", "\nTRAIN_SIZE = 5"):
        with pytest.raises(ValueError):
            validate_config(valid + extra)
    with pytest.raises(ValueError):
        validate_config(valid.replace("TEST_SIZE = 300", "# TEST_SIZE omitted"))
    with pytest.raises(ValueError):
        validate_config(valid.replace("FOLD_WORKERS = 8", "FOLD_WORKERS = 100"))
    with pytest.raises(ValueError, match="n_jobs"):
        validate_config(valid.replace("n_jobs=1,", ""))


def test_inclusions_cannot_override_frozen_causal_exclusions(monkeypatch):
    denied = ["home_margin", "spread_signed", "is_home_favorite", "missing_top1_flag",
              "missing_top2_flag", "minutes_lost_last_game", "usage_lost_last_game", "x_postgame_x"]
    monkeypatch.setattr(exp, "EXTRA_FEATURE_INCLUSIONS", denied + ["historical_extra"])
    frame = pd.DataFrame({name: [1.0] for name in denied + ["historical_extra"]})
    assert exp.build_features(frame) == ["historical_extra"]


def test_walk_forward_keeps_complete_dates_and_row_identity():
    dates = pd.date_range("2024-01-01", periods=8).repeat([2, 3, 1, 4, 2, 3, 2, 1])
    frame = pd.DataFrame({"GAME_DATE": dates, "id": range(len(dates))}).sample(frac=1, random_state=1)
    seen = []
    for train, test in exp.walk_forward_splits(frame, "GAME_DATE", 5, 4):
        assert train.GAME_DATE.max() < test.GAME_DATE.min()
        assert set(train.index).isdisjoint(test.index)
        assert (frame.loc[test.index, "id"] == test.id).all()
        for day in test.GAME_DATE.unique():
            assert len(test[test.GAME_DATE == day]) == len(frame[frame.GAME_DATE == day])
        seen.extend(test.index)
    assert len(seen) == len(set(seen))
    assert frame.loc[seen].GAME_DATE.max() == dates.max()
    with pytest.raises(ValueError):
        list(exp.walk_forward_splits(frame, "GAME_DATE", 0, 4))


def test_same_day_bets_use_one_opening_bankroll(monkeypatch):
    monkeypatch.setattr(exp, "MAX_KELLY", 0.08)
    monkeypatch.setattr(exp, "DAILY_MAX_EXPOSURE", 0.25)
    frame = pd.DataFrame({"GAME_DATE": pd.to_datetime(["2024-01-01"] * 2),
                          "home_margin": [10.0, 10.0], "spread_signed": [-3.0, -3.0],
                          "pred_ensemble": [30.0, 30.0], "sigma_ensemble": [1.0, 1.0],
                          "payout_home": [0.91, 0.91], "payout_away": [0.91, 0.91]})
    detail = exp.compute_roi(frame, detail=True)
    assert detail["bankroll"] == pytest.approx(1.1456)
    assert detail["bankroll_open"] == pytest.approx([1, 1])
    assert detail["bankroll_series"] == pytest.approx([1.1456, 1.1456])
    frame.loc[1, "GAME_DATE"] = pd.Timestamp("2024-01-02")
    assert exp.compute_roi(frame) == pytest.approx(1.0728**2 - 1)


def test_walk_forward_sorts_parsed_string_dates():
    frame = pd.DataFrame({"GAME_DATE": ["2024-2-01", "2024-10-01", "2024-11-01"]})
    for train, test in exp.walk_forward_splits(frame, "GAME_DATE", 1, 1):
        assert pd.to_datetime(train.GAME_DATE).max() < pd.to_datetime(test.GAME_DATE).min()


def test_top_players_ignore_target_game_participation_and_roster():
    from src.ingest.dataPrep.steps.engineer_advanced_features import historical_top_player_features

    past = pd.DataFrame({"GAME_ID": ["past"] * 9, "TEAM_ID": [1] * 9,
                         "date": ["2024-01-01"] * 9, "team": ["HOME"] * 9,
                         "PLAYER_ID": np.arange(9), "player_minutes_played_r5": np.arange(9) + 10,
                         "player_points_r5": np.arange(9) + 1, "did_play": [True] * 9,
                         "minutes_played": np.arange(9)})
    current = past.assign(GAME_ID="current", date="2024-01-02")

    def feature(current_rows):
        result = historical_top_player_features(pd.concat([past, current_rows]), 8)
        return result.set_index("GAME_ID").loc["current", "top8_points_r5"]

    expected = feature(current)
    assert np.isfinite(expected)
    changed = current.assign(minutes_played=0, did_play=False,
                             player_points_r5=999, PLAYER_ID=np.arange(9) + 100)
    assert feature(changed.sample(frac=1, random_state=2)) == expected
