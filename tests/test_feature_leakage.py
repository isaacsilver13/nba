"""Leakage regression tests for the feature pipeline.

Fast tests run step 2 on a synthetic schedule and check, for a late game, that
  * flipping the game's own final score changes none of its pre-game features, and
  * deleting every later game changes none of them either.
The slow point-in-time test (real data, ~2 min) is tools/pit_check.py; run it with RUN_PIT=1.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ingest.dataPrep.config import get_config  # noqa: E402
from ingest.dataPrep.steps import calculate_elo_and_rest as step2  # noqa: E402
from ingest.dataPrep.steps import engineer_advanced_features as step4  # noqa: E402

# columns that are the game's result (or defined by it); everything else must be a pre-game fact
OUTCOME_COLS = {
    "score_home", "score_away", "home_win", "margin", "total_points", "home_margin", "away_margin",
    "favorite_margin", "market_error", "favorite_cover", "favorite_cover_manual",
}


def make_games(n_days=70, n_teams=8, seed=1):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    rows, gid = [], 21900000
    for day in range(n_days):
        order = rng.permutation(teams)
        for k in range(0, n_teams, 2):
            if rng.random() < 0.25:  # not every team plays every day
                continue
            gid += 1
            fav_home = rng.random() < 0.5
            rows.append(dict(
                GAME_ID=gid, date=pd.Timestamp("2024-01-01") + pd.Timedelta(days=day), season=2024, regular=True,
                home=order[k], away=order[k + 1], score_home=int(rng.integers(90, 130)),
                score_away=int(rng.integers(90, 130)), spread=float(rng.integers(1, 12)) + 0.5,
                whos_favored="home" if fav_home else "away", id_spread=1,
            ))
    return pd.DataFrame(rows)


def run_step2(games):
    cfg = get_config()
    cfg["cache"]["enabled"] = False
    out = step2.run(cfg, {"games_with_ids": games.copy()})["games_processed"]
    return out.set_index("GAME_ID")


def _row(df, gid):
    return df.loc[gid]


def _same(x, y):
    if pd.isna(x) and pd.isna(y):
        return True
    try:
        return bool(np.isclose(float(x), float(y)))
    except (TypeError, ValueError):
        return x == y


def _assert_same_features(a, b, gid):
    cols = [c for c in a.columns if c in b.columns and c not in OUTCOME_COLS]
    bad = [c for c in cols if not _same(a.loc[gid, c], b.loc[gid, c])]
    assert not bad, f"pre-game features changed: {bad}"


@pytest.fixture(scope="module")
def games():
    return make_games()


@pytest.fixture(scope="module")
def target(games):
    late = games[games["date"] >= games["date"].max() - pd.Timedelta(days=5)]
    return int(late["GAME_ID"].iloc[0])


def test_step2_ignores_own_result(games, target):
    base = run_step2(games)
    flipped = games.copy()
    m = flipped["GAME_ID"] == target
    flipped.loc[m, ["score_home", "score_away"]] = [int(games.loc[games.GAME_ID == target, "score_away"].iloc[0]) + 9,
                                                    int(games.loc[games.GAME_ID == target, "score_home"].iloc[0])]
    alt = run_step2(flipped)
    assert target in base.index and target in alt.index
    _assert_same_features(base, alt, target)


def test_step2_ignores_later_games(games, target):
    base = run_step2(games)
    t_date = games.loc[games.GAME_ID == target, "date"].iloc[0]
    trunc = run_step2(games[(games["date"] < t_date) | (games["GAME_ID"] == target)])
    _assert_same_features(base, trunc, target)


def test_push_games_stay_in_the_history(games):
    pushed = games.copy()
    pushed.loc[pushed.index[100], ["score_home", "score_away", "spread", "whos_favored"]] = [100, 94, 6.0, "home"]
    assert pushed.loc[pushed.index[100], "GAME_ID"] in run_step2(pushed).index


def _history(games, team, before_date):
    h = games[((games.home == team) | (games.away == team)) & (games.date < before_date)].sort_values(["date", "GAME_ID"])
    return h


def test_back_to_back_and_rest_use_all_games(games, target):
    out = run_step2(games)
    g = games[games.GAME_ID == target].iloc[0]
    for side in ("home", "away"):
        prev = _history(games, g[side], g["date"]).iloc[-1]
        expected_rest = (g["date"] - prev["date"]).days
        assert out.loc[target, f"{side}_b2b"] == int(expected_rest == 1)


def test_ats_roll_last20_counts_covers_not_the_wrong_sign(games, target):
    out = run_step2(games)
    g = games[games.GAME_ID == target].iloc[0]

    def cover_rate(team):
        vals = []
        for _, r in _history(games, team, g["date"]).iterrows():
            sp = -r["spread"] if r["whos_favored"] == "home" else r["spread"]  # home handicap
            margin = r["score_home"] - r["score_away"]
            m = margin + sp if r["home"] == team else -margin - sp
            vals.append(1.0 if m > 0 else 0.0 if m < 0 else 0.5)
        vals = vals[-20:]
        return np.mean(vals) if len(vals) >= 5 else 0.0

    expected = cover_rate(g["home"]) - cover_rate(g["away"])
    assert np.isclose(out.loc[target, "ats_roll_last20"], expected)


def test_ats_roll_last20_skips_games_without_a_line(games, target):
    g = games[games.GAME_ID == target].iloc[0]
    last20 = _history(games, g["home"], g["date"]).iloc[-20:]
    blanked = games.copy()  # the home team's previous game has no line: average the other 19, don't count a push
    blanked.loc[blanked.GAME_ID == last20["GAME_ID"].iloc[-1], ["spread", "whos_favored"]] = [np.nan, np.nan]

    def cover(r):
        m = r["score_home"] - r["score_away"] + (-r["spread"] if r["whos_favored"] == "home" else r["spread"])
        m = m if r["home"] == g["home"] else -m
        return 1.0 if m > 0 else 0.0 if m < 0 else 0.5

    expected = np.mean([cover(r) for _, r in last20.iloc[:-1].iterrows()])
    assert np.isclose(run_step2(blanked).loc[target, "home_ats_roll_last20"], expected)


def test_pregame_out_excludes_coach_decisions_and_blank_reasons():
    players = pd.DataFrame({
        "GAME_ID": ["0021900001"] * 6, "TEAM_ID": [1] * 6, "PLAYER_ID": range(6), "team": ["TOR"] * 6,
        "minutes": ["30:00", "", "", "", "", ""],
        "comment": ["", "DNP - Coach's Decision", "DND - Injury/Illness", "NWT - Not With Team",
                    "DNP - Injury/Illness", ""],
    })
    games = pd.DataFrame({"GAME_ID": ["0021900001"], "date": [pd.Timestamp("2019-10-22")], "home": ["TOR"],
                          "away": ["NOP"], "season": [2020]})
    cleaned, _ = step4.clean_player_data(players, games)
    flag = cleaned.sort_values("PLAYER_ID")["pregame_out"].tolist()
    assert flag == [False, False, True, True, True, False]


def test_duplicated_player_game_rows_are_collapsed():
    row = {"GAME_ID": "0021900001", "TEAM_ID": 1, "PLAYER_ID": 7, "team": "TOR", "minutes": "30:00", "comment": "",
           "points": 20}
    players = pd.DataFrame([row] * 16 + [{**row, "PLAYER_ID": 8, "points": 5}])
    games = pd.DataFrame({"GAME_ID": ["0021900001"], "date": [pd.Timestamp("2019-10-22")], "home": ["TOR"],
                          "away": ["NOP"], "season": [2020]})
    cleaned, _ = step4.clean_player_data(players, games)
    assert len(cleaned) == 2 and cleaned["points"].sum() == 25  # not 16 * 20 + 5


@pytest.mark.skipif(not os.environ.get("RUN_PIT"), reason="slow: set RUN_PIT=1 (needs data/processed built by the current code)")
@pytest.mark.parametrize("date", ["2022-12-15", "2024-03-10"])
def test_point_in_time_rebuild_matches_full_build(date):
    r = subprocess.run([sys.executable, str(REPO / "tools" / "pit_check.py"), "--date", date],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(not os.environ.get("RUN_PIT"), reason="slow: set RUN_PIT=1 (needs data/processed built by the current code)")
@pytest.mark.parametrize("date", ["2022-12-15", "2024-03-10"])
def test_slate_rows_match_full_build(date):
    """Slate mode sees no score, line, box score or inactive list for `date`: every other column must equal the full
    build, so no feature depends on information a 9:30am run could not have."""
    sys.path.insert(0, str(REPO / "tools"))
    import run_features
    slate = run_features.slate_rows(date).set_index("GAME_ID")
    full = pd.read_csv(run_features.DATA / "processed" / "df_model_3.csv", low_memory=False)
    full = full[pd.to_datetime(full["date_x"]) == pd.Timestamp(date)]
    full = full.set_index(run_features._gid(full["GAME_ID"]))
    assert len(full) and set(full.index) <= set(slate.index)
    unknown_at_slate_time = {"GAME_ID", "home_margin", "away_margin", "spread", "spread_signed", "is_home_favorite",
                             *run_features.SAME_DAY_UNKNOWN}
    assert slate[run_features.SAME_DAY_UNKNOWN].isna().all().all()
    cols = [c for c in full.select_dtypes("number").columns if c not in unknown_at_slate_time]
    a, b = full[cols].to_numpy(float), slate.loc[full.index, cols].to_numpy(float)
    same = np.isclose(a, b, rtol=1e-7, atol=1e-7, equal_nan=True).all(axis=0)
    assert same.all(), [c for c, ok in zip(cols, same) if not ok]


def test_endpoint_families_with_repeated_keys_cannot_multiply_player_rows(tmp_path, monkeypatch):
    import types
    for name in ("nba_api", "nba_api.stats", "nba_api.stats.library", "nba_api.stats.library.http"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["nba_api.stats"].endpoints = types.ModuleType("endpoints")
    sys.modules["nba_api.stats.library.http"].NBAStatsHTTP = object
    from ingest.dataPrep.steps import fetch_player_boxscores as fetch

    gid = "0022200001"
    base = pd.DataFrame({"GAME_ID": gid, "TEAM_ID": [1, 1, 2], "PLAYER_ID": [10, 11, 20], "PTS": [5, 6, 7]})
    # every extra family returns TWO rows per player (the situation that produced 16x / 32x duplicates)
    extra = pd.concat([base.assign(USG=0.2), base.assign(USG=0.3)], ignore_index=True)
    for fam, frame in (("traditional", base), ("advanced", extra), ("usage", extra), ("misc", extra)):
        (tmp_path / fam).mkdir()
        frame.to_csv(tmp_path / fam / f"{gid}_player.csv", index=False)
    players, _ = fetch._rebuild_outputs_from_cache([gid], ["traditional", "advanced", "usage", "misc"], str(tmp_path))
    assert len(players) == 3 and not players.duplicated(["GAME_ID", "TEAM_ID", "PLAYER_ID"]).any()
    with pytest.raises(ValueError, match="repeat"):
        fetch._require_unique(pd.concat([players, players]), ["GAME_ID", "TEAM_ID", "PLAYER_ID"], "player boxscores")


def test_usage_concentration_is_the_top_n_share_of_team_usage():
    from ingest.dataPrep.steps import aggregate_player_to_team as agg
    # team A: usage proxies 50/30/10/10 (fta=0, tov=0 so proxy == fga); team B: five equal players
    rows = [("A", 50), ("A", 30), ("A", 10), ("A", 10)] + [("B", 20)] * 5
    d = pd.DataFrame({"GAME_ID": "0021900001", "team": [t for t, _ in rows], "fga": [u for _, u in rows],
                      "fta": 0, "turnovers": 0})
    out = agg.usage_concentration(d).set_index("team")
    assert np.isclose(out.loc["A", "usg_top2_share"], 0.80) and np.isclose(out.loc["A", "usg_top3_share"], 0.90)
    assert np.isclose(out.loc["B", "usg_top2_share"], 0.40) and np.isclose(out.loc["B", "usg_top3_share"], 0.60)


def test_games_downloaded_in_this_run_reach_the_output_tables(tmp_path, monkeypatch):
    # the cache index used to be built before downloading, so a run's newly fetched games were missing from its output
    import types
    for name in ("nba_api", "nba_api.stats", "nba_api.stats.library", "nba_api.stats.library.http"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["nba_api.stats"].endpoints = types.ModuleType("endpoints")
    sys.modules["nba_api.stats.library.http"].NBAStatsHTTP = object
    from ingest.dataPrep.config import get_config
    from ingest.dataPrep.steps import fetch_player_boxscores as fetch

    def fake_fetch(game_id, family, cache_dir, *a, **k):
        gid = str(int(game_id)).zfill(10)
        p = pd.DataFrame({"GAME_ID": gid, "TEAM_ID": [1, 2], "PLAYER_ID": [10, 20], "PTS": [5, 7]})
        t = pd.DataFrame({"GAME_ID": gid, "TEAM_ID": [1, 2], "PTS": [5, 7]})
        os.makedirs(os.path.join(cache_dir, family), exist_ok=True)
        p.to_csv(os.path.join(cache_dir, family, f"{gid}_player.csv"), index=False)
        t.to_csv(os.path.join(cache_dir, family, f"{gid}_team.csv"), index=False)
        return p, t, False
    monkeypatch.setattr(fetch, "fetch_endpoint_cached", fake_fetch)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)

    pd.DataFrame({"GAME_ID": ["0022500001"], "date": ["2025-10-21"]}).to_csv(tmp_path / "games.csv", index=False)
    cfg = get_config({"cache": {"write_outputs": False}, "steps": {"step3": {
        "paths": {"input_games": str(tmp_path / "games.csv"), "cache_dir": str(tmp_path / "cache"),
                  "output_players": str(tmp_path / "p.csv"), "output_team_v3": str(tmp_path / "t.csv"),
                  "failed_games": str(tmp_path / "f.csv")},
        "endpoint_families": {"advanced": False, "four_factors": False, "scoring": False, "usage": False,
                              "misc": False, "player_track": False}}}})
    out = fetch.run(cfg)
    assert set(out["player_boxscores"]["GAME_ID"]) == {"0022500001"}
