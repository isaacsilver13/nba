"""Stage 2 retrospective helpers (no local data or network)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import ml_clv_retro as r  # noqa: E402


def test_curve_win_prob_is_half_at_pickem_and_increases_with_the_prediction():
    p = r.p_curve([-6.0, 0.0, 6.0], 13.0, norm.cdf)
    assert abs(p[1] - 0.5) < 1e-9
    assert p[0] < p[1] < p[2]
    assert np.isclose(p[0] + p[2], 1.0)


def test_curve_scale_fit_recovers_a_known_scale():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 6, 8000)
    y = (x + rng.normal(0, 12, x.size) > 0).astype(float)
    assert abs(r.fit_curve_scale(x, y, norm.cdf) - 12) < 0.8


def test_ticker_is_date_away_home_team():
    assert r.parse_ticker("KXNBAGAME-26JAN15MILSAS-SAS") == (pd.Timestamp("2026-01-15"), "MIL", "SAS", "SAS")


def test_quote_filter_needs_a_two_sided_book_within_five_cents():
    c = {1: (0.72, 0.74), 2: (0.70, 0.76), 3: (0.0, 0.05), 4: (0.50, 0.55)}
    assert r.quotes(c, 1) == (0.72, 0.74)
    assert r.quotes(c, 2) is None   # 6c wide
    assert r.quotes(c, 3) is None   # no bid
    assert r.quotes(c, 4) == (0.50, 0.55)   # exactly 5c is allowed
    assert r.quotes(c, 99) is None  # no candle


def test_bet_rule_takes_the_larger_ev_side_and_measures_clv_against_the_close_mid():
    d = pd.DataFrame({"GAME_DATE": pd.to_datetime(["2026-01-01"] * 3), "p_curve": [0.60, 0.30, 0.51],
                      "E06_ask_home": [0.50, 0.50, 0.50], "E06_ask_away": [0.52, 0.52, 0.52],
                      "mid_close_home": [0.55, 0.40, 0.50], "mid_close_away": [0.45, 0.60, 0.50],
                      "home_margin": [5, -3, 2]})
    b = r.bets(d, "E06", "p_curve")
    assert len(b) == 2                                   # game 3: EV +2% < 3%, no bet
    assert np.allclose(b.ev, [0.20, 0.70 / 0.52 - 1])    # game 1 home, game 2 away
    assert np.allclose(b.clv, [0.55 / 0.50 - 1, 0.60 / 0.52 - 1])
    assert np.allclose(b.roi, [1, 1 / 0.52 - 1])         # both sides won
    assert np.all(b.clv_net < b.clv)
