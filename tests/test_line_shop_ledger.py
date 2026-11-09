"""Line-shopping ledger helpers (no data or network)."""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import line_shop_ledger as ls  # noqa: E402


def test_decimal_odds():
    assert np.allclose(ls.decimal([150, -200, 100]), [2.5, 1.5, 2.0])


def test_illinois_fees_are_waived_above_each_books_threshold():
    assert ls.fee("fanduel", 10) == 0.5 and ls.fee("fanduel", 25) == 0
    assert ls.fee("draftkings", 25) == 0.5 and ls.fee("draftkings", 50) == 0
    assert ls.fee("betmgm", 5) == 0


def test_ticker_uses_the_eastern_game_date_and_away_home_order():
    tip = datetime(2026, 10, 24, 1, 30, tzinfo=timezone.utc)   # 9:30pm ET on the 23rd
    assert ls.ticker(tip, "HOU", "SAS", "SAS") == "KXNBAGAME-26OCT23HOUSAS-SAS"


def test_kalshi_mid_uses_the_last_valid_quote_at_or_before_the_snapshot():
    c = {100: (0.55, 0.57), 160: (0.50, 0.58), 220: (0.60, 0.61)}
    assert ls.mid_at(c, 200) == 0.56          # 160 is 8c wide, 220 is after the snapshot
    assert ls.mid_at({}, 200) is None
    assert ls.mid_at({100: (0.55, 0.57)}, 100 + ls.MAX_QUOTE_AGE_S + 1) is None   # too old


def test_kalshi_fair_normalizes_the_two_markets():
    home, away = {10: (0.60, 0.62)}, {10: (0.39, 0.41)}
    assert np.isclose(ls.kalshi_fair(home, away, 10), 0.61 / (0.61 + 0.40))
    assert np.isnan(ls.kalshi_fair(home, {}, 10))


def test_consensus_is_the_other_books_devigged_median():
    rows = []
    for book, (h, a) in {"x": (-150, 130), "y": (-140, 120), "z": (-200, 170)}.items():
        rows += [{"request_id": "r", "event_id": "e", "bookmaker_id": book, "side_id": "home", "odds_american": h},
                 {"request_id": "r", "event_id": "e", "bookmaker_id": book, "side_id": "away", "odds_american": a}]
    d = pd.DataFrame(rows)
    c = ls.consensus(d)
    def p(h, a):
        ih, ia = 1 / ls.decimal([h])[0], 1 / ls.decimal([a])[0]
        return ih / (ih + ia)
    x_home = c[(d.bookmaker_id == "x") & (d.side_id == "home")].iloc[0]
    assert np.isclose(x_home, np.median([p(-140, 120), p(-200, 170)]))
