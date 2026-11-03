"""Research-tool selftests run in CI: the closing-line artifact guard and the model-free line-shop test."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import encompass_check  # noqa: E402
import holdout_eval  # noqa: E402
import line_shop_clv  # noqa: E402


def test_holdout_eval_selftest():
    holdout_eval.selftest()


def test_encompass_guard_and_clv_selftest():
    encompass_check.selftest()


def test_line_shop_selftest():
    line_shop_clv.selftest()


def test_line_shop_loader_ignores_non_moneyline_ml_markets(tmp_path):
    """FanDuel posts firstToScore as bet_type 'ml'; it must not be read as the game moneyline (found 2026-11-03)."""
    import pandas as pd
    rows = []
    for stat, side, american, t in (("points", "home", -150, "a"), ("points", "away", 130, "a"),
                                    ("firstToScore", "home", 900, "b"), ("firstToScore", "away", 900, "b")):  # later capture
        rows.append(dict(period_id="game", bet_type_id="ml", stat_id=stat, side_id=side, event_id="e1", bookmaker_id="fd",
                         slot="open", captured_at_utc=f"2026-11-01T1{t == 'b' and 5 or 0}:00", odds_american=american,
                         open_odds=american, close_odds=american))
    (tmp_path / "date=2026-11-01").mkdir()
    pd.DataFrame(rows).to_parquet(tmp_path / "date=2026-11-01" / "x.parquet")
    w = line_shop_clv.load(str(tmp_path))
    assert len(w) == 1 and abs(w.open_h.iloc[0] - (1 + 100 / 150)) < 1e-9
