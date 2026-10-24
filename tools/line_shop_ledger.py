"""Sharp-vs-soft moneyline ledger: Illinois book prices from SGO snapshots vs Kalshi's game-winner mid
(docs/line-shopping-scope.md). Rebuilt from immutable inputs on every run.

    python tools/line_shop_ledger.py                # outputs/line_shop/ml_ledger.csv + opportunity-frequency summary
    python tools/line_shop_ledger.py --stake 50     # net EV at this stake (Illinois per-bet fees; default $25)

Inputs: $NBA_DATA_DIR/raw/odds/sgo/date=*/*.parquet (git pull the data repo first) and Kalshi KXNBAGAME 1-minute
candles, fetched once per finalized market and cached in $NBA_DATA_DIR/kalshi/live/. A row is kept only if SGO saw
the price before the official tip, and Kalshi's fair price comes from a quote no later than the SGO snapshot.
CLV = p_close x decimal - 1: the bet's expected return if Kalshi's last pre-tip mid is the true probability.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kalshi_history import BASE, fetch  # noqa: E402

DATA = Path(os.environ.get("NBA_DATA_DIR", ROOT / "data"))
OUT = Path(os.environ.get("NBA_OUTPUTS_DIR", ROOT / "outputs")) / "line_shop"
ET = ZoneInfo("America/New_York")
IL_BOOKS = ["betmgm", "caesars", "draftkings", "espnbet", "fanduel"]
FEES = {"fanduel": (0.50, 25.0), "draftkings": (0.50, 50.0)}  # (fee, waived at stake >=); others charge none
MAX_SPREAD, MAX_QUOTE_AGE_S, LOOKBACK = 0.05, 15 * 60, timedelta(hours=12)
TEAMS = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}


def decimal(american):
    a = np.asarray(american, float)
    return np.where(a > 0, 1 + a / 100, 1 + 100 / -a)


def fee(book, stake):
    f, waived_at = FEES.get(book, (0.0, 0.0))
    return 0.0 if stake >= waived_at else f


def ticker(tip_utc, away, home, team):
    return f"KXNBAGAME-{tip_utc.astimezone(ET):%y%b%d}".upper() + f"{away}{home}-{team}"


def load_sgo():
    files = sorted((DATA / "raw" / "odds" / "sgo").glob("date=*/*.parquet"))
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d = d[(d.bet_type_id == "ml") & (d.period_id == "game") & (d.stat_id == "points")
          & d.available.astype(bool) & d.odds_american.notna()].copy()
    for c in ("captured_at_utc", "starts_at_utc", "last_updated_at"):
        d[c] = pd.to_datetime(d[c], utc=True)
    d = d[d.captured_at_utc < d.starts_at_utc]  # pre-tip prices only
    d["home"], d["away"] = d.home_team.map(TEAMS), d.away_team.map(TEAMS)
    d["team"] = np.where(d.side_id == "home", d.home, d.away)
    return d.dropna(subset=["home", "away"]).drop_duplicates(["request_id", "event_id", "side_id", "bookmaker_id"])


def candles(tk, tip):
    """{end_ts: (bid, ask)} of 1-minute candles over the 12 h before tip; cached once the market is finalized."""
    cache = DATA / "kalshi" / "live" / f"{tk}.json"
    if cache.exists():
        rows = json.loads(cache.read_text())
    else:
        end = int(tip.timestamp())
        r = fetch(f"{BASE}/series/KXNBAGAME/markets/{tk}/candlesticks?start_ts={end - int(LOOKBACK.total_seconds())}"
                  f"&end_ts={end}&period_interval=1")
        if r is None:
            return {}
        rows = [[c["end_period_ts"], c["yes_bid"].get("close_dollars"), c["yes_ask"].get("close_dollars")]
                for c in r.get("candlesticks", [])]
        m = fetch(f"{BASE}/markets/{tk}")
        if m and m.get("market", {}).get("status") in ("finalized", "settled") and rows:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(rows))
    return {int(t): (float(b), float(a)) for t, b, a in rows if b is not None and a is not None}


def mid_at(c, ts):
    """Mid of the last candle ending at or before ts (within MAX_QUOTE_AGE_S) with a two-sided book <= 5c wide."""
    for t in sorted((t for t in c if ts - MAX_QUOTE_AGE_S <= t <= ts), reverse=True):
        bid, ask = c[t]
        if 0 < bid < ask < 1 and ask - bid <= MAX_SPREAD + 1e-9:
            return (bid + ask) / 2
    return None


def kalshi_fair(c_team, c_opp, ts):
    """P(team wins): the two markets' mids normalized to sum to 1, or NaN."""
    a, b = mid_at(c_team, ts), mid_at(c_opp, ts)
    return np.nan if a is None or b is None else a / (a + b)


def consensus(d):
    """Leave-one-out median of other books' multiplicatively de-vigged P(team) in the same snapshot."""
    imp = 1 / decimal(d.odds_american)
    d = d.assign(imp=imp)
    tot = d.groupby(["request_id", "event_id", "bookmaker_id"]).imp.transform("sum")
    d["p_book"] = np.where(d.groupby(["request_id", "event_id", "bookmaker_id"]).imp.transform("size") == 2,
                           d.imp / tot, np.nan)
    out = []
    for _, g in d.groupby(["request_id", "event_id", "side_id"]):
        p = g.p_book.to_numpy()
        out.append(pd.Series([np.nanmedian(np.delete(p, i)) if np.isfinite(np.delete(p, i)).any() else np.nan
                              for i in range(len(p))], index=g.index))
    return pd.concat(out)


def build(stake):
    d = load_sgo()
    d["p_consensus"] = consensus(d)
    d = d[d.bookmaker_id.isin(IL_BOOKS)].copy()
    games = d.drop_duplicates("event_id")[["event_id", "starts_at_utc", "home", "away"]]
    cand = {}
    for g in games.itertuples():
        for team in (g.home, g.away):
            cand[(g.event_id, team)] = candles(ticker(g.starts_at_utc, g.away, g.home, team), g.starts_at_utc)
    opp = lambda r: r.away if r.team == r.home else r.home
    d["p_kalshi"] = [kalshi_fair(cand[(r.event_id, r.team)], cand[(r.event_id, opp(r))], int(r.captured_at_utc.timestamp()))
                     for r in d.itertuples()]
    d["p_close"] = [kalshi_fair(cand[(r.event_id, r.team)], cand[(r.event_id, opp(r))], int(r.starts_at_utc.timestamp()))
                    for r in d.itertuples()]
    d["dec"] = decimal(d.odds_american)
    d["fee"] = [fee(b, stake) for b in d.bookmaker_id]
    d["ev_kalshi"] = d.p_kalshi * d.dec - 1
    d["ev_net"] = d.ev_kalshi - d.fee / stake
    d["ev_consensus"] = d.p_consensus * d.dec - 1
    d["clv"] = d.p_close * d.dec - 1
    d["price_age_min"] = (d.captured_at_utc - d.last_updated_at).dt.total_seconds() / 60
    d["slot_kind"] = d.slot.str.split("@").str[0]
    cols = ["captured_at_utc", "slot", "slot_kind", "starts_at_utc", "minutes_to_tip", "event_id", "away", "home",
            "team", "bookmaker_id", "odds_american", "dec", "last_updated_at", "price_age_min", "p_kalshi",
            "p_consensus", "p_close", "ev_kalshi", "fee", "ev_net", "ev_consensus", "clv"]
    OUT.mkdir(parents=True, exist_ok=True)
    d[cols].sort_values(["captured_at_utc", "event_id", "team", "bookmaker_id"]).to_csv(OUT / "ml_ledger.csv", index=False)
    return d[cols]


def summary(d, stake):
    k = d[d.p_kalshi.notna()]
    print(f"{len(d)} Illinois-book prices, {d.event_id.nunique()} games, {d.captured_at_utc.nunique()} snapshots "
          f"({d.captured_at_utc.min():%m-%d} to {d.captured_at_utc.max():%m-%d}); with a Kalshi fair price: {len(k)}")
    print("\nMean EV vs Kalshi by book (price quality; negative = the book's margin):")
    print(k.groupby("bookmaker_id").agg(n=("ev_kalshi", "size"), mean_ev=("ev_kalshi", "mean"),
                                        mean_clv=("clv", "mean")).round(4).to_string())
    best = k.loc[k.groupby(["captured_at_utc", "event_id", "team"]).ev_kalshi.idxmax()]
    print(f"\nBest Illinois price per (snapshot, game, side): n={len(best)}, by slot kind")
    rows = []
    for kind, g in best.groupby("slot_kind"):
        rows.append({"slot": kind, "n": len(g), **{f">={t:.0%}": (g.ev_net >= t).mean() for t in (0.0, 0.01, 0.02, 0.03)},
                     "mean_ev_net": g.ev_net.mean()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    opp = best[best.ev_net >= 0.02]
    if len(opp):
        print(f"\nOpportunities (net EV >= 2% at ${stake:.0f}): n={len(opp)}; books {opp.bookmaker_id.value_counts().to_dict()}; "
              f"mean CLV {opp.clv.mean():+.4f}; mean price age {opp.price_age_min.mean():.1f} min")
    print("\nPreliminary and NOT a pre-registered test: frequency only, few days of data.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stake", type=float, default=25.0, help="stake for per-bet fees (default $25)")
    a = ap.parse_args()
    summary(build(a.stake), a.stake)
