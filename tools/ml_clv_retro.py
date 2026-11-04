"""Stage 2 part A: early-line moneyline CLV on 2025-26 Kalshi game-winner prices (docs/stage2-preregistration.md).

    python tools/ml_clv_retro.py preds     # line-free walk-forward + win probabilities -> outputs/stage2/noline_oos.csv
    python tools/ml_clv_retro.py prices    # Kalshi E06/E10/close quotes per game -> outputs/stage2/prices.csv; n bets
    python tools/ml_clv_retro.py score     # the one pre-registered scoring run

Needs `python tools/kalshi_history.py KXNBAGAME --candles --hours 24 --name day` first. The margin model is
predict_today's (raw home_margin target, no closing line): experiment.py's own predictions add the closing spread
back, so they must not be used here.
"""
import argparse
import json
import os
import sys
import warnings
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import curve_calibration as cc  # noqa: E402
from encompass_check import fit_logistic, fit_probit_scale, ols  # noqa: E402
from ingest_daily import tricode  # noqa: E402

DATA = Path(os.environ.get("NBA_DATA_DIR", ROOT / "data"))
OUT = Path(os.environ.get("NBA_OUTPUTS_DIR", ROOT / "outputs")) / "stage2"
CT = ZoneInfo("America/Chicago")
SEASON = 2026  # 2025-26 in the games table's end-year convention
ENTRIES = {"E06": time(6), "E10": time(10)}
MIN_HIST, MAX_SPREAD, MIN_LEAD_S, EV_MIN = 250, 0.05, 1800, 0.03


def p_curve(pred, s, G):
    """P(home margin > 0) under curve C centered on pred with scale s (ties impossible)."""
    xs, p = cc.pmf(np.asarray(pred, float), np.broadcast_to(np.asarray(s, float), np.shape(pred)).copy(), G, no_tie=True)
    return (p * (xs > 0)).sum(1)


def fit_curve_scale(x, y, G):
    nll = lambda s: -np.mean(y * np.log(np.clip(p_curve(x, s, G), 1e-9, 1))
                             + (1 - y) * np.log(np.clip(1 - p_curve(x, s, G), 1e-9, 1)))
    return minimize_scalar(nll, bounds=(3, 60), method="bounded").x


def win_probs(d, G):
    """Per fold, scales fit on out-of-sample games of earlier folds only (>= MIN_HIST games)."""
    for c in ("p_curve", "p_phi", "p_elo", "s_curve"):
        d[c] = np.nan
    for f in sorted(d.fold.unique()):
        h, t = (d.fold < f) & d.pred.notna(), d.fold == f
        if h.sum() < MIN_HIST:
            continue
        x, y = d.loc[h, "pred"].to_numpy(), (d.loc[h, "home_margin"] > 0).to_numpy(float)
        s = fit_curve_scale(x, y, G)
        d.loc[t, "s_curve"] = s
        d.loc[t, "p_curve"] = p_curve(d.loc[t, "pred"], s, G)
        d.loc[t, "p_phi"] = fit_probit_scale(x, y, d.loc[t, "pred"].to_numpy())
        e = h & d.elo_diff.notna()
        d.loc[t, "p_elo"] = fit_logistic(d.loc[e, "elo_diff"].to_numpy(), (d.loc[e, "home_margin"] > 0).to_numpy(float),
                                         d.loc[t, "elo_diff"].to_numpy())
    return d


def build_preds():
    warnings.simplefilter("ignore")
    import experiment as ex
    import predict_today as pt
    h = pt.load_history()
    features = [f for f in ex.build_features(h) if f not in pt.EXCLUDE]
    h[features] = h[features].apply(pd.to_numeric, errors="coerce")
    oos, _ = pt.walk_forward(h, features)
    fold = pd.Series(np.nan, index=h.index)
    for k, (_, test) in enumerate(ex.walk_forward_splits(h, "GAME_DATE", ex.TRAIN_SIZE, ex.TEST_SIZE), 1):
        fold[test.index] = k
    d = h[["GAME_ID", "GAME_DATE", "season", "home", "away", "home_margin", "elo_diff"]].assign(pred=oos, fold=fold)
    d = d[d.fold.notna()].copy()
    d["elo_diff"] = pd.to_numeric(d.elo_diff, errors="coerce")
    G, _ = cc.std_ecdf(cc.load(), "margin", "line_m")  # stage 1 shape, seasons 2008-2025 (all before 2025-26)
    d = win_probs(d, G)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "noline_oos.csv", index=False)
    s = d[d.season == SEASON]
    print(f"walk-forward: {len(d)} OOS games in {int(d.fold.max())} folds; 2025-26: {len(s)} games, "
          f"RMSE {np.sqrt(np.mean((s.home_margin - s.pred) ** 2)):.2f}, s_curve {s.s_curve.min():.2f}-{s.s_curve.max():.2f}, "
          f"max |p_curve - p_phi| {np.abs(s.p_curve - s.p_phi).max():.4f} -> {OUT / 'noline_oos.csv'}")


def espn_tips():
    """{(date, home, away): scheduled tip (UTC epoch s)} from cached ESPN scoreboards."""
    tips = {}
    for f in (DATA / "cache" / "espn" / "scoreboard").glob("*.json"):
        day = pd.Timestamp(f.stem)
        for e in json.loads(f.read_text(encoding="utf-8")).get("events", []):
            side = {t["homeAway"]: t["team"].get("abbreviation") for t in e["competitions"][0]["competitors"]}
            try:
                key = (day, tricode(side["home"]), tricode(side["away"]))
            except (ValueError, KeyError):
                continue  # exhibition opponents, or TBD teams (no abbreviation)
            tips[key] = int(datetime.fromisoformat(e["date"].replace("Z", "+00:00")).timestamp())
    return tips


def parse_ticker(t):
    """KXNBAGAME-26JAN15MILSAS-SAS -> (2026-01-15, away MIL, home SAS, team SAS)."""
    _, ev, team = t.split("-")
    return pd.Timestamp(datetime.strptime(ev[:7], "%y%b%d")), ev[7:10], ev[10:13], team


def quotes(c, ts):
    """(bid, ask) of the candle ending at ts, or None when missing or failing the quote filter."""
    r = c.get(ts)
    if r is None or not (0 < r[0] < r[1] < 1) or r[1] - r[0] > MAX_SPREAD + 1e-9:
        return None
    return r


def build_prices():
    markets = [json.loads(line) for line in open(DATA / "kalshi" / "KXNBAGAME_markets.jsonl")]
    cand = pd.read_csv(DATA / "kalshi" / "KXNBAGAME_day.csv").drop_duplicates(["ticker", "end_ts"])
    cand = cand.dropna(subset=["yes_bid", "yes_ask"])
    for col in ("yes_bid", "yes_ask"):
        cand[col] = cand[col].astype(float).where(cand[col].astype(float) <= 1, cand[col].astype(float) / 100)
    by_ticker = {t: dict(zip(g.end_ts.astype(int), zip(g.yes_bid, g.yes_ask))) for t, g in cand.groupby("ticker")}
    espn = espn_tips()

    games = {}
    for m in markets:
        day, away, home, team = parse_ticker(m["ticker"])
        kalshi_tip = int(datetime.fromisoformat(m["expected_expiration_time"].replace("Z", "+00:00")).timestamp()) - 3 * 3600
        g = games.setdefault((day, home, away), {"kalshi_tip": kalshi_tip})
        g["home" if team == home else "away"] = by_ticker.get(m["ticker"], {})

    both = [(k, g["kalshi_tip"]) for k, g in games.items() if k in espn]
    off = np.array([abs(kt - espn[k]) for k, kt in both])
    use_kalshi = (off > 900).mean() <= 0.05
    print(f"tip check on {len(both)} games with both sources: |Kalshi - ESPN| > 15 min on {(off > 900).mean():.1%} "
          f"-> Kalshi tips {'used as fallback' if use_kalshi else 'NOT used'}")

    rows = []
    for (day, home, away), g in games.items():
        tip = espn.get((day, home, away), g["kalshi_tip"] if use_kalshi else None)
        if tip is None or "home" not in g or "away" not in g:
            continue
        row = {"GAME_DATE": day, "home": home, "away": away, "tip_ts": tip}
        close_ts = max((ts for ts in g["home"] if ts <= tip), default=None)
        close = (quotes(g["home"], close_ts), quotes(g["away"], close_ts)) if close_ts else (None, None)
        if None not in close:
            row.update(close_ts=close_ts, mid_close_home=sum(close[0]) / 2, mid_close_away=sum(close[1]) / 2)
        for name, t in ENTRIES.items():
            ts = int(datetime.combine(day.date(), t, CT).timestamp())
            q = (quotes(g["home"], ts), quotes(g["away"], ts))
            if ts <= tip - MIN_LEAD_S and None not in q and None not in close:
                row.update({f"{name}_bid_home": q[0][0], f"{name}_ask_home": q[0][1],
                            f"{name}_bid_away": q[1][0], f"{name}_ask_away": q[1][1]})
        rows.append(row)
    prices = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    prices.to_csv(OUT / "prices.csv", index=False)

    preds = pd.read_csv(OUT / "noline_oos.csv", parse_dates=["GAME_DATE"])
    preds = preds[(preds.season == SEASON) & preds.p_curve.notna()]
    d = preds.merge(prices, on=["GAME_DATE", "home", "away"], how="left")
    print(f"2025-26 regular-season games with a model probability: {len(preds)}; with both Kalshi markets: "
          f"{d.tip_ts.notna().sum()}; with a valid close: {d.mid_close_home.notna().sum()}")
    for name in ENTRIES:
        ok = d[f"{name}_ask_home"].notna()
        ev = np.maximum(d.p_curve / d[f"{name}_ask_home"], (1 - d.p_curve) / d[f"{name}_ask_away"]) - 1
        print(f"{name}: priced games {ok.sum()}, bets (EV >= {EV_MIN:.0%}, curve C) {(ok & (ev >= EV_MIN)).sum()}")


logit = lambda p: np.log(p / (1 - p))


def bets(d, entry, pcol):
    """The pre-registered bet rule: larger-EV side if EV >= 3% at the ask; CLV vs the close mid, gross and net of fee."""
    d = d[d[f"{entry}_ask_home"].notna() & d[pcol].notna()]
    ev_h = d[pcol] / d[f"{entry}_ask_home"] - 1
    ev_a = (1 - d[pcol]) / d[f"{entry}_ask_away"] - 1
    home = ev_h >= ev_a
    ask = np.where(home, d[f"{entry}_ask_home"], d[f"{entry}_ask_away"])
    mid = np.where(home, d.mid_close_home, d.mid_close_away)
    won = np.where(home, d.home_margin > 0, d.home_margin < 0)
    fee = 0.07 * ask * (1 - ask)
    b = pd.DataFrame({"date": d.GAME_DATE.to_numpy(), "ev": np.maximum(ev_h, ev_a).to_numpy(),
                      "clv": mid / ask - 1, "clv_net": mid / (ask + fee) - 1,
                      "roi": np.where(won, 1 / ask - 1, -1.0)})
    return b[b.ev >= EV_MIN]


def mean_se(x, groups):
    b, se, _ = ols(np.asarray(x, float), None, groups)
    return b[0], se[0]


def fmt(m, se):
    return f"{m:+.4f} (se {se:.4f}, 90% CI [{m - 1.645 * se:+.4f}, {m + 1.645 * se:+.4f}])"


def score():
    preds = pd.read_csv(OUT / "noline_oos.csv", parse_dates=["GAME_DATE"])
    preds = preds[(preds.season == SEASON) & preds.p_curve.notna()]
    d = preds.merge(pd.read_csv(OUT / "prices.csv", parse_dates=["GAME_DATE"]), on=["GAME_DATE", "home", "away"])

    print("## Primary: mean gross CLV of E06 bets (curve C)")
    b = bets(d, "E06", "p_curve")
    m, se = mean_se(b.clv, b.date)
    verdict = "INCONCLUSIVE (n < 150)" if len(b) < 150 else ("PASS" if m - 1.645 * se > 0 else "FAIL")
    print(f"n={len(b)}  CLV {fmt(m, se)}  -> {verdict}")

    print("\n## Secondary")
    for entry in ENTRIES:
        for pcol in ("p_curve", "p_phi", "p_elo"):
            b = bets(d, entry, pcol)
            print(f"{entry} {pcol:8s} n={len(b):4d}  CLV {fmt(*mean_se(b.clv, b.date))}  "
                  f"net {mean_se(b.clv_net, b.date)[0]:+.4f}  ROI {fmt(*mean_se(b.roi, b.date))}")
    for entry in ENTRIES:
        g = d[d[f"{entry}_ask_home"].notna()]
        mid_e = (g[f"{entry}_bid_home"] + g[f"{entry}_ask_home"]) / 2
        coef, se, p = ols(logit(g.mid_close_home) - logit(mid_e), logit(g.p_curve) - logit(mid_e), g.GAME_DATE)
        print(f"{entry} CLV slope (n={len(g)}): {coef[1]:+.4f} (se {se[1]:.4f}, p {p[1]:.3g}); intercept {coef[0]:+.4f}")
    y = (preds.home_margin > 0).astype(float)
    for pcol in ("p_curve", "p_phi"):
        p = preds[pcol].clip(1e-9, 1 - 1e-9)
        print(f"calibration {pcol}: Brier {np.mean((p - y) ** 2):.4f}, log loss {-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)):.4f} "
              f"(n={len(preds)})")

    print("\n## Not pre-registered (context)")
    g = d[d.E06_ask_home.notna()]
    yw = (g.home_margin > 0).astype(float)
    for name, p in (("model p_curve", g.p_curve), ("Kalshi E06 mid", (g.E06_bid_home + g.E06_ask_home) / 2),
                    ("Kalshi close mid", g.mid_close_home)):
        print(f"Brier on E06-priced games (n={len(g)}): {name} {np.mean((p - yw) ** 2):.4f}")
    b = bets(d, "E06", "p_curve")
    for lo, hi in ((0.03, 0.06), (0.06, 0.10), (0.10, 0.20), (0.20, 10)):
        s = b[(b.ev >= lo) & (b.ev < hi)]
        if len(s):
            print(f"E06 EV {lo:.0%}-{hi:.0%}: n={len(s)}, CLV {s.clv.mean():+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["preds", "prices", "score"])
    {"preds": build_preds, "prices": build_prices, "score": score}[ap.parse_args().step]()
