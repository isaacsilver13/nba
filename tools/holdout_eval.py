"""One-shot evaluation of a frozen model on a held-out period (spread market). See docs/plans/2026-11-03-holdout-2025-26-prereg.md.

    python tools/holdout_eval.py --preds outputs/oos_frozen.csv            # DEV slice only (dry run, any number of times)
    python tools/holdout_eval.py --preds outputs/oos_frozen.csv --final    # score the holdout ONCE; writes a lock file
    python tools/holdout_eval.py --selftest

The holdout slice is never printed without --final, and --final refuses to run twice (outputs/holdout_eval_LOCK.json;
deleting it is a deliberate act that the prereg says must be recorded). preds = experiment.py's walk-forward OOS table, so
every holdout prediction comes from a model trained only on earlier games.

Primary statistic (pre-registered): encompassing slope of the market's error on the model's edge, one-sided H1 slope > 0.
Secondary: flat 1-unit ROI on |edge| >= EDGE_PTS bets (matched odds only, pushes refunded, settled on settle_signed).
Reference: the same ROI under SHUFFLED edges (the noise scale), and the minimum detectable slope at 80% power.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encompass_check import ols  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPLIT = "2025-10-01"   # holdout = games on/after this date (the 2025-26 season)
EDGE_PTS = 2.0         # pre-declared single bet threshold, in points of |model margin - market margin|
LOCK = ROOT / "outputs" / "holdout_eval_LOCK.json"


def bets(d, edge_pts=EDGE_PTS):
    """Per-game flat-stake return for |edge| >= edge_pts: edge = pred + spread_signed (>0 = bet home), home covers if margin > -settle."""
    edge = (d.pred_ensemble + d.spread_signed).values
    settle = d.settle_signed.fillna(d.spread_signed).values
    margin = d.home_margin.values
    side_home = edge > 0
    cover_home, push = margin > -settle, margin == -settle
    won = np.where(side_home, cover_home, ~cover_home & ~push)
    ret = np.where(push, 0.0, np.where(won, np.where(side_home, d.payout_home, d.payout_away), -1.0))
    sel = (np.abs(edge) >= edge_pts) & d.odds_matched.astype(bool).values
    return ret, sel, edge


def slice_report(d, label, n_shuffle=300):
    d = d.dropna(subset=["home_margin", "spread_signed", "pred_ensemble"]).reset_index(drop=True)
    G = pd.to_datetime(d.GAME_DATE).dt.date.values
    mkt = -d.spread_signed.values
    err, edge = d.home_margin.values - mkt, d.pred_ensemble.values - mkt
    b, se, _ = ols(err, edge, G)
    z = b[1] / se[1]
    sq = (d.pred_ensemble.values - d.home_margin.values) ** 2 - (mkt - d.home_margin.values) ** 2   # >0 = model worse than the line
    bd, sed, _ = ols(sq, None, G)
    ret, sel, _ = bets(d)
    n = int(sel.sum())
    roi, roi_se = (ret[sel].mean(), ret[sel].std(ddof=1) / np.sqrt(n)) if n > 1 else (float("nan"),) * 2
    rng = np.random.default_rng(0)
    null = []
    for _ in range(n_shuffle):                      # same bets, edge sides shuffled: what pure noise earns after the vig
        dd = d.assign(pred_ensemble=rng.permutation(d.pred_ensemble.values - mkt) + mkt)
        r, s, _ = bets(dd)
        null.append(r[s].mean() if s.sum() > 1 else np.nan)
    print(f"[{label}] games={len(d)}  {d.GAME_DATE.min():%Y-%m-%d}..{d.GAME_DATE.max():%Y-%m-%d}")
    print(f"  RMSE line={np.sqrt(np.mean((mkt - d.home_margin) ** 2)):.3f} model={np.sqrt(np.mean((d.pred_ensemble - d.home_margin) ** 2)):.3f}  "
          f"MSE(model)-MSE(line)={bd[0]:+.2f} (se {sed[0]:.2f})")
    print(f"  PRIMARY encompassing slope={b[1]:+.3f} (se {se[1]:.3f})  one-sided p(slope>0)={norm.sf(z):.3f}   "
          f"min detectable slope @80% power = {2.49 * se[1]:.2f}")
    print(f"  secondary: |edge|>={EDGE_PTS}pt bets n={n}  flat ROI={roi:+.4f} (se {roi_se:.4f})   shuffled-edge ROI mean {np.nanmean(null):+.4f} sd {np.nanstd(null):.4f}")
    return dict(slope=float(b[1]), se=float(se[1]), p=float(norm.sf(z)), n_bets=n, roi=float(roi))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selftest():
    rng = np.random.default_rng(3)
    n = 4000
    line = rng.normal(0, 7, n)                                   # market margin
    true_dev = rng.normal(0, 3, n)
    margin = line + 0.5 * true_dev + rng.normal(0, 12, n)        # half of the model's deviation is real
    base = pd.DataFrame({"GAME_DATE": pd.date_range("2024-01-01", periods=n // 5).repeat(5)[:n], "home_margin": margin,
                         "spread_signed": -line, "settle_signed": -line, "payout_home": 0.909, "payout_away": 0.909, "odds_matched": True})
    good = slice_report(base.assign(pred_ensemble=line + true_dev), "selftest real", 20)
    junk = slice_report(base.assign(pred_ensemble=line + rng.permutation(true_dev)), "selftest junk", 20)
    assert 0.35 < good["slope"] < 0.65 and good["p"] < 0.001, good
    assert abs(junk["slope"]) < 0.15 and junk["roi"] < good["roi"], (junk, good)
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preds")
    ap.add_argument("--final", action="store_true", help="score the holdout slice (once; writes the lock file)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit()
    d = pd.read_csv(a.preds, parse_dates=["GAME_DATE"])
    dev, hold = d[d.GAME_DATE < SPLIT], d[d.GAME_DATE >= SPLIT]
    slice_report(dev, f"DEV < {SPLIT}")
    if not a.final:
        print(f"\n[holdout] {len(hold)} games on/after {SPLIT} NOT scored (use --final once).")
        sys.exit()
    if LOCK.exists():
        sys.exit(f"[holdout] REFUSED: already evaluated ({LOCK.read_text()[:200]}...). Delete the lock only to record a deliberate re-look.")
    res = slice_report(hold, f"HOLDOUT >= {SPLIT}")
    LOCK.write_text(json.dumps(dict(when=datetime.now(timezone.utc).isoformat(), preds_sha256=sha(a.preds),
                                    experiment_sha256=sha(ROOT / "experiment.py"), split=SPLIT, edge_pts=EDGE_PTS, **res), indent=1))
    print(f"[holdout] lock written: {LOCK}")
