"""Stage 1: market-anchored margin/total curves from closing-line residuals (docs/curves-stage1-preregistration.md).

    python tools/curve_calibration.py                                   # development seasons 2012-2022, picks B or C
    python tools/curve_calibration.py --holdout --pick-margin C --pick-total C   # 2023-2025, run once after the pick

Every curve is centered on the closing line (anchor w = 0). A: one fixed normal sd. B: normal, log sd linear in the
closing total and |spread|. C: B's sd with the empirical shape of standardized training residuals. Walk-forward by
season on an expanding window. Scored by RPS over integer outcomes and by calibration of alt rungs L + d.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NBA_DATA_DIR", ROOT / "data"))
GRID = np.arange(-70, 71)  # integer outcomes floor(L) + GRID cover every observed residual
OFFSETS = np.arange(-15, 16)
DEV, HOLDOUT = range(2012, 2023), range(2023, 2026)


def load(path=None):
    d = pd.read_csv(path or DATA / "odds" / "nba_2008-2025.csv").dropna(subset=["spread"])  # |spread| is in B's sd
    d["margin"] = d.score_home - d.score_away
    d["points"] = d.score_home + d.score_away
    d["line_m"] = np.where(d.whos_favored == "home", d.spread, -d.spread)
    d["line_t"] = d.total
    return d


def design(df):
    return np.column_stack([np.ones(len(df)), (df.line_t - 210) / 10, df.line_m.abs() / 5])


def fit_scale(resid, X):
    """Normal MLE of log sd = X @ b. X with one column gives model A."""
    nll = lambda b: np.sum(X @ b) + 0.5 * np.sum((resid / np.exp(X @ b)) ** 2)
    b0 = np.zeros(X.shape[1]); b0[0] = np.log(resid.std())
    return minimize(nll, b0, method="BFGS").x


def pmf(line, sd, std_cdf, no_tie=False):
    """P(X = x) on x = floor(line) + GRID for each game; std_cdf maps standardized residuals to probabilities."""
    xs = np.floor(line)[:, None] + GRID
    cdf = std_cdf((xs + 0.5 - line[:, None]) / sd[:, None]) - std_cdf((xs - 0.5 - line[:, None]) / sd[:, None])
    if no_tie:
        cdf[xs == 0] = 0.0
    return xs, cdf / cdf.sum(1, keepdims=True)


def rps(xs, p, y):
    return np.sum((np.cumsum(p, 1) - (y[:, None] <= xs)) ** 2, 1)


def rung_probs(xs, p, line, d):
    """P(X > L + d | no push) per game; NaN-free because a push removes at most one grid point."""
    k = line + d
    over = (p * (xs > k[:, None])).sum(1)
    push = (p * (xs == k[:, None])).sum(1)
    return over / (1 - push)


def std_ecdf(train, col, line_col):
    """Curve C's shape: CDF of training residuals standardized by model B's sd. Returns (cdf, B's coefficients)."""
    r_tr = (train[col] - train[line_col]).to_numpy(float)
    bB = fit_scale(r_tr, design(train))
    z = np.sort(r_tr / np.exp(design(train) @ bB))
    return (lambda u: np.searchsorted(z, u, side="right") / len(z)), bB


def predict(train, test, col, line_col, no_tie):
    r_tr = (train[col] - train[line_col]).to_numpy(float)
    line = test[line_col].to_numpy(float)
    bA = fit_scale(r_tr, np.ones((len(train), 1)))
    ecdf, bB = std_ecdf(train, col, line_col)
    sdA = np.full(len(test), np.exp(bA[0]))
    sdB = np.exp(design(test) @ bB)
    return {
        "A": pmf(line, sdA, norm.cdf, no_tie),
        "B": pmf(line, sdB, norm.cdf, no_tie),
        "C": pmf(line, sdB, ecdf, no_tie),
    }, bB


def walk_forward(d, col, line_col, seasons, no_tie):
    rows, rung_rows = [], []
    for s in seasons:
        train, test = d[d.season < s], d[d.season == s]
        preds, bB = predict(train, test, col, line_col, no_tie)
        y = test[col].to_numpy(float)
        line = test[line_col].to_numpy(float)
        out = pd.DataFrame({"season": s, "game": test.index})
        for m, (xs, p) in preds.items():
            out[f"rps_{m}"] = rps(xs, p, y)
            out[f"pwin_{m}"] = (p * (xs > 0)).sum(1)
            for off in OFFSETS:
                k = line + off
                rung_rows.append(pd.DataFrame({"season": s, "model": m, "d": off,
                                               "p": rung_probs(xs, p, line, off),
                                               "y": np.where(y > k, 1.0, np.where(y < k, 0.0, np.nan))}))
        rows.append(out)
    return pd.concat(rows, ignore_index=True), pd.concat(rung_rows, ignore_index=True), bB


def calibration(rungs, model):
    r = rungs[(rungs.model == model) & rungs.y.notna()]
    g = r.groupby("d").agg(n=("y", "size"), pred=("p", "mean"), obs=("y", "mean"),
                           var=("p", lambda p: np.sum(p * (1 - p))))
    g["z"] = (g.obs - g.pred) * g.n / np.sqrt(g["var"])
    return g.drop(columns="var")


def room(rungs, pick):
    r = rungs[rungs.d.abs().between(3, 10)]
    a = r[r.model == "A"].p.to_numpy()
    b = r[r.model == pick].p.to_numpy()
    diff = np.abs(b - a) * 100
    return np.median(diff), np.quantile(diff, 0.9)


def bootstrap_gain(scores, pick, n=2000, seed=0):
    diff = (scores.rps_A - scores[f"rps_{pick}"]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), (n, len(diff)))].mean(1)
    return diff.mean(), np.quantile(boots, [0.05, 0.95])


def moneyline_check(d, scores_games, xs_p_by_season):
    """Brier of P(home win) from the margin curve vs the de-vigged closing moneyline."""
    dec = lambda a: np.where(a > 0, 1 + a / 100, 1 + 100 / -a)
    sub = d.loc[scores_games].copy()
    ok = sub.moneyline_home.notna() & sub.moneyline_away.notna() & (sub.moneyline_home != 0)
    ih, ia = 1 / dec(sub.moneyline_home), 1 / dec(sub.moneyline_away)
    sub["p_ml"] = ih / (ih + ia)
    sub["p_curve"] = xs_p_by_season
    sub = sub[ok]
    win = (sub.margin > 0).astype(float)
    return len(sub), np.mean((sub.p_ml - win) ** 2), np.mean((sub.p_curve - win) ** 2)


def report(d, col, line_col, seasons, no_tie, pick, holdout):
    name = "margin" if no_tie else "total"
    dd = d.dropna(subset=[line_col])
    scores, rungs, bB = walk_forward(dd, col, line_col, seasons, no_tie)
    resid = (dd[col] - dd[line_col])[dd.season.isin(seasons)]
    print(f"\n## {name} ({'holdout' if holdout else 'development'}, seasons {seasons.start}-{seasons.stop - 1}, "
          f"n={len(scores)})")
    print(f"Residual mean {resid.mean():+.3f} (se {resid.std() / np.sqrt(len(resid)):.3f}), sd {resid.std():.2f}. "
          f"Last fit log-sd coefs (B): {np.round(bB, 4).tolist()}")
    by = scores.groupby("season")[["rps_A", "rps_B", "rps_C"]].mean()
    print(by.round(4).to_string())
    print("pooled", scores[["rps_A", "rps_B", "rps_C"]].mean().round(4).to_dict())
    if pick is None:
        pick = min("BC", key=lambda m: scores[f"rps_{m}"].mean())
        print(f"Pick (lower mean RPS of B, C): {pick}")
    gain, ci = bootstrap_gain(scores, pick)
    cal = calibration(rungs, pick)
    med, p90 = room(rungs, pick)
    print(f"RPS gain of {pick} over A: {gain:.4f}, 90% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"Rung calibration ({pick}); max |z| {cal.z.abs().max():.2f}")
    print(cal.round(4).to_string())
    print(f"Calibration of A for comparison: max |z| {calibration(rungs, 'A').z.abs().max():.2f}")
    print(f"Room |p_{pick} - p_A| on |d| 3..10: median {med:.2f} pp, 90th pct {p90:.2f} pp")
    if holdout:
        print(f"H1 {'PASS' if ci[0] > 0 else 'FAIL'}; calibration {'PASS' if cal.z.abs().max() <= 3 else 'FAIL'}; "
              f"room {'PASS' if p90 >= 2 else 'FAIL'}")
    if no_tie:
        n, b_ml, b_curve = moneyline_check(dd, scores.game.to_numpy(), scores[f"pwin_{pick}"].to_numpy())
        print(f"P(home win), n={n}: Brier closing ML {b_ml:.4f} vs curve {b_curve:.4f}")
    return pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--pick-margin", choices=["B", "C"])
    ap.add_argument("--pick-total", choices=["B", "C"])
    a = ap.parse_args()
    if a.holdout and not (a.pick_margin and a.pick_total):
        ap.error("--holdout needs the locked --pick-margin and --pick-total from the development run")
    d = load()
    seasons = HOLDOUT if a.holdout else DEV
    report(d, "margin", "line_m", seasons, True, a.pick_margin, a.holdout)
    report(d, "points", "line_t", seasons, False, a.pick_total, a.holdout)


if __name__ == "__main__":
    main()
