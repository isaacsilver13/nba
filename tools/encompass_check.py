"""Forecast-encompassing and CLV checks on experiment.py's out-of-sample predictions (experiment.py is not edited).

    python tools/encompass_check.py                                   # run experiment.py, cache preds, report
    python tools/encompass_check.py --preds outputs/oos_preds.csv     # reuse cached preds
    python tools/encompass_check.py --exclude spread_signed,is_home_favorite --out outputs/oos_noline.csv
    python tools/encompass_check.py --selftest

Spread: regress the closing line's error on the model's edge (slope 0 = no information beyond the close).
Moneyline (game_odds: lake table, else csv/game_odds.csv; TeamRankings): win prob p = Phi(pred / s), s fit on EARLIER folds' outcomes; de-vigged
open (first snapshot, ~night before), game-day 10am ET, and close. Reports the CLV slope (logit move to close ~ logit
model gap), mean CLV and ROI of EV>3% bets at each entry, encompassing vs the close, and an Elo-only placebo.
experiment.py feeds the CLOSING spread to the model, so CLV vs earlier prices is look-ahead unless you --exclude it.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ingest import datalake  # noqa: E402

DATA = Path(os.environ.get("NBA_DATA_DIR", ROOT / "data"))
logit = lambda p: np.log(p / (1 - p))


def ols(y, X, groups=None):
    """OLS with intercept; cluster-robust SEs by `groups`, else HC1. X may be None (mean only). Returns (b, se, p)."""
    X = np.column_stack([np.ones(len(y))] + ([np.asarray(X, float).reshape(len(y), -1)] if X is not None else []))
    y = np.asarray(y, float)
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    if groups is None:
        meat = (X * u[:, None] ** 2).T @ X * len(y) / (len(y) - X.shape[1])
    else:
        idx = pd.Series(range(len(y))).groupby(np.asarray(groups)).indices
        S = np.array([(X[i] * u[i, None]).sum(0) for i in idx.values()])
        meat = S.T @ S
    se = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))
    return b, se, 2 * norm.sf(np.abs(b / se))


def devig(dec_home, dec_away):
    """Multiplicative de-vig of decimal odds -> (home prob, overround)."""
    ih, ia = 1 / dec_home, 1 / dec_away
    return ih / (ih + ia), ih + ia


def capture(exclude, out):
    """Run experiment.run_experiment() with extra feature exclusions; save its OOS table with GAME_ID_KEY and elo_diff."""
    sys.path.insert(0, str(ROOT))
    import experiment as ex
    ex.EXTRA_FEATURE_EXCLUSIONS = list(ex.EXTRA_FEATURE_EXCLUSIONS) + exclude
    cap, orig = {}, ex.compute_roi
    ex.compute_roi = lambda p, detail=False: (cap.setdefault("df", p.copy()), orig(p, detail=detail))[1]
    print(ex.run_experiment())
    p, df = cap["df"], ex.load_data()  # run_experiment's table keeps load_data's row index
    assert np.array_equal(df.loc[p.index, "home_margin"].values, p.home_margin.values), "row alignment broke"
    p["GAME_ID_KEY"] = df.loc[p.index, "GAME_ID_KEY"].values
    if "elo_diff" in df:
        p["elo_diff"] = pd.to_numeric(df.loc[p.index, "elo_diff"], errors="coerce").values
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    p.to_csv(out, index=False)
    return out


def spread_report(d):
    d = d.dropna(subset=["home_margin", "spread_signed", "pred_ensemble"]).copy()
    d["market"] = -d.spread_signed
    d["mkt_err"] = d.home_margin - d.market
    d["edge"] = d.pred_ensemble - d.market
    rm = lambda e: float(np.sqrt(np.mean(e ** 2)))
    for label, x in (("all OOS", d), ("top 20% |edge|", d[d.edge.abs() >= d.edge.abs().quantile(0.8)])):
        b, se, p = ols(x.mkt_err, x.edge, x.GAME_DATE.values)
        print(f"[spread] {label:15s} n={len(x):>5} RMSE line={rm(x.mkt_err):.3f} model={rm(x.home_margin - x.pred_ensemble):.3f}  "
              f"error~edge slope={b[1]:+.3f} (se {se[1]:.3f}, p={p[1]:.3f})")


def win_probs(d, col, fit):
    """Walk-forward: for each fold, fit on folds < f (needs >= 250 rows), predict fold f."""
    out = pd.Series(np.nan, index=d.index)
    for f in sorted(d.fold.unique()):
        h, t = d.fold < f, d.fold == f
        if h.sum() >= 250:
            out[t] = fit(d.loc[h, col].values, d.loc[h, "win"].values, d.loc[t, col].values)
    return out


def fit_probit_scale(x, y, xt):
    s = minimize(lambda s: -np.mean(y * norm.logcdf(x / s[0]) + (1 - y) * norm.logsf(x / s[0])), [14.0], bounds=[(3, 60)]).x[0]
    return norm.cdf(xt / s)


def fit_logistic(x, y, xt):
    c = minimize(lambda c: np.mean(np.logaddexp(0, -(c[0] + c[1] * x)) * y + np.logaddexp(0, c[0] + c[1] * x) * (1 - y)), [0.0, 0.003]).x
    return 1 / (1 + np.exp(-(c[0] + c[1] * xt)))


def load_odds(path=None):
    """(game_odds frame, label). An explicit CSV path wins; otherwise the lake's game_odds table, then
    csv/game_odds.csv. (None, None) when there is no source."""
    if path:
        p = Path(path)
        return (pd.read_csv(p, dtype={"game_id": str}), p.name) if p.exists() else (None, None)
    if "game_odds" in datalake.tables():
        return datalake.read_pandas("game_odds"), "lake/game_odds"
    p = DATA / "csv" / "game_odds.csv"
    return (pd.read_csv(p, dtype={"game_id": str}), p.name) if p.exists() else (None, None)


def moneyline_report(d, odds):
    g = odds.copy()
    g["gid"] = g.game_id.astype(int)
    g = g[g.gid.isin(set(d.GAME_ID_KEY.astype(int)))]
    snap = g[~g.odds_date.isin(["open", "close"])].copy()
    snap["h"] = (pd.to_datetime(snap.odds_date, utc=True) - pd.to_datetime(snap.game_date, utc=True)).dt.total_seconds() / 3600
    mid = snap[snap.h <= 10].sort_values("h").groupby("gid").last()[["decimal_home", "decimal_away"]].add_suffix("_10am")
    oc = g[g.odds_date.isin(["open", "close"])].pivot_table(index="gid", columns="odds_date", values=["decimal_home", "decimal_away"], aggfunc="last")
    oc.columns = [f"{a}_{b}" for a, b in oc.columns]
    d = d.merge(oc.join(mid), left_on=d.GAME_ID_KEY.astype(int), right_index=True, how="inner")
    entries = ("open", "10am", "close")
    for t in entries:
        d[f"p_{t}"], d[f"over_{t}"] = devig(d[f"decimal_home_{t}"], d[f"decimal_away_{t}"])
    d = d[np.logical_and.reduce([d[f"over_{t}"].between(1.0, 1.08) for t in entries])].copy()  # drop broken quotes
    d["win"] = (d.home_margin > 0).astype(float)
    d["p_model"] = win_probs(d, "pred_ensemble", fit_probit_scale)
    preds = ["p_model"]
    if "elo_diff" in d and d.elo_diff.notna().all():
        d["p_elo"] = win_probs(d, "elo_diff", fit_logistic)
        preds.append("p_elo")
    d = d.dropna(subset=preds).copy()
    G = d.GAME_DATE.values
    ll = lambda p: -np.mean(d.win * np.log(p) + (1 - d.win) * np.log(1 - p))
    print(f"\n[moneyline] n={len(d)}  logloss: " + "  ".join(f"{c}={ll(d[c].clip(.01, .99)):.4f}" for c in preds + ["p_open", "p_10am", "p_close"]))
    for pc in preds:
        p = d[pc].clip(0.01, 0.99)
        b, se, pv = ols(d.win - d.p_close, p - d.p_close, G)
        print(f"[moneyline] {pc}: encompassing vs close slope={b[1]:+.3f} (se {se[1]:.3f}, p={pv[1]:.3f})")
        for t in ("open", "10am"):
            b, se, _ = ols(logit(d.p_close) - logit(d[f"p_{t}"]), logit(p) - logit(d[f"p_{t}"]), G)
            evh, eva = p * d[f"decimal_home_{t}"] - 1, (1 - p) * d[f"decimal_away_{t}"] - 1
            home = (evh > 0.03) & (evh >= eva)
            bet = home | (eva > 0.03)
            x, hb = d[bet], home[bet]
            clv = np.where(hb, x.p_close * x[f"decimal_home_{t}"], (1 - x.p_close) * x[f"decimal_away_{t}"]) - 1
            ret = np.where(hb, np.where(x.win == 1, x[f"decimal_home_{t}"] - 1, -1), np.where(x.win == 0, x[f"decimal_away_{t}"] - 1, -1))
            n = len(x)
            print(f"    entry={t:4s} CLV slope={b[1]:+.3f} (se {se[1]:.3f})  EV>3% bets n={n:>4}  mean CLV={clv.mean():+.4f} "
                  f"(se {clv.std(ddof=1) / np.sqrt(n):.4f})  ROI={ret.mean():+.4f} (se {ret.std(ddof=1) / np.sqrt(n):.4f})")
    rng = np.random.default_rng(0)
    b, _, _ = ols(logit(d.p_close) - logit(d.p_open), rng.permutation((logit(d.p_model.clip(.01, .99)) - logit(d.p_open)).values))
    print(f"[moneyline] placebo CLV slope (shuffled gap) = {b[1]:+.4f}")


def selftest():
    p, over = devig(np.array([1.5]), np.array([2.6]))
    assert abs(p[0] + (1 / 2.6) / over[0] - 1) < 1e-12 and over[0] > 1
    rng = np.random.default_rng(1)
    truth = rng.normal(0, 1, 20000)
    model = truth + rng.normal(0, 1, 20000)          # half-informative forecast
    y = truth + rng.normal(0, 1, 20000)
    b, se, _ = ols(y, model, groups=np.arange(20000) // 4)
    assert abs(b[1] - 0.5) < 4 * se[1], b       # cov(y, model) / var(model) = 1/2
    assert abs(ols(y, rng.permutation(model))[0][1]) < 0.05
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preds", help="cached OOS predictions CSV (skips running experiment.py)")
    ap.add_argument("--out", default=str(Path(os.environ.get("NBA_OUTPUTS_DIR", ROOT / "outputs")) / "oos_preds.csv"))
    ap.add_argument("--exclude", default="", help="comma-separated features to drop before re-running experiment.py")
    ap.add_argument("--odds", default=None, help="game_odds CSV (default: the lake's game_odds, else csv/game_odds.csv)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit()
    path = a.preds or capture([c for c in a.exclude.split(",") if c], a.out)
    d = pd.read_csv(path, parse_dates=["GAME_DATE"])
    spread_report(d)
    odds, label = load_odds(a.odds) if "GAME_ID_KEY" in d else (None, None)
    if odds is not None:
        print(f"[moneyline] odds from {label}")
        moneyline_report(d, odds)
    else:
        print("[moneyline] skipped: needs GAME_ID_KEY in preds and game_odds (lake table or csv/game_odds.csv)")
