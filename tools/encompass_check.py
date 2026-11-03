"""Forecast-encompassing and CLV checks on experiment.py's out-of-sample predictions (experiment.py is not edited).

    python tools/encompass_check.py                                   # run experiment.py, cache preds, report
    python tools/encompass_check.py --preds outputs/oos_preds.csv     # reuse cached preds
    python tools/encompass_check.py --raw-target --exclude spread_signed,is_home_favorite --out outputs/oos_raw.csv
    python tools/encompass_check.py --selftest

Spread: regress the closing line's error on the model's edge (slope 0 = no information beyond the close).
Moneyline (csv/game_odds.csv, TeamRankings): win prob p = Phi(pred / s), s fit on EARLIER folds' outcomes; de-vigged
open (first snapshot, ~night before), game-day 10am ET, and close. Reports the CLV slope (logit move to close ~ logit
model gap), mean CLV and ROI of EV>3% bets at each entry, encompassing vs the close, and an Elo-only placebo.

Closing-line guard (2026-11-03). experiment.py's _run_fold trains on the market's error and adds the closing line back,
so pred_ensemble CONTAINS the closing line even when spread_signed is excluded from the features; probabilities from it
are "close + noise" and fake a CLV edge against any earlier price (the retracted 2026-10-08 finding). So:
  * capture() measures line_sensitivity = d(pred)/d(spread_signed) on one fold and stores it in the preds CSV; the
    moneyline section REFUSES to run unless it is ~0 (pass --raw-target to fit home_margin directly) or you override
    with --allow-line-leak. Preds without that column are refused too.
  * the moneyline section always prints a PLACEBO row: closing line + SHUFFLED model deviations (zero information).
    A real CLV that is not clearly above its placebo is not evidence of an edge.
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


LINE_SENS_MAX = 0.2   # |d pred / d spread_signed| above this = closing line is inside the prediction (leaky run: ~ -1.0)


def _run_fold_raw(train_df, test_df, features):
    """experiment._run_fold minus the market-error offset: fit home_margin directly so the line cannot enter pred."""
    import experiment as ex
    from sklearn.metrics import mean_squared_error
    X_tr, y_tr, X_te = train_df[features], train_df["home_margin"], test_df[features]
    valid = X_tr.columns[X_tr.notna().any()].tolist()
    X_tr, X_te = X_tr[valid], X_te[valid]
    res = {}
    for spec in ex.MODEL_SPECS:
        p_te = ex._fit_predict(spec, X_tr, y_tr, X_te)
        cut = int(len(X_tr) * 0.8)
        sigma = float(np.std(y_tr.iloc[cut:].values - ex._fit_predict(spec, X_tr.iloc[:cut], y_tr.iloc[:cut], X_tr.iloc[cut:]))) or 1.0
        res[spec["id"]] = (p_te, sigma, float(np.sqrt(mean_squared_error(test_df["home_margin"], p_te))))
    return res


def line_sensitivity(ex, shift=5.0):
    """Mean change in test predictions per +1 of spread_signed on the first walk-forward fold (leaky ~ -1, clean ~ 0)."""
    df = ex.load_data()
    feats = ex.build_features(df)
    for c in feats + ["home_margin", "spread_signed"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    train, test = next(iter(ex.walk_forward_splits(df, "GAME_DATE", ex.TRAIN_SIZE, ex.TEST_SIZE)))
    bumped = test.assign(spread_signed=test.spread_signed + shift)
    base, up = ex._run_fold(train, test, feats), ex._run_fold(train, bumped, feats)
    return float(np.mean([np.mean(up[m][0] - base[m][0]) for m in base]) / shift)


def capture(exclude, out, raw_target=False):
    """Run experiment.run_experiment() with extra feature exclusions; save its OOS table with GAME_ID_KEY, elo_diff, line_sensitivity."""
    sys.path.insert(0, str(ROOT))
    import experiment as ex
    ex.EXTRA_FEATURE_EXCLUSIONS = list(ex.EXTRA_FEATURE_EXCLUSIONS) + exclude
    if raw_target:
        ex._run_fold = _run_fold_raw
    cap, orig = {}, ex.compute_roi
    ex.compute_roi = lambda p, detail=False: (cap.setdefault("df", p.copy()), orig(p, detail=detail))[1]
    print(ex.run_experiment())
    sens = line_sensitivity(ex)
    print(f"[guard] line_sensitivity d(pred)/d(spread_signed) = {sens:+.3f}  ({'LINE IS IN THE PREDICTION' if abs(sens) > LINE_SENS_MAX else 'ok'})")
    p, df = cap["df"], ex.load_data()  # run_experiment's table keeps load_data's row index
    assert np.array_equal(df.loc[p.index, "home_margin"].values, p.home_margin.values), "row alignment broke"
    p["GAME_ID_KEY"] = df.loc[p.index, "GAME_ID_KEY"].values
    p["line_sensitivity"] = sens
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


def ev_bets(d, p, t):
    """EV>3% bets at entry t for home-win prob p: returns (CLV vs the de-vigged close, realised return) per bet."""
    evh, eva = p * d[f"decimal_home_{t}"] - 1, (1 - p) * d[f"decimal_away_{t}"] - 1
    home = (evh > 0.03) & (evh >= eva)
    bet = home | (eva > 0.03)
    x, hb = d[bet], home[bet]
    clv = np.where(hb, x.p_close * x[f"decimal_home_{t}"], (1 - x.p_close) * x[f"decimal_away_{t}"]) - 1
    ret = np.where(hb, np.where(x.win == 1, x[f"decimal_home_{t}"] - 1, -1), np.where(x.win == 0, x[f"decimal_away_{t}"] - 1, -1))
    return clv, ret


def moneyline_report(d, odds_path):
    g = pd.read_csv(odds_path, dtype={"game_id": str})
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
            clv, ret = ev_bets(d, p, t)
            n = len(clv)
            print(f"    entry={t:4s} CLV slope={b[1]:+.3f} (se {se[1]:.3f})  EV>3% bets n={n:>4}  mean CLV={clv.mean():+.4f} "
                  f"(se {clv.std(ddof=1) / np.sqrt(n):.4f})  ROI={ret.mean():+.4f} (se {ret.std(ddof=1) / np.sqrt(n):.4f})")
    # PLACEBO: closing line + shuffled model deviations = zero information. If the real CLV is not clearly above this, no edge.
    line = -d.spread_signed
    pf = win_probs(d.assign(pf=line.values + np.random.default_rng(1).permutation((d.pred_ensemble - line).values)), "pf", fit_probit_scale)
    for t in ("open", "10am"):
        clv, _ = ev_bets(d, pf.clip(0.01, 0.99).fillna(0.5), t)
        print(f"    PLACEBO (closing line + shuffled deviations) entry={t:4s} mean CLV={clv.mean():+.4f} (se {clv.std(ddof=1) / np.sqrt(len(clv)):.4f})")
    rng = np.random.default_rng(0)
    b, _, _ = ols(logit(d.p_close) - logit(d.p_open), rng.permutation((logit(d.p_model.clip(.01, .99)) - logit(d.p_open)).values))
    print(f"[moneyline] placebo CLV slope (shuffled gap) = {b[1]:+.4f}")


def leak_guard(d):
    """Reason string if the preds may contain the closing line (moneyline results would be an artifact), else None."""
    if "line_sensitivity" not in d:
        return "preds CSV has no line_sensitivity column (made before the guard); regenerate with this tool"
    s = float(d.line_sensitivity.iloc[0])
    return f"line_sensitivity={s:+.3f}: pred_ensemble contains the closing line; use --raw-target" if abs(s) > LINE_SENS_MAX else None


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
    # the 2026-11-03 artifact: a predictor glued to the close "beats" a noisy open with zero information; an independent one loses
    n, sig = 40000, lambda z: 1 / (1 + np.exp(-z))
    lc = rng.normal(0, 0.8, n)
    sim = pd.DataFrame({"p_close": sig(lc), "win": (rng.random(n) < sig(lc)).astype(float)})
    po = sig(lc + rng.normal(0, 0.2, n))
    sim["decimal_home_open"], sim["decimal_away_open"] = 1 / (po * 1.04), 1 / ((1 - po) * 1.04)
    near_close, indep = sig(lc + rng.normal(0, 0.3, n)), sig(rng.normal(0, 0.8, n))
    assert ev_bets(sim, pd.Series(near_close), "open")[0].mean() > 0.02
    assert ev_bets(sim, pd.Series(indep), "open")[0].mean() < 0
    assert leak_guard(pd.DataFrame({"line_sensitivity": [-1.0]})) and leak_guard(pd.DataFrame({"x": [1]}))
    assert leak_guard(pd.DataFrame({"line_sensitivity": [0.0]})) is None
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preds", help="cached OOS predictions CSV (skips running experiment.py)")
    ap.add_argument("--out", default=str(Path(os.environ.get("NBA_OUTPUTS_DIR", ROOT / "outputs")) / "oos_preds.csv"))
    ap.add_argument("--exclude", default="", help="comma-separated features to drop before re-running experiment.py")
    ap.add_argument("--odds", default=str(DATA / "csv" / "game_odds.csv"))
    ap.add_argument("--raw-target", action="store_true", help="fit home_margin directly (no market-error offset) so the closing line cannot be in pred")
    ap.add_argument("--allow-line-leak", action="store_true", help="run the moneyline section even though the prediction contains the closing line (results are an artifact)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit()
    path = a.preds or capture([c for c in a.exclude.split(",") if c], a.out, a.raw_target)
    d = pd.read_csv(path, parse_dates=["GAME_DATE"])
    spread_report(d)
    why = leak_guard(d)
    if why and not a.allow_line_leak:
        sys.exit(f"[moneyline] REFUSED: {why}. (--allow-line-leak overrides; the numbers will not mean what they appear to.)")
    if why:
        print(f"[moneyline] WARNING (overridden): {why}")
    if "GAME_ID_KEY" in d and Path(a.odds).exists():
        moneyline_report(d, a.odds)
    else:
        print("[moneyline] skipped: needs GAME_ID_KEY in preds and csv/game_odds.csv")
