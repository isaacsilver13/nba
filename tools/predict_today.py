"""Model margins and home-win probabilities for one day's NBA slate, for bet_watch's 10:00 CT model picks.

    python tools/predict_today.py                             # today's slate (US/Eastern date) from ESPN
    python tools/predict_today.py --date 2024-03-10           # a past slate (schedule from the local games table)
    python tools/predict_today.py --date 2024-03-10 --check   # also compare with that date's walk-forward fold

Writes $NBA_PREDICTIONS_DIR/<YYYY-MM-DD>.json (default <NBA_OUTPUTS_DIR>/predictions/):
    {"slate_date", "model_version", "generated_at", "data_through",
     "games": [{"game_date", "home", "away", "pred_margin", "p_home_win", "sigma"}]}
home/away are stats.nba.com tricodes (bet_watch teams.abbr); pred_margin is home minus away points; sigma is the
out-of-sample residual SD of the margin, so P(home covers line L, L = home handicap) = Phi((pred_margin + L) / sigma).

The model is experiment.py's (MODEL_SPECS, ensemble weights, TRAIN_SIZE-game rolling window) refit on the latest
completed games, with nothing from the closing line: EXCLUDE drops the line, the favourite flag, the spread prices and
same-day absences, and the target is the raw home margin. (experiment.py learns the market's error, home_margin +
closing spread, and adds the close back, so its predictions contain the close even with spread_signed excluded.)
Win probability: p = Phi(pred / s), s fit by probit on the walk-forward out-of-sample predictions
(tools/encompass_check.py fit_probit_scale), as in docs/edge-research-findings-2026-10.md.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import experiment as ex  # noqa: E402
import run_features as rf  # noqa: E402
from encompass_check import fit_probit_scale  # noqa: E402

EXCLUDE = ["spread_signed", "is_home_favorite", "payout_home", "payout_away", *rf.SAME_DAY_UNKNOWN]
# Days since the last completed game. In season it is 1-7 (All-Star break); an offseason is ~130-200 days. Anything
# else means the completed-games table was not refreshed, and the rolling features would describe stale teams.
STALE_GAP_DAYS = ((10, 100), (240, 10**6))


def predictions_dir() -> Path:
    return Path(os.environ.get("NBA_PREDICTIONS_DIR", ex.NBA_OUTPUTS_DIR / "predictions"))


def model_version(features) -> str:
    spec = json.dumps([features, ex.TRAIN_SIZE, ex.TEST_SIZE, ex.ENSEMBLE_WEIGHTS,
                       [(s["id"], repr(s["model"].get_params())) for s in ex.MODEL_SPECS]], default=str)
    return "margin-noline-" + hashlib.md5(spec.encode()).hexdigest()[:8]


def _fit(train, test, features):
    """{model_id: predictions of test's raw home margin} for one training window."""
    X_tr, X_te = train[features], test[features]
    cols = X_tr.columns[X_tr.notna().any()].tolist()
    return {s["id"]: ex._fit_predict(s, X_tr[cols], train["home_margin"], X_te[cols]) for s in ex.MODEL_SPECS}


def _weights(rmse, upto):
    """experiment.py's ensemble weights for fold `upto`: inverse mean RMSE of folds < upto (equal at fold 1)."""
    if isinstance(ex.ENSEMBLE_WEIGHTS, dict):
        raw = dict(ex.ENSEMBLE_WEIGHTS)
    elif ex.ENSEMBLE_WEIGHTS == "inverse_rmse" and upto > 1:
        raw = {m: 1.0 / np.mean(v[: upto - 1]) for m, v in rmse.items()}
    else:
        raw = {m: 1.0 for m in rmse}
    return {m: w / sum(raw.values()) for m, w in raw.items()}


def walk_forward(df, features, slate=None):
    """Walk-forward over completed games `df` (sorted by date), then fit the newest TRAIN_SIZE-game window for
    `slate`. Returns (out-of-sample ensemble Series on df's index, slate predictions or None)."""
    folds = list(ex.walk_forward_splits(df, "GAME_DATE", ex.TRAIN_SIZE, ex.TEST_SIZE))
    jobs = list(folds)
    if slate is not None:
        # walk_forward_splits' window rule for a test block that starts after the last completed game
        days = df["GAME_DATE"].to_numpy()
        bounds = np.r_[0, np.flatnonzero(days[1:] != days[:-1]) + 1, len(df)]
        start = bounds[np.searchsorted(bounds, len(df) - ex.TRAIN_SIZE, side="right") - 1]
        jobs.append((df.iloc[start:], slate))
    res = Parallel(n_jobs=min(ex.FOLD_WORKERS, len(jobs)))(delayed(_fit)(tr, te, features) for tr, te in jobs)
    rmse = {s["id"]: [] for s in ex.MODEL_SPECS}
    oos = pd.Series(np.nan, index=df.index)
    for k, ((_, test), preds) in enumerate(zip(folds, res), 1):
        w = _weights(rmse, k)
        oos.loc[test.index] = sum(preds[m] * w[m] for m in w)
        for m, p in preds.items():
            rmse[m].append(float(np.sqrt(np.mean((test["home_margin"].to_numpy() - p) ** 2))))
    if slate is None:
        return oos, None
    w = _weights(rmse, len(folds) + 1)
    return oos, sum(res[-1][m] * w[m] for m in w)


def load_history():
    df = ex.load_data()
    df["GAME_ID"] = rf._gid(df["GAME_ID"])
    return df


def predict(date, schedule=None, hist=None):
    D = pd.Timestamp(date).normalize()
    hist = load_history() if hist is None else hist
    done = hist[hist["GAME_DATE"] < D]
    data_through = done["GAME_DATE"].max()
    slate = rf.slate_rows(D, schedule)
    if slate.empty:
        return None
    slate = ex.add_oliver_features(slate.rename(columns={"date_x": "GAME_DATE"}))
    features = [f for f in ex.build_features(done) if f not in EXCLUDE]
    for t in (done, slate):
        t[features] = t[features].apply(pd.to_numeric, errors="coerce")
    oos, pred = walk_forward(done, features, slate)
    ok = oos.notna()
    y, x = done.loc[ok, "home_margin"].to_numpy(), oos[ok].to_numpy()
    sigma = float(np.std(y - x, ddof=1))
    p_home = fit_probit_scale(x, (y > 0).astype(float), pred)
    games = [{"game_date": f"{D:%Y-%m-%d}", "home": h, "away": a, "pred_margin": round(float(m), 2),
              "p_home_win": round(float(p), 4), "sigma": round(sigma, 2)}
             for h, a, m, p in zip(slate["home"], slate["away"], pred, p_home)]
    return {"slate_date": f"{D:%Y-%m-%d}", "model_version": model_version(features),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_through": f"{data_through:%Y-%m-%d}", "games": games}, slate.assign(pred_margin=pred)


def live_schedule(D):
    """Regular-season games on D from ESPN, the source tools/ingest_daily.py uses (stats.nba.com refuses this PC).
    The model is trained on regular season only. GAME_IDs are placeholders above any ingested game's."""
    from ingest_daily import game_id_base, scoreboard
    g = scoreboard(D.date(), final_only=False)
    return g.assign(GAME_ID=[game_id_base(s) + 9000 + i for i, s in enumerate(g["season"], 1)])[["GAME_ID", "home", "away"]]


def write(payload, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{payload['slate_date']}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # bet_watch never sees a half-written file
    return path


def check(D, hist, slate):
    """Max |difference| between these predictions and the walk-forward fold that contains D (same code path)."""
    rows = hist[hist["GAME_DATE"] <= D]
    features = [f for f in ex.build_features(rows) if f not in EXCLUDE]
    rows = rows.assign(**{f: pd.to_numeric(rows[f], errors="coerce") for f in features})
    oos, _ = walk_forward(rows, features)
    ref = pd.Series(oos[rows["GAME_DATE"] == D].to_numpy(), index=rows.loc[rows["GAME_DATE"] == D, "GAME_ID"])
    starts = {test["GAME_DATE"].min() for _, test in ex.walk_forward_splits(rows, "GAME_DATE", ex.TRAIN_SIZE, ex.TEST_SIZE)}
    got = slate.set_index("GAME_ID")["pred_margin"].reindex(ref.index)
    diff = (got - ref).abs()
    print(f"[check] {len(ref)} games; {D.date()} {'IS' if D in starts else 'is NOT'} a fold start "
          f"(only then is the training window identical); |pred - fold pred| max {diff.max():.6f}, mean {diff.mean():.6f}")
    return float(diff.max())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", help="YYYY-MM-DD (default: today, US/Eastern)")
    ap.add_argument("--check", action="store_true", help="past dates: compare with the matching walk-forward fold")
    a = ap.parse_args()
    D = pd.Timestamp(a.date or datetime.now(ZoneInfo("America/New_York")).date())
    hist = load_history()
    gap = (D - hist.loc[hist["GAME_DATE"] < D, "GAME_DATE"].max()).days
    if any(lo < gap < hi for lo, hi in STALE_GAP_DAYS):
        raise SystemExit(f"completed games end {gap} days before {D.date()}: ingest the missing games and rebuild "
                         "features (tools/run_features.py) first")
    schedule = None if D <= hist["GAME_DATE"].max() else live_schedule(D)  # past dates: local games table
    result = predict(D, schedule, hist)
    if result is None:
        print(f"no regular-season games on {D.date()} (or the scoreboard fetch failed); nothing written")
        return
    payload, slate = result
    print(pd.DataFrame(payload["games"]).to_string(index=False))
    print(f"wrote {write(payload, predictions_dir())}  ({payload['model_version']}, data through {payload['data_through']})")
    if a.check:
        check(D, hist, slate)


if __name__ == "__main__":
    main()
