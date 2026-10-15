"""Replay experiment.py unchanged and capture its per-row detail as a dashboard run bundle.

It calls the real `experiment.run_experiment()` (so metrics are exactly experiment.py's), spying on
load_data / walk_forward_splits / compute_roi to record what run_experiment does not expose.
It never writes results.tsv and never edits experiment.py. Trains models on local data: no network.

    python -m src.research_dashboard.replay_run --yes [--run-id ID] [--outputs-dir DIR]

Column meaning (matches experiment.py, which models the SPREAD, not the moneyline winner):
    predictions.p_home  = P(home covers) = norm.cdf((pred + spread_signed) / sigma), as compute_roi computes it
    predictions.home_win = home covered (home_margin > -spread_signed)
"""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .capture_run import capture
from .catalog import outputs_dir

def roi_detail_fn(exp):
    """experiment.compute_roi with detail=True: the same selection/Kelly logic, returning per-row pieces."""
    return lambda df: exp.compute_roi(df, detail=True)


def raw_odds_prices(exp) -> pd.DataFrame:
    """Re-run load_data's odds join up to (not including) mirror/default. Index: GAME_ID_KEY.
    Games absent from the join are absent here -> 'unmatched'. Parity-checked against load_data's output."""
    import os
    if not (os.path.exists(exp.ODDS_PATH) and os.path.exists(exp.PROCESSED_GAMES_PATH)):
        return pd.DataFrame(columns=["home_raw", "away_raw"])
    odds = exp.read_odds().rename(columns={"home_american": "home_raw", "away_american": "away_raw"})
    dedup = odds.drop_duplicates(["date", "home", "away"], keep="last")
    games = pd.read_csv(exp.PROCESSED_GAMES_PATH, usecols=["GAME_ID", "date", "home", "away"])
    games["GAME_ID_KEY"] = exp._canonical_game_id(games["GAME_ID"])
    games["date"] = pd.to_datetime(games["date"]).dt.normalize()
    j = dedup.merge(games, on=["date", "home", "away"], how="inner")
    j["GAME_ID_KEY"] = exp._canonical_game_id(j["GAME_ID"])
    return j.drop_duplicates("GAME_ID_KEY", keep="last").set_index("GAME_ID_KEY")[["home_raw", "away_raw"]]


def odds_tiers(game_keys: pd.Series, raw: pd.DataFrame) -> pd.Series:
    """matched: both prices; mirrored: one price copied to the other side; default: odds row but no price
    (-110 used); unmatched: game not in the odds join (-110 used)."""
    r = raw.reindex(game_keys.values)
    in_join = game_keys.isin(raw.index).values
    h, a = r["home_raw"].notna().values, r["away_raw"].notna().values
    t = np.select([~in_join, h & a, h | a], ["unmatched", "matched", "mirrored"], "default")
    return pd.Series(t, index=game_keys.index)


def check_join_parity(df: pd.DataFrame, raw: pd.DataFrame, default_odds: float) -> None:
    """Mirror/default raw prices exactly like load_data and require identical final prices."""
    r = raw.reindex(df["GAME_ID_KEY"].values)
    h, a = r["home_raw"].copy(), r["away_raw"].copy()
    h = h.where(h.notna(), a).fillna(default_odds).values
    a = r["away_raw"].where(r["away_raw"].notna(), r["home_raw"]).fillna(default_odds).values
    if not (np.allclose(h, df["home_american"].values, equal_nan=True)
            and np.allclose(a, df["away_american"].values, equal_nan=True)):
        raise RuntimeError("odds-join replica disagrees with experiment.load_data; refusing to label odds tiers")


def build_artifacts(df, test_in, fold_of, detail, raw, default_odds, out_dir: Path, metrics: dict) -> dict:
    """Pure: turn captured pieces into the five artifact files. Returns {artifact: path}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    d = detail["df"]  # compute_roi's date-sorted frame; carries GAME_ID_KEY / fold from test_in
    d = d.assign(win_prob=np.asarray(detail["win_prob"]), covered=detail["covered"], pnl=detail["pnl"])
    date = d["GAME_DATE"].dt.strftime("%Y-%m-%d")

    pred = pd.DataFrame({"game_id": d["GAME_ID_KEY"], "game_date": date, "fold": d["fold"],
                         "p_home": d["win_prob"].round(6), "home_win": (d["covered"] == "HOME").astype(int)})
    tier = odds_tiers(d["GAME_ID_KEY"], raw)
    audit = pd.DataFrame({"game_id": d["GAME_ID_KEY"], "game_date": date, "odds_tier": tier})

    sel = np.asarray(detail["selected"], dtype=bool)
    book = np.cumprod(1.0 + d["pnl"].values)
    s = d[sel]
    home = s["_bet_side"] == "HOME"
    prob = np.where(home, s["win_prob"], 1 - s["win_prob"])
    payout = np.where(home, s["payout_home"].clip(lower=0.01), s["payout_away"].clip(lower=0.01))
    bets = pd.DataFrame({
        "game_id": s["GAME_ID_KEY"], "game_date": date[sel], "bet_side": s["_bet_side"],
        "odds_tier": tier[sel], "prob": np.round(prob, 6), "edge": np.round(prob * payout - (1 - prob), 6),
        "kelly_fraction": s["_kelly"].round(6), "win": (s["covered"] == s["_bet_side"]).astype(int),
        "pnl": s["pnl"].round(6), "bankroll": np.round(book[sel], 6),
    })

    folds = []
    for f, g in d.groupby("fold"):
        err = g["home_margin"] - g["pred_ensemble"]
        folds.append({"fold": int(f), "test_start": g["GAME_DATE"].min().strftime("%Y-%m-%d"),
                      "test_end": g["GAME_DATE"].max().strftime("%Y-%m-%d"), "rows": len(g),
                      "rmse": round(float(np.sqrt((err ** 2).mean())), 4), "mae": round(float(err.abs().mean()), 4)})

    paths = {}
    for name, frame in {"fold_metrics": pd.DataFrame(folds), "predictions": pred, "bets": bets, "odds_audit": audit}.items():
        paths[name] = out_dir / f"{name}.csv"
        frame.to_csv(paths[name], index=False, lineterminator="\n")  # LF so manifest hashes match across OSes
    paths["metrics"] = out_dir / "metrics.json"
    paths["metrics"].write_text(json.dumps({
        **metrics, "compound_roi": float(book[-1] - 1.0) if len(book) else 0.0,
        "units": "kelly_fraction and pnl are fractions of the CURRENT bankroll; bankroll starts at 1.0, compounds in date order",
        "rmse_scope": "fold rmse/mae are ensemble errors on home_margin",
        "prob_meaning": "P(home covers spread) per experiment.compute_roi; home_win = home covered"}, indent=2))
    return paths


def replay(run_id: str, outputs: Path) -> Path:
    import experiment as exp

    cap = {"folds": {}}
    orig_load, orig_split, orig_roi = exp.load_data, exp.walk_forward_splits, exp.compute_roi

    def load_spy():
        cap["df"] = orig_load()
        return cap["df"]

    def split_spy(*a, **k):
        for i, (tr, te) in enumerate(orig_split(*a, **k), 1):
            cap["folds"][i] = te.index.values  # second pass overwrites with identical values
            yield tr, te

    def roi_spy(preds_df, **kw):
        cap["roi_in"] = preds_df.copy()
        return orig_roi(preds_df, **kw)

    exp.load_data, exp.walk_forward_splits, exp.compute_roi = load_spy, split_spy, roi_spy
    try:
        metrics = exp.run_experiment()
    finally:
        exp.load_data, exp.walk_forward_splits, exp.compute_roi = orig_load, orig_split, orig_roi

    df, test_in = cap["df"], cap["roi_in"]
    fold_of = pd.Series({i: f for f, idx in cap["folds"].items() for i in idx})
    test_in["GAME_ID_KEY"] = df.loc[test_in.index, "GAME_ID_KEY"]
    test_in["fold"] = fold_of.reindex(test_in.index).values
    detail = roi_detail_fn(exp)(test_in)
    if round(detail["bankroll"] - 1.0, 4) != metrics["roi"]:
        raise RuntimeError(f"replayed ROI {detail['bankroll'] - 1.0:.4f} != experiment ROI {metrics['roi']}")
    raw = raw_odds_prices(exp)
    check_join_parity(df, raw, exp.DEFAULT_AMERICAN_ODDS)

    params = exp.params_summary()
    meta = {**metrics, "exp_id": hashlib.md5(params.encode()).hexdigest()[:8], "params": json.loads(params),
            "data_rows": len(df)}
    with tempfile.TemporaryDirectory() as t:
        return capture(run_id, build_artifacts(df, test_in, fold_of, detail, raw, exp.DEFAULT_AMERICAN_ODDS,
                                               Path(t), meta), outputs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="confirm: trains the configured models on local data")
    ap.add_argument("--run-id")
    ap.add_argument("--outputs-dir", type=Path)
    a = ap.parse_args()
    if not a.yes:
        ap.error("pass --yes to confirm a full local training replay")
    out = a.outputs_dir or outputs_dir(Path(__file__).resolve().parents[2])
    run_id = a.run_id or "replay_" + pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
    print(replay(run_id, out))


if __name__ == "__main__":
    main()
