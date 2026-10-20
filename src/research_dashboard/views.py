"""Pure helpers for dashboard pages (testable without Streamlit)."""

import numpy as np
import pandas as pd


def chart_gate_message(run_id: str | None, status: str | None) -> str | None:
    """Explain why run-backed charts are unavailable; complete runs need no warning."""
    if run_id is None:
        return "No run selected."
    if status != "complete":
        return f"Run `{run_id}` is **{status}**; charts are disabled. See Run Health."
    return None


def comparison_table(runs, tiers) -> pd.DataFrame:
    """One auditable comparison row per complete run with explicit run/tier metric scopes."""
    rows = []
    for run in runs:
        if run.status != "complete":
            continue
        metrics = run.read_metrics()
        all_bets = run.read("bets")
        selected_bets = filter_tiers(all_bets, tiers)
        selected_audit = filter_tiers(run.read("odds_audit"), tiers)
        selected = bet_summary(selected_bets)
        full = bet_summary(all_bets)
        odds_counts = run.data.get("odds_counts", {})
        audited = sum(odds_counts.values())
        rows.append({
            "run_id": run.run_id,
            "created_at": run.data.get("created_at"),
            "experiment_id": metrics.get("exp_id"),
            "source_fingerprint": run.data.get("source_fingerprint"),
            "config_fingerprint": run.data.get("config_fingerprint"),
            "data_fingerprint": run.data.get("data_fingerprint"),
            "run_model_rmse": metrics.get("ensemble_rmse", metrics.get("rmse")),
            "run_flat_roi": metrics.get("flat_roi"),
            "run_flat_roi_se": metrics.get("flat_roi_se"),
            "run_score": metrics.get("score"),
            "run_compound_return": metrics.get("compound_roi", metrics.get("roi")),
            "run_bets": metrics.get("n_bets", full["bets"]),
            "run_max_drawdown": full.get("max_drawdown"),
            "tier_bets": selected["bets"],
            "tier_win_rate": selected.get("win_rate"),
            "tier_roi": selected.get("roi"),
            "tier_coverage": selected["bets"] / len(selected_audit) if len(selected_audit) else None,
            "matched_odds_share": odds_counts.get("matched", 0) / audited if audited else None,
            "folds": metrics.get("n_folds", len(run.data.get("folds", []))),
            "data_rows": metrics.get("data_rows"),
        })
    return pd.DataFrame(rows)


def filter_tiers(df: pd.DataFrame, tiers) -> pd.DataFrame:
    return df[df["odds_tier"].isin(list(tiers))]


def bet_summary(bets: pd.DataFrame) -> dict:
    """Convert bankroll-relative stake/P&L fractions to common bankroll units before ROI."""
    if bets.empty:
        return {"bets": 0}
    ordered = bets.sort_values("game_date", kind="stable")
    bk = ordered["bankroll"].astype(float)
    pnl_fraction = ordered["pnl"].astype(float)
    if "bankroll_open" in ordered:
        opening = ordered["bankroll_open"].astype(float)
    else:
        opening = bk / (1.0 + pnl_fraction)
    stakes = ordered["kelly_fraction"].astype(float) * opening
    profits = pnl_fraction * opening
    peaks = np.maximum.accumulate(np.r_[1.0, bk.to_numpy()])[1:]
    return {
        "bets": len(bets),
        "win_rate": ordered["win"].mean(),
        "roi": profits.sum() / stakes.sum() if stakes.sum() else 0.0,
        "max_drawdown": float(np.max((peaks - bk.to_numpy()) / peaks)),
    }


def reliability(pred: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Reliability on ALL test games (not just selected bets), with per-bin counts."""
    b = pd.cut(pred["p_home"], np.linspace(0, 1, bins + 1), include_lowest=True)
    out = pred.groupby(b, observed=True).agg(mean_pred=("p_home", "mean"), actual=("home_win", "mean"),
                                              n=("home_win", "size")).reset_index(drop=True)
    return out
