"""Simulate bankroll evolution from copilot_model_output bet signals."""

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_PATH = "outputs/copilot_model_output.csv"
OUTPUT_PATH = "outputs/copilot_betting_sim.csv"
SUMMARY_PATH = "outputs/copilot_betting_summary.csv"
BET_LOG_PATH = "outputs/copilot_bet_logs.csv"
PLOT_DIR = "outputs/bet_plots"

STARTING_BANKROLL = 10000.0


def _get_model_suffixes(df: pd.DataFrame) -> List[str]:
    pnl_cols = [c for c in df.columns if c.startswith("pnl_kelly_")]
    return [c.replace("pnl_kelly_", "") for c in pnl_cols]


def _compute_bankroll_series(df: pd.DataFrame, pnl_col: str, starting_bankroll: float) -> pd.Series:
    pnl = df[pnl_col].fillna(0.0).to_numpy()
    bankroll = np.empty(len(pnl), dtype=float)
    current = starting_bankroll
    for i, step in enumerate(pnl):
        current = current * (1 + step)
        bankroll[i] = current
    return pd.Series(bankroll, index=df.index)


def _compute_drawdown(series: pd.Series) -> pd.Series:
    peak = series.cummax()
    return (series - peak) / peak.replace(0, np.nan)


def _summary_for_model(df: pd.DataFrame, suffix: str, bankroll_series: pd.Series) -> Dict[str, float]:
    bet_side_col = f"bet_side_{suffix}"
    bet_win_col = f"bet_win_{suffix}"

    if bet_side_col in df.columns:
        bet_mask = df[bet_side_col] != "NO BET"
        num_bets = int(bet_mask.sum())
    else:
        bet_mask = df[f"pnl_kelly_{suffix}"] != 0
        num_bets = int(bet_mask.sum())

    if bet_win_col in df.columns and num_bets > 0:
        win_rate = float(df.loc[bet_mask, bet_win_col].mean())
    else:
        win_rate = float("nan")

    start = bankroll_series.iloc[0] if len(bankroll_series) else STARTING_BANKROLL
    end = bankroll_series.iloc[-1] if len(bankroll_series) else STARTING_BANKROLL
    roi = (end - start) / start if start else float("nan")

    dd = _compute_drawdown(bankroll_series)
    max_dd = float(dd.min()) if len(dd) else float("nan")

    return {
        "model_id": suffix,
        "starting_bankroll": float(start),
        "ending_bankroll": float(end),
        "roi": float(roi),
        "num_bets": num_bets,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
    }


def _build_bet_log(df: pd.DataFrame, suffixes: List[str]) -> pd.DataFrame:
    """Build consolidated bet log with all model predictions per game on one row.

    Args:
        df: Full dataframe with all model columns.
        suffixes: List of model suffixes (xgboost_v1, lightgbm_v2, etc.).

    Returns:
        DataFrame with one row per game containing all model predictions/metrics.
        Only includes games where at least one model placed a bet.
    """
    # Identify which games have at least one bet
    bet_mask = pd.Series(False, index=df.index)
    for suffix in suffixes:
        bet_side_col = f"bet_side_{suffix}"
        pnl_col = f"pnl_kelly_{suffix}"
        if bet_side_col in df.columns:
            bet_mask |= (df[bet_side_col] != "NO BET")
        else:
            bet_mask |= (df[pnl_col] != 0)

    # Start with game identifiers
    base_cols = ["GAME_DATE", "GAME_ID", "home", "away", "spread_signed", "home_margin"]
    optional_cols = ["home_margin_needed", "did_home_cover", "team_that_covered"]

    bet_log = df.loc[bet_mask, base_cols].copy()

    # Add optional columns if present
    for col in optional_cols:
        if col in df.columns:
            bet_log[col] = df.loc[bet_mask, col]

    # Add all model-specific columns for the betting games
    for suffix in suffixes:
        model_cols = [
            f"pred_{suffix}",
            f"edge_{suffix}",
            f"win_prob_home_{suffix}",
            f"ev_home_{suffix}",
            f"ev_away_{suffix}",
            f"bet_side_{suffix}",
            f"kelly_frac_{suffix}",
            f"bet_win_{suffix}",
            f"pnl_kelly_{suffix}",
        ]
        for col in model_cols:
            if col in df.columns:
                bet_log[col] = df.loc[bet_mask, col]

    # Reset index and return
    return bet_log.reset_index(drop=True)


def _write_plot(df: pd.DataFrame, suffix: str, output_dir: str) -> str:
    bankroll_col = f"bankroll_{suffix}"
    drawdown_col = f"drawdown_{suffix}"

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"bankroll_{suffix}.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["GAME_DATE"], df[bankroll_col], label="Bankroll")
    ax.set_title(f"Bankroll Trajectory - {suffix}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Bankroll")
    ax.grid(True)

    ax2 = ax.twinx()
    ax2.plot(df["GAME_DATE"], df[drawdown_col], color="tab:red", alpha=0.6, label="Drawdown")
    ax2.set_ylabel("Drawdown")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, parse_dates=["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    suffixes = _get_model_suffixes(df)
    if not suffixes:
        raise ValueError("No pnl_kelly_* columns found in copilot_model_output.")

    output_df = df[["GAME_DATE", "GAME_ID", "home", "away", "home_margin", "spread_signed"]].copy()
    summaries = []

    for suffix in suffixes:
        pnl_col = f"pnl_kelly_{suffix}"
        bankroll_col = f"bankroll_{suffix}"
        drawdown_col = f"drawdown_{suffix}"

        bankroll_series = _compute_bankroll_series(df, pnl_col, STARTING_BANKROLL)
        output_df[bankroll_col] = bankroll_series
        output_df[drawdown_col] = _compute_drawdown(bankroll_series)

        summaries.append(_summary_for_model(df, suffix, bankroll_series))
        _write_plot(output_df, suffix, PLOT_DIR)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)
    pd.DataFrame(summaries).to_csv(SUMMARY_PATH, index=False)

    # Build consolidated bet log (one row per game with all model columns)
    bet_log = _build_bet_log(df, suffixes)
    if len(bet_log) > 0:
        bet_log.to_csv(BET_LOG_PATH, index=False)

    print(f"Saved bankroll simulation to {OUTPUT_PATH}")
    print(f"Saved summary to {SUMMARY_PATH}")
    print(f"Saved bet logs to {BET_LOG_PATH}")
    print(f"Saved plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
