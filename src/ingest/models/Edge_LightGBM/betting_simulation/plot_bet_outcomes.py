"""Generate histograms of bet outcomes by bet side for each model (versioned outputs)."""

import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_PATH = "outputs/copilot_bet_logs_odds_v2026_02_19.csv"
PLOT_DIR = "outputs/bet_outcome_histograms_odds_v2026_02_19"
SUMMARY_PATH = "outputs/bet_outcomes_summary_odds_v2026_02_19.csv"


def _get_model_suffixes(df: pd.DataFrame) -> List[str]:
    pnl_cols = [c for c in df.columns if c.startswith("pnl_kelly_")]
    return [c.replace("pnl_kelly_", "") for c in pnl_cols]


def _plot_bet_outcomes_by_side(df: pd.DataFrame, suffix: str, output_dir: str) -> str:
    bet_side_col = f"bet_side_{suffix}"
    pnl_col = f"pnl_kelly_{suffix}"

    if bet_side_col not in df.columns or pnl_col not in df.columns:
        print(f"Skipping {suffix}: missing bet_side or pnl_kelly columns")
        return None

    bet_mask = df[bet_side_col] != "NO BET"
    bets = df.loc[bet_mask].copy()

    if len(bets) == 0:
        print(f"Skipping {suffix}: no bets placed")
        return None

    home_bets = bets[bets[bet_side_col] == "HOME"][pnl_col].values
    away_bets = bets[bets[bet_side_col] == "AWAY"][pnl_col].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if len(home_bets) > 0:
        axes[0].hist(home_bets, bins=30, color="tab:blue", alpha=0.7, edgecolor="black")
        axes[0].axvline(home_bets.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {home_bets.mean():.4f}")
        axes[0].set_title(f"{suffix} - HOME Bets (n={len(home_bets)})", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("P&L (Kelly)")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "No HOME bets", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title(f"{suffix} - HOME Bets")

    if len(away_bets) > 0:
        axes[1].hist(away_bets, bins=30, color="tab:orange", alpha=0.7, edgecolor="black")
        axes[1].axvline(away_bets.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {away_bets.mean():.4f}")
        axes[1].set_title(f"{suffix} - AWAY Bets (n={len(away_bets)})", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("P&L (Kelly)")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No AWAY bets", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title(f"{suffix} - AWAY Bets")

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"bet_outcomes_{suffix}.png")
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return out_path


def _plot_combined_outcomes(df: pd.DataFrame, suffix: str, output_dir: str) -> str:
    bet_side_col = f"bet_side_{suffix}"
    pnl_col = f"pnl_kelly_{suffix}"

    if bet_side_col not in df.columns or pnl_col not in df.columns:
        return None

    bet_mask = df[bet_side_col] != "NO BET"
    bets = df.loc[bet_mask].copy()

    if len(bets) == 0:
        return None

    home_bets = bets[bets[bet_side_col] == "HOME"][pnl_col].values
    away_bets = bets[bets[bet_side_col] == "AWAY"][pnl_col].values

    fig, ax = plt.subplots(figsize=(10, 6))

    all_data = np.concatenate([home_bets, away_bets]) if len(home_bets) > 0 and len(away_bets) > 0 else (home_bets if len(home_bets) > 0 else away_bets)
    bins = np.linspace(all_data.min(), all_data.max(), 30)

    if len(home_bets) > 0:
        ax.hist(home_bets, bins=bins, color="tab:blue", alpha=0.6, label=f"HOME (n={len(home_bets)}, mean={home_bets.mean():.4f})", edgecolor="black")

    if len(away_bets) > 0:
        ax.hist(away_bets, bins=bins, color="tab:orange", alpha=0.6, label=f"AWAY (n={len(away_bets)}, mean={away_bets.mean():.4f})", edgecolor="black")

    ax.axvline(0, color="red", linestyle=":", linewidth=2, alpha=0.7, label="Break-even")
    ax.set_title(f"{suffix} - Bet Outcomes by Side", fontsize=14, fontweight="bold")
    ax.set_xlabel("P&L (Kelly)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"combined_outcomes_{suffix}.png")
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return out_path


def _compute_summary_stats(df: pd.DataFrame, suffix: str) -> dict:
    bet_side_col = f"bet_side_{suffix}"
    pnl_col = f"pnl_kelly_{suffix}"
    bet_win_col = f"bet_win_{suffix}"

    if bet_side_col not in df.columns or pnl_col not in df.columns:
        return {}

    bet_mask = df[bet_side_col] != "NO BET"
    bets = df.loc[bet_mask].copy()

    if len(bets) == 0:
        return {}

    stats = {"model": suffix}
    stats["total_bets"] = len(bets)
    stats["total_pnl"] = bets[pnl_col].sum()
    stats["avg_pnl_per_bet"] = bets[pnl_col].mean()

    if bet_win_col in df.columns:
        stats["win_rate"] = (bets[bet_win_col] == 1).mean()

    home_bets = bets[bets[bet_side_col] == "HOME"]
    stats["home_bets"] = len(home_bets)
    stats["home_pnl"] = home_bets[pnl_col].sum()
    stats["home_avg_pnl"] = home_bets[pnl_col].mean()
    if bet_win_col in df.columns:
        stats["home_win_rate"] = (home_bets[bet_win_col] == 1).mean()

    away_bets = bets[bets[bet_side_col] == "AWAY"]
    stats["away_bets"] = len(away_bets)
    stats["away_pnl"] = away_bets[pnl_col].sum()
    stats["away_avg_pnl"] = away_bets[pnl_col].mean()
    if bet_win_col in df.columns:
        stats["away_win_rate"] = (away_bets[bet_win_col] == 1).mean()

    return stats


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    suffixes = _get_model_suffixes(df)

    if not suffixes:
        raise ValueError("No pnl_kelly_* columns found in bet log.")

    print(f"Found {len(suffixes)} models: {', '.join(suffixes)}")

    print("\nGenerating bet outcome histograms...")
    for suffix in suffixes:
        _plot_bet_outcomes_by_side(df, suffix, PLOT_DIR)
        _plot_combined_outcomes(df, suffix, PLOT_DIR)
        print(f"  ✓ {suffix}")

    print("\nBet Outcome Summary Statistics:")
    print("=" * 120)

    summaries = []
    for suffix in suffixes:
        stats = _compute_summary_stats(df, suffix)
        if stats:
            summaries.append(stats)

    if summaries:
        summary_df = pd.DataFrame(summaries)
        print(summary_df.to_string(index=False))
        summary_df.to_csv(SUMMARY_PATH, index=False)
        print(f"\nSaved summary to {SUMMARY_PATH}")

    print(f"\nSaved plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
