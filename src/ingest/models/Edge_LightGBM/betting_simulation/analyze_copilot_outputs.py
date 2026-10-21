"""Analyze latest copilot model outputs and generate stats + plots in a dedicated run folder."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VERSION_RE = re.compile(r"_odds_v(\d{4}_\d{2}_\d{2})\.csv$", re.IGNORECASE)


@dataclass
class InputSpec:
    key: str
    odds_glob: str
    fallback_name: str
    required: bool = False


@dataclass
class ExperimentSpec:
    experiment_id: str
    kelly_scale: float
    kelly_cap: float
    top_n_bets_per_day: int
    daily_max_exposure: float
    objective_type: str = "roi"
    objective_lambda: float = 0.0
    exclude_default_payout: bool = False
    allowed_market_tiers: Optional[Tuple[str, ...]] = None
    max_daily_loss: Optional[float] = None


INPUT_SPECS: Sequence[InputSpec] = (
    InputSpec("model_output", "copilot_model_output_odds_v*.csv", "copilot_model_output.csv", required=True),
    InputSpec("model_rmse", "copilot_model_rmse_odds_v*.csv", "copilot_model_rmse.csv"),
    InputSpec("join_audit", "copilot_model_join_audit_odds_v*.csv", "copilot_model_join_audit.csv"),
    InputSpec("objective_sweep", "copilot_objective_sweep_odds_v*.csv", "copilot_objective_sweep.csv"),
)


REQUIRED_BASE_COLUMNS = ["GAME_DATE"]
OPTIONAL_BASE_COLUMNS = [
    "GAME_ID",
    "home",
    "away",
    "home_margin",
    "spread_signed",
    "home_margin_needed",
    "did_home_cover",
    "team_that_covered",
    "payout_home",
    "payout_away",
    "payout_source",
    "used_default_payout",
    "market_match_quality",
]


DEFAULT_EXPERIMENT_MATRIX: Sequence[ExperimentSpec] = (
    ExperimentSpec(
        experiment_id="R0_baseline",
        kelly_scale=1.00,
        kelly_cap=0.10,
        top_n_bets_per_day=2,
        daily_max_exposure=0.25,
    ),
    ExperimentSpec(
        experiment_id="R1_conservative",
        kelly_scale=0.25,
        kelly_cap=0.01,
        top_n_bets_per_day=2,
        daily_max_exposure=0.03,
    ),
    ExperimentSpec(
        experiment_id="R2_ultra_risk",
        kelly_scale=0.10,
        kelly_cap=0.005,
        top_n_bets_per_day=1,
        daily_max_exposure=0.01,
        max_daily_loss=-0.02,
    ),
    ExperimentSpec(
        experiment_id="R3_market_gated",
        kelly_scale=0.25,
        kelly_cap=0.01,
        top_n_bets_per_day=2,
        daily_max_exposure=0.03,
        exclude_default_payout=True,
        allowed_market_tiers=("tier_b_moneyline",),
    ),
    ExperimentSpec(
        experiment_id="R3b_market_gated_mirror",
        kelly_scale=0.25,
        kelly_cap=0.01,
        top_n_bets_per_day=2,
        daily_max_exposure=0.03,
        exclude_default_payout=True,
        allowed_market_tiers=("tier_b_moneyline", "tier_c_mirrored"),
    ),
    ExperimentSpec(
        experiment_id="R4_objective_roi",
        kelly_scale=0.25,
        kelly_cap=0.01,
        top_n_bets_per_day=2,
        daily_max_exposure=0.03,
        objective_type="roi",
        objective_lambda=0.0,
    ),
    ExperimentSpec(
        experiment_id="R4_objective_lgdd_025",
        kelly_scale=0.25,
        kelly_cap=0.01,
        top_n_bets_per_day=2,
        daily_max_exposure=0.03,
        objective_type="log_growth_dd",
        objective_lambda=0.25,
    ),
    ExperimentSpec(
        experiment_id="R4_objective_lgdd_050",
        kelly_scale=0.25,
        kelly_cap=0.01,
        top_n_bets_per_day=2,
        daily_max_exposure=0.03,
        objective_type="log_growth_dd",
        objective_lambda=0.50,
    ),
    ExperimentSpec(
        experiment_id="R5_stability_tier_b",
        kelly_scale=0.05,
        kelly_cap=0.003,
        top_n_bets_per_day=1,
        daily_max_exposure=0.005,
        exclude_default_payout=True,
        allowed_market_tiers=("tier_b_moneyline",),
        max_daily_loss=-0.01,
    ),
    ExperimentSpec(
        experiment_id="R6_stability_tier_b_c",
        kelly_scale=0.10,
        kelly_cap=0.005,
        top_n_bets_per_day=1,
        daily_max_exposure=0.008,
        exclude_default_payout=True,
        allowed_market_tiers=("tier_b_moneyline", "tier_c_mirrored"),
        max_daily_loss=-0.015,
    ),
)


def _parse_version_date(path: Path) -> Optional[datetime]:
    match = VERSION_RE.search(path.name)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y_%m_%d")


def _file_rank(path: Path) -> Tuple[datetime, float, str]:
    version_dt = _parse_version_date(path) or datetime(1900, 1, 1)
    return version_dt, path.stat().st_mtime, path.name


def _pick_latest(candidates: Sequence[Path]) -> Optional[Path]:
    if not candidates:
        return None
    return sorted(candidates, key=_file_rank)[-1]


def discover_inputs(outputs_dir: Path, prefer_odds: bool = True, allow_fallback: bool = True) -> Tuple[Dict[str, Path], pd.DataFrame]:
    selected: Dict[str, Path] = {}
    manifest_rows: List[Dict[str, object]] = []

    for spec in INPUT_SPECS:
        odds_candidates = list(outputs_dir.glob(spec.odds_glob))
        fallback_path = outputs_dir / spec.fallback_name
        chosen: Optional[Path] = None
        source = ""
        version_tag = ""

        if prefer_odds:
            chosen = _pick_latest(odds_candidates)
            if chosen is not None:
                source = "odds_versioned_latest"
                dt = _parse_version_date(chosen)
                version_tag = dt.strftime("%Y_%m_%d") if dt else ""

        if chosen is None and allow_fallback and fallback_path.exists():
            chosen = fallback_path
            source = "fallback_canonical"

        if chosen is not None:
            selected[spec.key] = chosen

        manifest_rows.append(
            {
                "input_key": spec.key,
                "required": spec.required,
                "selected_path": str(chosen) if chosen else "",
                "selection_source": source,
                "odds_candidates_found": len(odds_candidates),
                "fallback_exists": fallback_path.exists(),
                "selected_version_date": version_tag,
            }
        )

        if spec.required and chosen is None:
            raise FileNotFoundError(
                f"Could not resolve required input '{spec.key}'. Checked odds pattern '{spec.odds_glob}' and fallback '{spec.fallback_name}'."
            )

    manifest_df = pd.DataFrame(manifest_rows)
    return selected, manifest_df


def _get_model_suffixes(df: pd.DataFrame) -> List[str]:
    pnl_cols = [c for c in df.columns if c.startswith("pnl_kelly_")]
    return [c.replace("pnl_kelly_", "") for c in pnl_cols]


def _compute_bankroll_series(df: pd.DataFrame, pnl_col: str, starting_bankroll: float) -> pd.Series:
    pnl = df[pnl_col].fillna(0.0).to_numpy(dtype=float)
    bankroll = np.empty(len(pnl), dtype=float)
    current = float(starting_bankroll)
    for i, step in enumerate(pnl):
        current = current * (1.0 + step)
        bankroll[i] = current
    return pd.Series(bankroll, index=df.index)


def _compute_drawdown(series: pd.Series) -> pd.Series:
    peak = series.cummax()
    return (series - peak) / peak.replace(0, np.nan)


def validate_schema(model_df: pd.DataFrame, strict_schema: bool = True) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for col in REQUIRED_BASE_COLUMNS:
        rows.append({"column": col, "required": True, "present": col in model_df.columns})

    for col in OPTIONAL_BASE_COLUMNS:
        rows.append({"column": col, "required": False, "present": col in model_df.columns})

    suffixes = _get_model_suffixes(model_df)
    rows.append(
        {
            "column": "pnl_kelly_*",
            "required": True,
            "present": len(suffixes) > 0,
        }
    )

    schema_df = pd.DataFrame(rows)
    if strict_schema:
        missing = schema_df[(schema_df["required"] == True) & (schema_df["present"] == False)]  # noqa: E712
        if len(missing) > 0:
            missing_cols = ", ".join(missing["column"].tolist())
            raise ValueError(f"Missing required schema elements: {missing_cols}")
    return schema_df


def _build_bet_log(df: pd.DataFrame, suffixes: List[str]) -> pd.DataFrame:
    bet_mask = pd.Series(False, index=df.index)
    for suffix in suffixes:
        bet_side_col = f"bet_side_{suffix}"
        pnl_col = f"pnl_kelly_{suffix}"
        if bet_side_col in df.columns:
            bet_mask |= df[bet_side_col].fillna("NO BET") != "NO BET"
        elif pnl_col in df.columns:
            bet_mask |= df[pnl_col].fillna(0.0) != 0

    keep_cols = [c for c in OPTIONAL_BASE_COLUMNS if c in df.columns]
    if "GAME_DATE" in df.columns and "GAME_DATE" not in keep_cols:
        keep_cols = ["GAME_DATE"] + keep_cols

    extra_cols: List[str] = []
    for suffix in suffixes:
        model_cols = [
            f"pred_{suffix}",
            f"sigma_{suffix}",
            f"edge_{suffix}",
            f"edge_std_{suffix}",
            f"win_prob_home_{suffix}",
            f"ev_home_{suffix}",
            f"ev_away_{suffix}",
            f"bet_side_{suffix}",
            f"kelly_frac_{suffix}",
            f"bet_win_{suffix}",
            f"pnl_kelly_{suffix}",
        ]
        extra_cols.extend([c for c in model_cols if c in df.columns])

    selected_cols = keep_cols + [c for c in extra_cols if c not in keep_cols]
    return df.loc[bet_mask, selected_cols].reset_index(drop=True)


def build_bankroll_outputs(model_df: pd.DataFrame, starting_bankroll: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    model_df = model_df.copy()
    model_df["GAME_DATE"] = pd.to_datetime(model_df["GAME_DATE"])
    model_df = model_df.sort_values("GAME_DATE").reset_index(drop=True)

    suffixes = _get_model_suffixes(model_df)
    if not suffixes:
        raise ValueError("No pnl_kelly_* columns found in model output.")

    base_cols = [c for c in ["GAME_DATE", "GAME_ID", "home", "away", "home_margin", "spread_signed"] if c in model_df.columns]
    sim_df = model_df[base_cols].copy()
    summary_rows: List[Dict[str, object]] = []

    for suffix in suffixes:
        pnl_col = f"pnl_kelly_{suffix}"
        bet_col = f"bet_side_{suffix}"
        win_col = f"bet_win_{suffix}"
        edge_col = f"edge_{suffix}"
        ev_home_col = f"ev_home_{suffix}"
        ev_away_col = f"ev_away_{suffix}"

        bankroll_col = f"bankroll_{suffix}"
        drawdown_col = f"drawdown_{suffix}"

        bankroll = _compute_bankroll_series(model_df, pnl_col, starting_bankroll)
        drawdown = _compute_drawdown(bankroll)

        sim_df[bankroll_col] = bankroll
        sim_df[drawdown_col] = drawdown

        if bet_col in model_df.columns:
            bet_mask = model_df[bet_col].fillna("NO BET") != "NO BET"
        else:
            bet_mask = model_df[pnl_col].fillna(0.0) != 0

        total_bets = int(bet_mask.sum())
        win_rate = float(model_df.loc[bet_mask, win_col].mean()) if win_col in model_df.columns and total_bets > 0 else np.nan
        avg_edge = float(model_df.loc[bet_mask, edge_col].mean()) if edge_col in model_df.columns and total_bets > 0 else np.nan

        ev_side = pd.Series(np.nan, index=model_df.index)
        if ev_home_col in model_df.columns and ev_away_col in model_df.columns and bet_col in model_df.columns:
            ev_side = np.where(
                model_df[bet_col] == "HOME",
                model_df[ev_home_col],
                np.where(model_df[bet_col] == "AWAY", model_df[ev_away_col], np.nan),
            )
            ev_side = pd.Series(ev_side, index=model_df.index)

        avg_ev = float(ev_side[bet_mask].astype(float).mean()) if total_bets > 0 else np.nan
        total_pnl = float(model_df[pnl_col].fillna(0.0).sum())

        start = float(starting_bankroll)
        end = float(bankroll.iloc[-1]) if len(bankroll) else float(starting_bankroll)
        roi = (end - start) / start if start else np.nan
        max_drawdown = float(drawdown.min()) if len(drawdown) else np.nan

        summary_rows.append(
            {
                "model": suffix,
                "starting_bankroll": start,
                "ending_bankroll": end,
                "roi": roi,
                "total_pnl": total_pnl,
                "num_bets": total_bets,
                "win_rate": win_rate,
                "avg_edge": avg_edge,
                "avg_ev_selected_side": avg_ev,
                "max_drawdown": max_drawdown,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("roi", ascending=False)
    bet_log_df = _build_bet_log(model_df, suffixes)
    return sim_df, summary_df, bet_log_df, suffixes


def build_bet_outcome_summary(bet_log_df: pd.DataFrame, suffixes: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for suffix in suffixes:
        bet_col = f"bet_side_{suffix}"
        pnl_col = f"pnl_kelly_{suffix}"
        win_col = f"bet_win_{suffix}"

        if bet_col not in bet_log_df.columns or pnl_col not in bet_log_df.columns:
            continue

        bets = bet_log_df[bet_log_df[bet_col].fillna("NO BET") != "NO BET"].copy()
        if len(bets) == 0:
            continue

        home_bets = bets[bets[bet_col] == "HOME"]
        away_bets = bets[bets[bet_col] == "AWAY"]

        row = {
            "model": suffix,
            "total_bets": int(len(bets)),
            "total_pnl": float(bets[pnl_col].sum()),
            "avg_pnl_per_bet": float(bets[pnl_col].mean()),
            "std_pnl_per_bet": float(bets[pnl_col].std(ddof=0)),
            "home_bets": int(len(home_bets)),
            "home_pnl": float(home_bets[pnl_col].sum()) if len(home_bets) else 0.0,
            "home_avg_pnl": float(home_bets[pnl_col].mean()) if len(home_bets) else np.nan,
            "away_bets": int(len(away_bets)),
            "away_pnl": float(away_bets[pnl_col].sum()) if len(away_bets) else 0.0,
            "away_avg_pnl": float(away_bets[pnl_col].mean()) if len(away_bets) else np.nan,
        }

        if win_col in bets.columns:
            row["win_rate"] = float((bets[win_col] == 1).mean())
            row["home_win_rate"] = float((home_bets[win_col] == 1).mean()) if len(home_bets) else np.nan
            row["away_win_rate"] = float((away_bets[win_col] == 1).mean()) if len(away_bets) else np.nan

        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("total_pnl", ascending=False)


def _plot_bankrolls(sim_df: pd.DataFrame, suffixes: List[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in suffixes:
        bankroll_col = f"bankroll_{suffix}"
        drawdown_col = f"drawdown_{suffix}"
        if bankroll_col not in sim_df.columns or drawdown_col not in sim_df.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sim_df["GAME_DATE"], sim_df[bankroll_col], label="Bankroll")
        ax.set_title(f"Bankroll Trajectory - {suffix}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Bankroll")
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(sim_df["GAME_DATE"], sim_df[drawdown_col], color="tab:red", alpha=0.6, label="Drawdown")
        ax2.set_ylabel("Drawdown")

        fig.tight_layout()
        fig.savefig(out_dir / f"bankroll_{suffix}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)


def _plot_bet_outcomes(bet_log_df: pd.DataFrame, suffixes: List[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for suffix in suffixes:
        bet_col = f"bet_side_{suffix}"
        pnl_col = f"pnl_kelly_{suffix}"
        if bet_col not in bet_log_df.columns or pnl_col not in bet_log_df.columns:
            continue

        bets = bet_log_df[bet_log_df[bet_col].fillna("NO BET") != "NO BET"].copy()
        if len(bets) == 0:
            continue

        home = bets.loc[bets[bet_col] == "HOME", pnl_col].to_numpy(dtype=float)
        away = bets.loc[bets[bet_col] == "AWAY", pnl_col].to_numpy(dtype=float)

        # side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        if len(home) > 0:
            axes[0].hist(home, bins=30, color="tab:blue", alpha=0.7, edgecolor="black")
            axes[0].axvline(home.mean(), color="red", linestyle="--", linewidth=1.5)
            axes[0].set_title(f"{suffix} - HOME (n={len(home)})")
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, "No HOME bets", ha="center", va="center", transform=axes[0].transAxes)

        if len(away) > 0:
            axes[1].hist(away, bins=30, color="tab:orange", alpha=0.7, edgecolor="black")
            axes[1].axvline(away.mean(), color="red", linestyle="--", linewidth=1.5)
            axes[1].set_title(f"{suffix} - AWAY (n={len(away)})")
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, "No AWAY bets", ha="center", va="center", transform=axes[1].transAxes)

        fig.tight_layout()
        fig.savefig(out_dir / f"bet_outcomes_{suffix}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

        # combined overlay
        fig, ax = plt.subplots(figsize=(10, 6))
        if len(home) > 0:
            ax.hist(home, bins=30, alpha=0.6, color="tab:blue", label=f"HOME (n={len(home)})", edgecolor="black")
        if len(away) > 0:
            ax.hist(away, bins=30, alpha=0.6, color="tab:orange", label=f"AWAY (n={len(away)})", edgecolor="black")
        ax.axvline(0.0, color="red", linestyle=":", linewidth=2)
        ax.set_title(f"{suffix} - Bet Outcomes")
        ax.set_xlabel("P&L (Kelly)")
        ax.set_ylabel("Frequency")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"combined_outcomes_{suffix}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)


def _plot_model_comparison(summary_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(summary_df) == 0:
        return

    plot_df = summary_df.copy()
    plot_df = plot_df.sort_values("roi", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(plot_df["model"], plot_df["roi"], color="tab:green", alpha=0.8)
    axes[0].set_title("ROI by Model")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, axis="y", alpha=0.3)

    if "win_rate" in plot_df.columns:
        axes[1].bar(plot_df["model"], plot_df["win_rate"], color="tab:purple", alpha=0.8)
        axes[1].set_title("Win Rate by Model")
        axes[1].set_ylim(0, 1)
    else:
        axes[1].bar(plot_df["model"], plot_df["total_pnl"], color="tab:orange", alpha=0.8)
        axes[1].set_title("Total PnL by Model")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def _objective_score(edge: pd.Series, kelly: pd.Series, objective_type: str, objective_lambda: float) -> pd.Series:
    edge_abs = edge.abs().fillna(0.0)
    if objective_type == "log_growth_dd":
        denom = 1.0 + objective_lambda * kelly.fillna(0.0)
        denom = denom.replace(0, np.nan).fillna(1.0)
        return edge_abs / denom
    return edge_abs


def _simulate_variant_for_model(
    model_df: pd.DataFrame,
    suffix: str,
    spec: ExperimentSpec,
    starting_bankroll: float,
    default_payout: float = 100.0 / 110.0,
) -> Dict[str, object]:
    bet_col = f"bet_side_{suffix}"
    win_col = f"bet_win_{suffix}"
    kelly_col = f"kelly_frac_{suffix}"
    edge_col = f"edge_{suffix}"

    required = [bet_col, win_col, kelly_col]
    if any(c not in model_df.columns for c in required):
        return {
            "experiment_id": spec.experiment_id,
            "model": suffix,
            "status": "missing_columns",
        }

    work = model_df.copy()
    work["GAME_DATE"] = pd.to_datetime(work["GAME_DATE"]).dt.normalize()
    work = work.sort_values("GAME_DATE").reset_index(drop=True)

    payout_home = pd.to_numeric(work.get("payout_home", default_payout), errors="coerce").fillna(default_payout)
    payout_away = pd.to_numeric(work.get("payout_away", default_payout), errors="coerce").fillna(default_payout)
    payout_home = payout_home.where(payout_home > 0, default_payout)
    payout_away = payout_away.where(payout_away > 0, default_payout)

    kelly_raw = pd.to_numeric(work[kelly_col], errors="coerce").fillna(0.0)
    kelly_adj = np.clip(kelly_raw * spec.kelly_scale, 0.0, spec.kelly_cap)
    work["_kelly_adj"] = kelly_adj

    candidate = work[bet_col].isin(["HOME", "AWAY"]) & (work["_kelly_adj"] > 0)

    if spec.exclude_default_payout and "used_default_payout" in work.columns:
        candidate &= ~work["used_default_payout"].fillna(False).astype(bool)

    if spec.allowed_market_tiers is not None and "market_match_quality" in work.columns:
        candidate &= work["market_match_quality"].isin(spec.allowed_market_tiers)

    edge = pd.to_numeric(work.get(edge_col, 0.0), errors="coerce").fillna(0.0)
    score = _objective_score(edge=edge, kelly=work["_kelly_adj"], objective_type=spec.objective_type, objective_lambda=spec.objective_lambda)
    work["_score"] = score

    work["_rank"] = np.nan
    work.loc[candidate, "_rank"] = (
        work.loc[candidate]
        .groupby("GAME_DATE")["_score"]
        .rank(method="first", ascending=False)
    )

    selected = candidate & (work["_rank"] <= int(spec.top_n_bets_per_day))
    work["_selected"] = selected

    pre_exposure = pd.Series(np.where(selected, work["_kelly_adj"], 0.0), index=work.index).groupby(work["GAME_DATE"]).transform("sum")
    scale = np.where(pre_exposure > spec.daily_max_exposure, spec.daily_max_exposure / pre_exposure.replace(0, np.nan), 1.0)
    scale = pd.Series(scale, index=work.index).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    work["_kelly_final"] = np.clip(np.where(selected, work["_kelly_adj"] * scale, 0.0), 0.0, spec.kelly_cap)

    pnl = np.zeros(len(work), dtype=float)
    home_mask = work[bet_col] == "HOME"
    away_mask = work[bet_col] == "AWAY"
    wins = work[win_col].fillna(0).astype(int) == 1

    home_win = selected & home_mask & wins
    away_win = selected & away_mask & wins
    losses = selected & (~wins)

    pnl[home_win] = work.loc[home_win, "_kelly_final"].to_numpy() * payout_home.loc[home_win].to_numpy()
    pnl[away_win] = work.loc[away_win, "_kelly_final"].to_numpy() * payout_away.loc[away_win].to_numpy()
    pnl[losses] = -work.loc[losses, "_kelly_final"].to_numpy()

    if spec.max_daily_loss is not None:
        pnl_adj = pnl.copy()
        for game_date, idx in work.groupby("GAME_DATE").groups.items():
            day_idx = list(idx)
            if not day_idx:
                continue
            cum_day = 0.0
            for i in day_idx:
                if cum_day <= spec.max_daily_loss:
                    pnl_adj[i] = 0.0
                    continue
                cum_day += pnl_adj[i]
        pnl = pnl_adj

    growth = 1.0 + pnl
    growth = np.clip(growth, 1e-12, None)
    bankroll = pd.Series(growth).cumprod() * float(starting_bankroll)
    drawdown = (bankroll / bankroll.cummax().replace(0, np.nan)) - 1.0

    bet_count = int(selected.sum())
    win_rate = float(work.loc[selected, win_col].mean()) if bet_count > 0 else np.nan
    home_selected = int((selected & home_mask).sum())
    away_selected = int((selected & away_mask).sum())

    return {
        "experiment_id": spec.experiment_id,
        "model": suffix,
        "status": "ok",
        "objective_type": spec.objective_type,
        "objective_lambda": spec.objective_lambda,
        "kelly_scale": spec.kelly_scale,
        "kelly_cap": spec.kelly_cap,
        "top_n_bets_per_day": spec.top_n_bets_per_day,
        "daily_max_exposure": spec.daily_max_exposure,
        "exclude_default_payout": spec.exclude_default_payout,
        "allowed_market_tiers": "|".join(spec.allowed_market_tiers) if spec.allowed_market_tiers else "",
        "max_daily_loss": spec.max_daily_loss,
        "bets_placed": bet_count,
        "home_bets": home_selected,
        "away_bets": away_selected,
        "win_rate": win_rate,
        "total_pnl": float(np.sum(pnl)),
        "avg_pnl_per_bet": float(np.mean(pnl[selected])) if bet_count > 0 else np.nan,
        "avg_kelly": float(np.mean(work.loc[selected, "_kelly_final"])) if bet_count > 0 else 0.0,
        "ending_bankroll": float(bankroll.iloc[-1]) if len(bankroll) else float(starting_bankroll),
        "roi": float((bankroll.iloc[-1] - starting_bankroll) / starting_bankroll) if len(bankroll) else 0.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
    }


def run_experiment_matrix(model_df: pd.DataFrame, suffixes: List[str], starting_bankroll: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    for spec in DEFAULT_EXPERIMENT_MATRIX:
        for suffix in suffixes:
            rows.append(_simulate_variant_for_model(model_df, suffix, spec, starting_bankroll=starting_bankroll))

    exp_df = pd.DataFrame(rows)
    ok_df = exp_df[exp_df["status"] == "ok"].copy() if len(exp_df) else pd.DataFrame()
    if len(ok_df) == 0:
        return exp_df, pd.DataFrame()

    best_per_experiment = (
        ok_df.sort_values(["experiment_id", "roi"], ascending=[True, False])
        .groupby("experiment_id", as_index=False)
        .head(1)
        .rename(columns={"model": "best_model", "roi": "best_roi", "max_drawdown": "best_model_max_drawdown"})
    )

    aggregate = (
        ok_df.groupby("experiment_id", as_index=False)
        .agg(
            avg_roi=("roi", "mean"),
            median_roi=("roi", "median"),
            avg_max_drawdown=("max_drawdown", "mean"),
            total_bets=("bets_placed", "sum"),
            avg_win_rate=("win_rate", "mean"),
        )
    )

    summary = aggregate.merge(best_per_experiment[["experiment_id", "best_model", "best_roi", "best_model_max_drawdown"]], on="experiment_id", how="left")
    summary = summary.sort_values("avg_roi", ascending=False)
    return exp_df, summary


def build_market_quality_diagnostics(model_df: pd.DataFrame, suffixes: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows_default: List[Dict[str, object]] = []
    rows_tier: List[Dict[str, object]] = []

    has_default_col = "used_default_payout" in model_df.columns
    has_tier_col = "market_match_quality" in model_df.columns

    for suffix in suffixes:
        bet_col = f"bet_side_{suffix}"
        pnl_col = f"pnl_kelly_{suffix}"
        win_col = f"bet_win_{suffix}"
        if bet_col not in model_df.columns or pnl_col not in model_df.columns:
            continue

        bets = model_df[model_df[bet_col].fillna("NO BET") != "NO BET"].copy()
        if len(bets) == 0:
            continue

        if has_default_col:
            for default_flag, gdf in bets.groupby("used_default_payout", dropna=False):
                row = {
                    "model": suffix,
                    "used_default_payout": bool(default_flag) if pd.notna(default_flag) else np.nan,
                    "bets": int(len(gdf)),
                    "total_pnl": float(gdf[pnl_col].sum()),
                    "avg_pnl_per_bet": float(gdf[pnl_col].mean()),
                }
                if win_col in gdf.columns:
                    row["win_rate"] = float(gdf[win_col].mean())
                rows_default.append(row)

        if has_tier_col:
            for tier, gdf in bets.groupby("market_match_quality", dropna=False):
                row = {
                    "model": suffix,
                    "market_match_quality": str(tier),
                    "bets": int(len(gdf)),
                    "total_pnl": float(gdf[pnl_col].sum()),
                    "avg_pnl_per_bet": float(gdf[pnl_col].mean()),
                }
                if win_col in gdf.columns:
                    row["win_rate"] = float(gdf[win_col].mean())
                rows_tier.append(row)

    default_df = pd.DataFrame(rows_default)
    tier_df = pd.DataFrame(rows_tier)

    if len(default_df) > 0:
        default_df = default_df.sort_values(["model", "used_default_payout"], ascending=[True, True])
    if len(tier_df) > 0:
        tier_df = tier_df.sort_values(["model", "market_match_quality"], ascending=[True, True])

    return default_df, tier_df


def _plot_experiment_matrix(exp_df: pd.DataFrame, out_dir: Path) -> None:
    if len(exp_df) == 0:
        return
    ok_df = exp_df[exp_df["status"] == "ok"].copy()
    if len(ok_df) == 0:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    best = ok_df.sort_values(["experiment_id", "roi"], ascending=[True, False]).groupby("experiment_id", as_index=False).head(1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(best["experiment_id"], best["roi"], color="tab:green", alpha=0.8)
    axes[0].set_title("Best Model ROI by Experiment")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(best["experiment_id"], best["max_drawdown"], color="tab:red", alpha=0.8)
    axes[1].set_title("Best Model Max Drawdown by Experiment")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "experiment_matrix_comparison.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def _determine_run_folder(outputs_dir: Path, selected_inputs: Dict[str, Path], run_date: Optional[str]) -> Path:
    if run_date:
        return outputs_dir / f"copilot_analysis_odds_v{run_date}"

    model_path = selected_inputs["model_output"]
    parsed = _parse_version_date(model_path)
    if parsed is not None:
        return outputs_dir / f"copilot_analysis_odds_v{parsed.strftime('%Y_%m_%d')}"

    return outputs_dir / "copilot_analysis_latest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze latest copilot model outputs and generate stats/plots.")
    parser.add_argument("--outputs-dir", default="outputs", help="Directory containing copilot output CSV files.")
    parser.add_argument("--starting-bankroll", type=float, default=10000.0, help="Starting bankroll for simulation.")
    parser.add_argument("--run-date", default=None, help="Optional run date tag YYYY_MM_DD for output folder naming.")

    parser.add_argument("--prefer-odds", dest="prefer_odds", action="store_true", default=True)
    parser.add_argument("--no-prefer-odds", dest="prefer_odds", action="store_false")

    parser.add_argument("--allow-fallback", dest="allow_fallback", action="store_true", default=True)
    parser.add_argument("--no-allow-fallback", dest="allow_fallback", action="store_false")

    parser.add_argument("--strict-schema", dest="strict_schema", action="store_true", default=True)
    parser.add_argument("--no-strict-schema", dest="strict_schema", action="store_false")

    parser.add_argument("--input-model-output", default=None, help="Optional explicit path override for model output CSV.")
    parser.add_argument("--input-model-rmse", default=None, help="Optional explicit path override for model RMSE CSV.")
    parser.add_argument("--input-join-audit", default=None, help="Optional explicit path override for join audit CSV.")
    parser.add_argument("--input-objective-sweep", default=None, help="Optional explicit path override for objective sweep CSV.")

    parser.add_argument("--run-experiment-matrix", dest="run_experiment_matrix", action="store_true", default=True)
    parser.add_argument("--no-run-experiment-matrix", dest="run_experiment_matrix", action="store_false")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)

    selected_inputs, manifest_df = discover_inputs(
        outputs_dir=outputs_dir,
        prefer_odds=args.prefer_odds,
        allow_fallback=args.allow_fallback,
    )

    overrides = {
        "model_output": args.input_model_output,
        "model_rmse": args.input_model_rmse,
        "join_audit": args.input_join_audit,
        "objective_sweep": args.input_objective_sweep,
    }
    for key, override in overrides.items():
        if override:
            selected_inputs[key] = Path(override)
            manifest_df.loc[manifest_df["input_key"] == key, "selected_path"] = str(override)
            manifest_df.loc[manifest_df["input_key"] == key, "selection_source"] = "manual_override"

    run_folder = _determine_run_folder(outputs_dir, selected_inputs, args.run_date)
    run_folder.mkdir(parents=True, exist_ok=True)

    model_df = pd.read_csv(selected_inputs["model_output"])
    schema_df = validate_schema(model_df, strict_schema=args.strict_schema)

    sim_df, summary_df, bet_log_df, suffixes = build_bankroll_outputs(model_df, starting_bankroll=args.starting_bankroll)
    outcome_df = build_bet_outcome_summary(bet_log_df, suffixes)
    market_default_df, market_tier_df = build_market_quality_diagnostics(model_df, suffixes)

    # Optional direct copies for contextual files if present
    for key in ("model_rmse", "join_audit", "objective_sweep"):
        src = selected_inputs.get(key)
        if src and src.exists():
            dest_name = {
                "model_rmse": "model_rmse_selected.csv",
                "join_audit": "model_join_audit_selected.csv",
                "objective_sweep": "objective_sweep_selected.csv",
            }[key]
            pd.read_csv(src).to_csv(run_folder / dest_name, index=False)

    sim_df.to_csv(run_folder / "copilot_betting_sim.csv", index=False)
    summary_df.to_csv(run_folder / "copilot_betting_summary.csv", index=False)
    bet_log_df.to_csv(run_folder / "copilot_bet_logs.csv", index=False)
    outcome_df.to_csv(run_folder / "bet_outcomes_summary.csv", index=False)
    market_default_df.to_csv(run_folder / "market_default_split_summary.csv", index=False)
    market_tier_df.to_csv(run_folder / "market_tier_split_summary.csv", index=False)
    manifest_df.to_csv(run_folder / "input_selection_manifest.csv", index=False)
    schema_df.to_csv(run_folder / "schema_validation_report.csv", index=False)

    _plot_bankrolls(sim_df, suffixes, run_folder / "bet_plots")
    _plot_bet_outcomes(bet_log_df, suffixes, run_folder / "bet_outcome_histograms")
    _plot_model_comparison(summary_df, run_folder)

    if args.run_experiment_matrix:
        exp_df, exp_summary_df = run_experiment_matrix(model_df=model_df, suffixes=suffixes, starting_bankroll=args.starting_bankroll)
        exp_df.to_csv(run_folder / "experiment_matrix_results.csv", index=False)
        exp_summary_df.to_csv(run_folder / "experiment_matrix_summary.csv", index=False)
        _plot_experiment_matrix(exp_df, run_folder)

    print(f"Analysis complete. Artifacts written to: {run_folder}")
    print(f"Models analyzed: {', '.join(suffixes)}")


if __name__ == "__main__":
    main()
