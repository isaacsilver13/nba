"""Enhanced copilot output analysis with stability experiments, calibration, and volatility regimes."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VERSION_RE = re.compile(r"_odds_v(\d{4}_\d{2}_\d{2})\.csv$", re.IGNORECASE)
DEFAULT_PAYOUT = 100.0 / 110.0
PREFERRED_TUNING_MODELS: Tuple[str, ...] = ("linear_regression", "elasticnet")
PRODUCTION_PROFILE_EXPERIMENT_IDS: Tuple[str, ...] = ("R5_stability_tier_b", "R6_stability_tier_b_c")
DEFAULT_FEATURE_SOURCE = "data/processed/df_model_3.csv"

LEAKY_FEATURE_EXACT: Set[str] = {
    "home_margin",
    "away_margin",
    "did_home_cover",
    "team_that_covered",
    "favorite_margin",
    "favorite_cover",
    "favorite_cover_numeric",
    "market_error",
    "abs_edge",
    "edge_rank_today",
    "edge_percentile_last_30_days",
    "model_error_roll_last20",
}

LEAKY_FEATURE_PREFIXES: Tuple[str, ...] = (
    "pnl_",
    "cum_pnl_",
    "bet_win",
    "edge_",
    "error_",
)

TEAM_CONTEXT_FEATURE_CANDIDATES: Tuple[str, ...] = (
    "rest_diff",
    "travel_diff_3d",
    "travel_diff_7d",
    "fatigue_diff",
    "elo_diff",
    "net_rating_diff",
    "pace_diff",
    "combined_pace",
    "home_b2b",
    "away_b2b",
    "is_home_favorite",
    "team_style_vs_opponent_style",
)


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
    min_abs_edge: float = 0.0
    use_prob_calibration: bool = True


INPUT_SPECS: Sequence[InputSpec] = (
    InputSpec("model_output", "copilot_model_output_odds_v*.csv", "copilot_model_output.csv", required=True),
    InputSpec("model_rmse", "copilot_model_rmse_odds_v*.csv", "copilot_model_rmse.csv"),
    InputSpec("join_audit", "copilot_model_join_audit_odds_v*.csv", "copilot_model_join_audit.csv"),
    InputSpec("objective_sweep", "copilot_objective_sweep_odds_v*.csv", "copilot_objective_sweep.csv"),
)


REQUIRED_BASE_COLUMNS = ["GAME_DATE"]
OPTIONAL_BASE_COLUMNS = [
    "GAME_ID",
    "season",
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
    ExperimentSpec("R0_baseline", 1.00, 0.10, 2, 0.25, objective_type="roi", objective_lambda=0.0),
    ExperimentSpec("R1_conservative", 0.25, 0.01, 2, 0.03, objective_type="roi", objective_lambda=0.0),
    ExperimentSpec("R2_ultra_risk", 0.10, 0.005, 1, 0.01, objective_type="roi", objective_lambda=0.0, max_daily_loss=-0.02),
    ExperimentSpec(
        "R3_market_gated",
        0.25,
        0.01,
        2,
        0.03,
        objective_type="roi",
        objective_lambda=0.0,
        exclude_default_payout=True,
        allowed_market_tiers=("tier_b_moneyline",),
    ),
    ExperimentSpec(
        "R4_objective_lgdd_025",
        0.25,
        0.01,
        2,
        0.03,
        objective_type="log_growth_dd",
        objective_lambda=0.25,
    ),
    ExperimentSpec(
        "R4_objective_lgdd_050",
        0.25,
        0.01,
        2,
        0.03,
        objective_type="log_growth_dd",
        objective_lambda=0.50,
    ),
    ExperimentSpec(
        "R5_stability_tier_b",
        0.05,
        0.003,
        1,
        0.005,
        objective_type="log_growth_dd",
        objective_lambda=0.50,
        exclude_default_payout=True,
        allowed_market_tiers=("tier_b_moneyline",),
        max_daily_loss=-0.01,
        min_abs_edge=4.5,
        use_prob_calibration=True,
    ),
    ExperimentSpec(
        "R6_stability_tier_b_c",
        0.10,
        0.005,
        1,
        0.008,
        objective_type="log_growth_dd",
        objective_lambda=0.35,
        exclude_default_payout=True,
        allowed_market_tiers=("tier_b_moneyline", "tier_c_mirrored"),
        max_daily_loss=-0.015,
        min_abs_edge=4.0,
        use_prob_calibration=True,
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


def _parse_vol_quantiles(raw: str) -> Tuple[float, float]:
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("--vol-quantiles must be two comma-separated values like '0.33,0.67'.")

    q1 = float(parts[0])
    q2 = float(parts[1])
    if not (0.0 < q1 < q2 < 1.0):
        raise ValueError(f"Invalid --vol-quantiles: got ({q1}, {q2}). Must satisfy 0 < q1 < q2 < 1.")
    return q1, q2


def _validate_override_paths(overrides: Dict[str, Optional[str]]) -> None:
    missing = []
    for key, value in overrides.items():
        if not value:
            continue
        p = Path(value)
        if not p.exists():
            missing.append((key, value))
    if missing:
        details = "; ".join([f"{k}='{v}'" for k, v in missing])
        raise FileNotFoundError(f"Override path(s) not found: {details}")


def discover_inputs(outputs_dir: Path, prefer_odds: bool = True, allow_fallback: bool = True) -> Tuple[Dict[str, Path], pd.DataFrame]:
    selected: Dict[str, Path] = {}
    rows: List[Dict[str, object]] = []

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

        rows.append(
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
                f"Missing required input '{spec.key}'. Checked '{spec.odds_glob}' and '{spec.fallback_name}'."
            )

    return selected, pd.DataFrame(rows)


def _get_model_suffixes(df: pd.DataFrame) -> List[str]:
    return [c.replace("pnl_kelly_", "") for c in df.columns if c.startswith("pnl_kelly_")]


def validate_schema(model_df: pd.DataFrame, strict_schema: bool = True) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for c in REQUIRED_BASE_COLUMNS:
        rows.append({"column": c, "is_hard_required": True, "present": c in model_df.columns, "severity": "error"})

    for c in OPTIONAL_BASE_COLUMNS:
        rows.append({"column": c, "is_hard_required": False, "present": c in model_df.columns, "severity": "warning"})

    has_pnl = len(_get_model_suffixes(model_df)) > 0
    rows.append({"column": "pnl_kelly_*", "is_hard_required": True, "present": has_pnl, "severity": "error"})

    out = pd.DataFrame(rows)
    if strict_schema:
        bad = out[(out["is_hard_required"] == True) & (out["present"] == False)]  # noqa: E712
        if len(bad):
            raise ValueError(f"Missing hard-required schema: {', '.join(bad['column'].tolist())}")
    return out


def _compute_bankroll_series(df: pd.DataFrame, pnl_col: str, starting_bankroll: float) -> pd.Series:
    pnl = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    bankroll = np.empty(len(pnl), dtype=float)
    current = float(starting_bankroll)
    for i, step in enumerate(pnl):
        current = current * (1.0 + step)
        bankroll[i] = current
    return pd.Series(bankroll, index=df.index)


def _compute_drawdown(series: pd.Series) -> pd.Series:
    peak = series.cummax().replace(0, np.nan)
    return (series - peak) / peak


def _build_bet_log(df: pd.DataFrame, suffixes: List[str]) -> pd.DataFrame:
    bet_mask = pd.Series(False, index=df.index)
    for suffix in suffixes:
        bcol = f"bet_side_{suffix}"
        pcol = f"pnl_kelly_{suffix}"
        if bcol in df.columns:
            bet_mask |= df[bcol].fillna("NO BET") != "NO BET"
        elif pcol in df.columns:
            bet_mask |= pd.to_numeric(df[pcol], errors="coerce").fillna(0.0) != 0

    keep_cols = [c for c in OPTIONAL_BASE_COLUMNS if c in df.columns]
    if "GAME_DATE" in df.columns and "GAME_DATE" not in keep_cols:
        keep_cols = ["GAME_DATE"] + keep_cols

    extra_cols: List[str] = []
    for suffix in suffixes:
        extra_cols.extend(
            [
                c
                for c in [
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
                if c in df.columns
            ]
        )

    cols = keep_cols + [c for c in extra_cols if c not in keep_cols]
    return df.loc[bet_mask, cols].reset_index(drop=True)


def build_bankroll_outputs(model_df: pd.DataFrame, starting_bankroll: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    df = model_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    suffixes = _get_model_suffixes(df)
    if not suffixes:
        raise ValueError("No pnl_kelly_* columns found.")

    base_cols = [c for c in ["GAME_DATE", "GAME_ID", "home", "away", "home_margin", "spread_signed"] if c in df.columns]
    sim_df = df[base_cols].copy()
    rows: List[Dict[str, object]] = []

    for suffix in suffixes:
        pcol = f"pnl_kelly_{suffix}"
        bcol = f"bet_side_{suffix}"
        wcol = f"bet_win_{suffix}"
        ecol = f"edge_{suffix}"
        ehcol = f"ev_home_{suffix}"
        eacol = f"ev_away_{suffix}"

        bankroll = _compute_bankroll_series(df, pcol, starting_bankroll)
        drawdown = _compute_drawdown(bankroll)

        sim_df[f"bankroll_{suffix}"] = bankroll
        sim_df[f"drawdown_{suffix}"] = drawdown

        if bcol in df.columns:
            bet_mask = df[bcol].fillna("NO BET") != "NO BET"
        else:
            bet_mask = pd.to_numeric(df[pcol], errors="coerce").fillna(0.0) != 0

        total_bets = int(bet_mask.sum())
        win_rate = float(df.loc[bet_mask, wcol].mean()) if wcol in df.columns and total_bets > 0 else np.nan
        avg_edge = float(df.loc[bet_mask, ecol].mean()) if ecol in df.columns and total_bets > 0 else np.nan

        avg_ev = np.nan
        if ehcol in df.columns and eacol in df.columns and bcol in df.columns and total_bets > 0:
            ev_sel = np.where(df[bcol] == "HOME", df[ehcol], np.where(df[bcol] == "AWAY", df[eacol], np.nan))
            avg_ev = float(pd.Series(ev_sel, index=df.index)[bet_mask].astype(float).mean())

        start = float(starting_bankroll)
        end = float(bankroll.iloc[-1]) if len(bankroll) else start

        rows.append(
            {
                "model": suffix,
                "starting_bankroll": start,
                "ending_bankroll": end,
                "roi": (end - start) / start if start else np.nan,
                "total_pnl": float(pd.to_numeric(df[pcol], errors="coerce").fillna(0.0).sum()),
                "num_bets": total_bets,
                "win_rate": win_rate,
                "avg_edge": avg_edge,
                "avg_ev_selected_side": avg_ev,
                "max_drawdown": float(drawdown.min()) if len(drawdown) else np.nan,
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("roi", ascending=False)
    bet_log_df = _build_bet_log(df, suffixes)
    return sim_df, summary_df, bet_log_df, suffixes


def build_bet_outcome_summary(bet_log_df: pd.DataFrame, suffixes: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for suffix in suffixes:
        bcol = f"bet_side_{suffix}"
        pcol = f"pnl_kelly_{suffix}"
        wcol = f"bet_win_{suffix}"
        if bcol not in bet_log_df.columns or pcol not in bet_log_df.columns:
            continue

        bets = bet_log_df[bet_log_df[bcol].fillna("NO BET") != "NO BET"].copy()
        if len(bets) == 0:
            continue

        home = bets[bets[bcol] == "HOME"]
        away = bets[bets[bcol] == "AWAY"]

        row = {
            "model": suffix,
            "total_bets": int(len(bets)),
            "total_pnl": float(bets[pcol].sum()),
            "avg_pnl_per_bet": float(bets[pcol].mean()),
            "std_pnl_per_bet": float(bets[pcol].std(ddof=0)),
            "home_bets": int(len(home)),
            "home_pnl": float(home[pcol].sum()) if len(home) else 0.0,
            "home_avg_pnl": float(home[pcol].mean()) if len(home) else np.nan,
            "away_bets": int(len(away)),
            "away_pnl": float(away[pcol].sum()) if len(away) else 0.0,
            "away_avg_pnl": float(away[pcol].mean()) if len(away) else np.nan,
        }
        if wcol in bets.columns:
            row["win_rate"] = float((bets[wcol] == 1).mean())
            row["home_win_rate"] = float((home[wcol] == 1).mean()) if len(home) else np.nan
            row["away_win_rate"] = float((away[wcol] == 1).mean()) if len(away) else np.nan

        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("total_pnl", ascending=False)


def build_market_quality_diagnostics(model_df: pd.DataFrame, suffixes: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    default_rows: List[Dict[str, object]] = []
    tier_rows: List[Dict[str, object]] = []
    split2d_rows: List[Dict[str, object]] = []

    has_default = "used_default_payout" in model_df.columns
    has_tier = "market_match_quality" in model_df.columns

    for suffix in suffixes:
        bcol = f"bet_side_{suffix}"
        pcol = f"pnl_kelly_{suffix}"
        wcol = f"bet_win_{suffix}"
        if bcol not in model_df.columns or pcol not in model_df.columns:
            continue

        bets = model_df[model_df[bcol].fillna("NO BET") != "NO BET"].copy()
        if len(bets) == 0:
            continue

        if has_default:
            for flag, g in bets.groupby("used_default_payout", dropna=False):
                row = {
                    "model": suffix,
                    "used_default_payout": bool(flag) if pd.notna(flag) else np.nan,
                    "bets": int(len(g)),
                    "total_pnl": float(g[pcol].sum()),
                    "avg_pnl_per_bet": float(g[pcol].mean()),
                }
                if wcol in g.columns:
                    row["win_rate"] = float(g[wcol].mean())
                default_rows.append(row)

        if has_tier:
            for tier, g in bets.groupby("market_match_quality", dropna=False):
                row = {
                    "model": suffix,
                    "market_match_quality": str(tier),
                    "bets": int(len(g)),
                    "total_pnl": float(g[pcol].sum()),
                    "avg_pnl_per_bet": float(g[pcol].mean()),
                }
                if wcol in g.columns:
                    row["win_rate"] = float(g[wcol].mean())
                tier_rows.append(row)

        if has_default and has_tier:
            for (flag, tier), g in bets.groupby(["used_default_payout", "market_match_quality"], dropna=False):
                row = {
                    "model": suffix,
                    "used_default_payout": bool(flag) if pd.notna(flag) else np.nan,
                    "market_match_quality": str(tier),
                    "bets": int(len(g)),
                    "total_pnl": float(g[pcol].sum()),
                    "avg_pnl_per_bet": float(g[pcol].mean()),
                }
                if wcol in g.columns:
                    row["win_rate"] = float(g[wcol].mean())
                split2d_rows.append(row)

    return pd.DataFrame(default_rows), pd.DataFrame(tier_rows), pd.DataFrame(split2d_rows)


def build_side_counts_summary(outcome_df: pd.DataFrame) -> pd.DataFrame:
    if len(outcome_df) == 0:
        return pd.DataFrame()
    out = outcome_df.copy()
    out["home_wins"] = np.round(out["home_bets"] * out.get("home_win_rate", np.nan), 0)
    out["away_wins"] = np.round(out["away_bets"] * out.get("away_win_rate", np.nan), 0)
    cols = ["model", "home_bets", "away_bets", "home_wins", "away_wins"]
    return out[cols]


def _canonical_game_id_series(values: pd.Series) -> pd.Series:
    s = values.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.replace(r"^0+", "", regex=True)
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return s


def _is_leaky_feature_column(column: str) -> bool:
    c = str(column).strip().lower()
    if c in LEAKY_FEATURE_EXACT:
        return True
    return any(c.startswith(prefix) for prefix in LEAKY_FEATURE_PREFIXES)


def _build_feature_join(
    model_df: pd.DataFrame,
    feature_source: Path,
    feature_join_key: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not feature_source.exists():
        raise FileNotFoundError(f"Feature source not found: {feature_source}")

    feat_df = pd.read_csv(feature_source)
    if len(feat_df) == 0:
        return model_df.copy(), pd.DataFrame(), pd.DataFrame()

    out_df = model_df.copy()

    model_key_source = "GAME_ID_KEY" if "GAME_ID_KEY" in out_df.columns else "GAME_ID"
    if model_key_source not in out_df.columns:
        raise ValueError("Model output missing GAME_ID/GAME_ID_KEY required for feature join.")

    if feature_join_key in feat_df.columns:
        feat_key_source = feature_join_key
    elif "GAME_ID" in feat_df.columns:
        feat_key_source = "GAME_ID"
    else:
        raise ValueError(f"Feature source missing join key '{feature_join_key}' and GAME_ID fallback.")

    out_df["_join_key"] = _canonical_game_id_series(out_df[model_key_source])
    feat_df["_join_key"] = _canonical_game_id_series(feat_df[feat_key_source])

    feat_df = feat_df[feat_df["_join_key"].notna()].copy()
    before_dedup = len(feat_df)
    feat_df = feat_df.sort_values("_join_key").drop_duplicates(subset=["_join_key"], keep="first")
    duplicates_removed = int(before_dedup - len(feat_df))

    feature_cols = [
        c
        for c in feat_df.columns
        if c not in {"_join_key", "GAME_ID", "GAME_ID_KEY", "date", "date_x", "season", "home_margin", "away_margin"}
        and c not in out_df.columns
    ]
    feat_keep = feat_df[["_join_key"] + feature_cols].copy()

    merged = out_df.merge(feat_keep, on="_join_key", how="left")

    matched = int(merged["_join_key"].notna().sum())
    match_non_null = int(merged[feature_cols].notna().any(axis=1).sum()) if feature_cols else 0
    join_report = pd.DataFrame(
        [
            {
                "model_rows": int(len(out_df)),
                "feature_rows_raw": int(before_dedup),
                "feature_rows_unique_key": int(len(feat_df)),
                "duplicates_removed": duplicates_removed,
                "rows_with_join_key": matched,
                "rows_with_any_feature_match": match_non_null,
                "feature_match_rate": float(match_non_null / len(out_df)) if len(out_df) else np.nan,
                "model_key_source": model_key_source,
                "feature_key_source": feat_key_source,
                "feature_source": str(feature_source),
            }
        ]
    )

    manifest_rows: List[Dict[str, object]] = []
    for c in feature_cols:
        series = merged[c]
        is_numeric = bool(pd.api.types.is_numeric_dtype(series))
        is_leaky = _is_leaky_feature_column(c)
        include = is_numeric and (not is_leaky)
        reason = "ok" if include else ("excluded_leakage" if is_leaky else "excluded_non_numeric")
        manifest_rows.append(
            {
                "column": c,
                "is_numeric": is_numeric,
                "is_leaky": is_leaky,
                "include_for_factor_slicing": include,
                "non_null_rows": int(series.notna().sum()),
                "reason": reason,
            }
        )

    manifest_df = pd.DataFrame(manifest_rows).sort_values(["include_for_factor_slicing", "non_null_rows"], ascending=[False, False])
    return merged.drop(columns=["_join_key"], errors="ignore"), join_report, manifest_df


def _build_model_bets_long(model_df: pd.DataFrame, suffixes: List[str]) -> pd.DataFrame:
    base_cols = [
        c
        for c in [
            "GAME_DATE",
            "GAME_ID",
            "season",
            "home",
            "away",
            "spread_signed",
            "payout_source",
            "used_default_payout",
            "market_match_quality",
        ]
        if c in model_df.columns
    ]

    chunks: List[pd.DataFrame] = []
    for suffix in suffixes:
        bcol = f"bet_side_{suffix}"
        pcol = f"pnl_kelly_{suffix}"
        wcol = f"bet_win_{suffix}"
        if bcol not in model_df.columns or pcol not in model_df.columns:
            continue

        cols = base_cols + [
            c
            for c in [
                bcol,
                pcol,
                wcol,
                f"edge_{suffix}",
                f"kelly_frac_{suffix}",
                f"win_prob_home_{suffix}",
                f"ev_home_{suffix}",
                f"ev_away_{suffix}",
            ]
            if c in model_df.columns
        ]
        part = model_df[cols].copy()
        part = part[part[bcol].fillna("NO BET").isin(["HOME", "AWAY"])].copy()
        if len(part) == 0:
            continue

        part["model"] = suffix
        part["bet_side"] = part[bcol].fillna("NO BET")
        part["pnl"] = pd.to_numeric(part[pcol], errors="coerce").fillna(0.0)
        part["bet_win"] = pd.to_numeric(part[wcol], errors="coerce") if wcol in part.columns else np.nan
        part["edge"] = pd.to_numeric(part.get(f"edge_{suffix}"), errors="coerce")
        part["kelly_frac"] = pd.to_numeric(part.get(f"kelly_frac_{suffix}"), errors="coerce")
        part["win_prob_home"] = pd.to_numeric(part.get(f"win_prob_home_{suffix}"), errors="coerce")

        ev_home = pd.to_numeric(part.get(f"ev_home_{suffix}"), errors="coerce")
        ev_away = pd.to_numeric(part.get(f"ev_away_{suffix}"), errors="coerce")
        part["selected_ev"] = np.where(part["bet_side"] == "HOME", ev_home, np.where(part["bet_side"] == "AWAY", ev_away, np.nan))

        chunks.append(part)

    if not chunks:
        return pd.DataFrame()

    long_df = pd.concat(chunks, ignore_index=True)
    if "GAME_DATE" in long_df.columns:
        long_df["GAME_DATE"] = pd.to_datetime(long_df["GAME_DATE"], errors="coerce")
        long_df["month"] = long_df["GAME_DATE"].dt.to_period("M").astype(str)
        long_df["day_of_week"] = long_df["GAME_DATE"].dt.day_name()
    if "season" not in long_df.columns and "GAME_DATE" in long_df.columns:
        long_df["season"] = long_df["GAME_DATE"].dt.year

    spread_signed = pd.to_numeric(long_df.get("spread_signed"), errors="coerce")
    long_df["spread_abs"] = spread_signed.abs()
    long_df["favorite_side"] = np.where(spread_signed < 0, "HOME", np.where(spread_signed > 0, "AWAY", "PICKEM"))
    long_df["bet_type"] = np.where(
        long_df["favorite_side"] == "PICKEM",
        "pickem",
        np.where(long_df["bet_side"] == long_df["favorite_side"], "favorite", "underdog"),
    )
    long_df["spread_bucket"] = pd.cut(
        long_df["spread_abs"],
        bins=[-np.inf, 2, 5, 8, np.inf],
        labels=["0-2", "2-5", "5-8", "8+"],
    ).astype(str)

    edge_abs = pd.to_numeric(long_df.get("edge"), errors="coerce").abs()
    if edge_abs.notna().sum() > 20 and edge_abs.nunique(dropna=True) > 1:
        long_df["edge_decile"] = pd.qcut(edge_abs.rank(method="first"), q=10, labels=False, duplicates="drop")
        long_df["edge_decile"] = long_df["edge_decile"].map(lambda v: f"D{int(v) + 1}" if pd.notna(v) else np.nan)
    else:
        long_df["edge_decile"] = np.nan

    long_df["kelly_tier"] = pd.cut(
        pd.to_numeric(long_df.get("kelly_frac"), errors="coerce"),
        bins=[-np.inf, 0.0025, 0.005, 0.01, 0.02, np.inf],
        labels=["<=0.25%", "0.25-0.5%", "0.5-1%", "1-2%", "2%+"],
    ).astype(str)
    long_df["win_prob_band"] = pd.cut(
        pd.to_numeric(long_df.get("win_prob_home"), errors="coerce"),
        bins=[-np.inf, 0.45, 0.50, 0.55, 0.60, np.inf],
        labels=["<=0.45", "0.45-0.50", "0.50-0.55", "0.55-0.60", "0.60+"],
    ).astype(str)
    long_df["selected_ev_band"] = pd.cut(
        pd.to_numeric(long_df.get("selected_ev"), errors="coerce"),
        bins=[-np.inf, -0.02, 0.0, 0.02, 0.05, np.inf],
        labels=["<-0.02", "-0.02-0", "0-0.02", "0.02-0.05", ">0.05"],
    ).astype(str)
    return long_df


def _summarize_factor(
    bets_long_df: pd.DataFrame,
    factor_group: str,
    factor_name: str,
    factor_col: str,
    min_group_bets: int,
) -> pd.DataFrame:
    if factor_col not in bets_long_df.columns:
        return pd.DataFrame()
    work = bets_long_df.copy()
    work[factor_col] = work[factor_col].astype(str)
    work = work[work[factor_col].notna() & (work[factor_col] != "nan")].copy()
    if len(work) == 0:
        return pd.DataFrame()

    out = (
        work.groupby(["model", factor_col], as_index=False)
        .agg(
            bets=("pnl", "size"),
            total_pnl=("pnl", "sum"),
            avg_pnl_per_bet=("pnl", "mean"),
            win_rate=("bet_win", "mean"),
            avg_edge=("edge", "mean"),
            avg_kelly=("kelly_frac", "mean"),
            avg_selected_ev=("selected_ev", "mean"),
        )
        .rename(columns={factor_col: "factor_level"})
    )
    out["roi_per_bet"] = out["avg_pnl_per_bet"]
    out["factor_group"] = factor_group
    out["factor_name"] = factor_name
    out = out[out["bets"] >= int(min_group_bets)].copy()
    if len(out) == 0:
        return out
    cols = [
        "model",
        "factor_group",
        "factor_name",
        "factor_level",
        "bets",
        "total_pnl",
        "roi_per_bet",
        "avg_pnl_per_bet",
        "win_rate",
        "avg_edge",
        "avg_kelly",
        "avg_selected_ev",
    ]
    return out[cols].sort_values(["model", "total_pnl"], ascending=[True, False]).reset_index(drop=True)


def _rank_profitable_models(summary_df: pd.DataFrame) -> pd.DataFrame:
    if len(summary_df) == 0:
        return pd.DataFrame()
    out = summary_df.copy()
    out["bankroll_growth"] = out["ending_bankroll"] / out["starting_bankroll"]
    out["profitable_flag"] = out["ending_bankroll"] > out["starting_bankroll"]
    out = out.sort_values(["bankroll_growth", "roi", "max_drawdown"], ascending=[False, False, False]).reset_index(drop=True)
    out["profit_rank"] = np.arange(1, len(out) + 1)
    cols = [
        "profit_rank",
        "model",
        "profitable_flag",
        "bankroll_growth",
        "starting_bankroll",
        "ending_bankroll",
        "roi",
        "num_bets",
        "win_rate",
        "avg_edge",
        "avg_ev_selected_side",
        "max_drawdown",
    ]
    return out[cols]


def _deep_diagnostics(
    bets_long_df: pd.DataFrame,
    model_df_for_features: pd.DataFrame,
    min_group_bets: int,
    feature_quantiles: int,
    top_factor_levels: int,
) -> Dict[str, pd.DataFrame]:
    if len(bets_long_df) == 0:
        return {
            "bet_type": pd.DataFrame(),
            "time": pd.DataFrame(),
            "confidence": pd.DataFrame(),
            "data_quality": pd.DataFrame(),
            "team_context": pd.DataFrame(),
        }

    diagnostics: Dict[str, List[pd.DataFrame]] = {
        "bet_type": [],
        "time": [],
        "confidence": [],
        "data_quality": [],
        "team_context": [],
    }

    for name, col in [("bet_side", "bet_side"), ("bet_type", "bet_type"), ("spread_bucket", "spread_bucket")]:
        diagnostics["bet_type"].append(_summarize_factor(bets_long_df, "bet_type", name, col, min_group_bets))

    for name, col in [("season", "season"), ("month", "month"), ("day_of_week", "day_of_week")]:
        diagnostics["time"].append(_summarize_factor(bets_long_df, "time", name, col, min_group_bets))

    for name, col in [
        ("edge_decile", "edge_decile"),
        ("kelly_tier", "kelly_tier"),
        ("win_prob_band", "win_prob_band"),
        ("selected_ev_band", "selected_ev_band"),
    ]:
        diagnostics["confidence"].append(_summarize_factor(bets_long_df, "confidence", name, col, min_group_bets))

    for name, col in [
        ("market_match_quality", "market_match_quality"),
        ("used_default_payout", "used_default_payout"),
        ("payout_source", "payout_source"),
    ]:
        diagnostics["data_quality"].append(_summarize_factor(bets_long_df, "data_quality", name, col, min_group_bets))

    if "GAME_ID" in bets_long_df.columns and "GAME_ID" in model_df_for_features.columns:
        feature_frame = model_df_for_features.copy()
        feature_frame["_join_game_id"] = _canonical_game_id_series(feature_frame["GAME_ID"])
        bet_frame = bets_long_df.copy()
        bet_frame["_join_game_id"] = _canonical_game_id_series(bet_frame["GAME_ID"])
        joined = bet_frame.merge(feature_frame.drop_duplicates("_join_game_id"), on="_join_game_id", how="left", suffixes=("", "_feat"))

        for col in TEAM_CONTEXT_FEATURE_CANDIDATES:
            if col not in joined.columns:
                continue
            if _is_leaky_feature_column(col):
                continue
            if pd.api.types.is_numeric_dtype(joined[col]):
                nz = pd.to_numeric(joined[col], errors="coerce")
                if nz.notna().sum() >= 30 and nz.nunique(dropna=True) > 1:
                    q = min(int(feature_quantiles), int(nz.nunique(dropna=True)))
                    bucket_col = f"{col}_bucket"
                    joined[bucket_col] = pd.qcut(nz.rank(method="first"), q=q, labels=False, duplicates="drop")
                    joined[bucket_col] = joined[bucket_col].map(lambda v: f"Q{int(v) + 1}" if pd.notna(v) else np.nan)
                    diagnostics["team_context"].append(_summarize_factor(joined, "team_context", col, bucket_col, min_group_bets))
                else:
                    joined[col] = nz.astype(str)
                    diagnostics["team_context"].append(_summarize_factor(joined, "team_context", col, col, min_group_bets))
            else:
                diagnostics["team_context"].append(_summarize_factor(joined, "team_context", col, col, min_group_bets))

    out: Dict[str, pd.DataFrame] = {}
    for key, frames in diagnostics.items():
        non_empty = [f for f in frames if len(f) > 0]
        if not non_empty:
            out[key] = pd.DataFrame()
            continue
        combined = pd.concat(non_empty, ignore_index=True)
        combined = combined.sort_values(["model", "factor_name", "bets", "total_pnl"], ascending=[True, True, False, False])
        combined = (
            combined.groupby(["model", "factor_group", "factor_name"], as_index=False, group_keys=False)
            .head(int(top_factor_levels))
            .reset_index(drop=True)
        )
        out[key] = combined
    return out


def _plot_model_comparison(summary_df: pd.DataFrame, out_dir: Path) -> None:
    if len(summary_df) == 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_df = summary_df.sort_values("roi", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(plot_df["model"], plot_df["roi"], color="tab:green", alpha=0.85)
    axes[0].set_title("ROI by Model")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(plot_df["model"], plot_df.get("win_rate", 0), color="tab:purple", alpha=0.85)
    axes[1].set_title("Win Rate by Model")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_home_away_bets_wins_by_model(side_df: pd.DataFrame, out_dir: Path) -> None:
    if len(side_df) == 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(side_df))
    w = 0.2

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - 1.5 * w, side_df["home_bets"], width=w, label="HOME bets", color="tab:blue", alpha=0.85)
    ax.bar(x - 0.5 * w, side_df["away_bets"], width=w, label="AWAY bets", color="tab:orange", alpha=0.85)
    ax.bar(x + 0.5 * w, side_df["home_wins"], width=w, label="HOME wins", color="tab:green", alpha=0.85)
    ax.bar(x + 1.5 * w, side_df["away_wins"], width=w, label="AWAY wins", color="tab:red", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(side_df["model"], rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("HOME/AWAY Bets and Wins by Model")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / "home_away_bets_wins_by_model.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_bankrolls(sim_df: pd.DataFrame, suffixes: List[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if "GAME_DATE" not in sim_df.columns:
        return

    for suffix in suffixes:
        bcol = f"bankroll_{suffix}"
        dcol = f"drawdown_{suffix}"
        if bcol not in sim_df.columns or dcol not in sim_df.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sim_df["GAME_DATE"], sim_df[bcol], label="Bankroll")
        ax.set_title(f"Bankroll - {suffix}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Bankroll")
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(sim_df["GAME_DATE"], sim_df[dcol], color="tab:red", alpha=0.55, label="Drawdown")
        ax2.set_ylabel("Drawdown")

        fig.tight_layout()
        fig.savefig(out_dir / f"bankroll_{suffix}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)


def _plot_bet_outcomes(bet_log_df: pd.DataFrame, suffixes: List[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for suffix in suffixes:
        bcol = f"bet_side_{suffix}"
        pcol = f"pnl_kelly_{suffix}"
        if bcol not in bet_log_df.columns or pcol not in bet_log_df.columns:
            continue

        bets = bet_log_df[bet_log_df[bcol].fillna("NO BET") != "NO BET"].copy()
        if len(bets) == 0:
            continue

        home = bets.loc[bets[bcol] == "HOME", pcol].to_numpy(dtype=float)
        away = bets.loc[bets[bcol] == "AWAY", pcol].to_numpy(dtype=float)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        if len(home) > 0:
            axes[0].hist(home, bins=30, color="tab:blue", alpha=0.7, edgecolor="black")
            axes[0].set_title(f"{suffix} HOME (n={len(home)})")
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, "No HOME bets", transform=axes[0].transAxes, ha="center", va="center")

        if len(away) > 0:
            axes[1].hist(away, bins=30, color="tab:orange", alpha=0.7, edgecolor="black")
            axes[1].set_title(f"{suffix} AWAY (n={len(away)})")
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, "No AWAY bets", transform=axes[1].transAxes, ha="center", va="center")

        fig.tight_layout()
        fig.savefig(out_dir / f"bet_outcomes_{suffix}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)


def _resolve_tuning_models(all_suffixes: List[str], requested: Sequence[str]) -> List[str]:
    explicit = [m for m in requested if m in all_suffixes]
    if explicit:
        return explicit
    pref = [m for m in PREFERRED_TUNING_MODELS if m in all_suffixes]
    return pref if pref else all_suffixes


def _home_cover_series(df: pd.DataFrame) -> pd.Series:
    if "did_home_cover" in df.columns:
        return pd.to_numeric(df["did_home_cover"], errors="coerce")
    if "team_that_covered" in df.columns:
        return (df["team_that_covered"].astype(str) == "HOME").astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _calibrate_home_prob(prob_home: pd.Series, home_cover: pd.Series, bins: int = 10) -> pd.Series:
    p = pd.to_numeric(prob_home, errors="coerce").clip(1e-4, 1 - 1e-4)
    y = pd.to_numeric(home_cover, errors="coerce")
    valid = p.notna() & y.notna()
    if valid.sum() < 200:
        return p

    q = pd.qcut(p[valid], q=min(bins, int(valid.sum())), duplicates="drop")
    tmp = pd.DataFrame({"p": p[valid], "y": y[valid], "bin": q})
    agg = tmp.groupby("bin", observed=True).agg(mean_p=("p", "mean"), mean_y=("y", "mean"), n=("y", "size")).reset_index()

    global_rate = float(tmp["y"].mean())
    alpha = 25.0
    agg["calib"] = (agg["mean_y"] * agg["n"] + global_rate * alpha) / (agg["n"] + alpha)

    interval_index = pd.IntervalIndex(agg["bin"]) if len(agg) else None
    if interval_index is None:
        return p

    mapped = pd.cut(p, bins=interval_index, include_lowest=True)
    calib_map = {k: v for k, v in zip(agg["bin"], agg["calib"])}
    out = mapped.map(calib_map).astype(float).fillna(p)
    return out.clip(1e-4, 1 - 1e-4)


def _selected_side_ev(prob_home: pd.Series, bet_side: pd.Series, payout_home: pd.Series, payout_away: pd.Series) -> pd.Series:
    ev_home = prob_home * payout_home - (1 - prob_home)
    ev_away = (1 - prob_home) * payout_away - prob_home
    return pd.Series(np.where(bet_side == "HOME", ev_home, np.where(bet_side == "AWAY", ev_away, np.nan)), index=prob_home.index)


def _kelly_selected_side(prob_home: pd.Series, bet_side: pd.Series, payout_home: pd.Series, payout_away: pd.Series) -> pd.Series:
    ph = prob_home.to_numpy(dtype=float)
    bh = payout_home.to_numpy(dtype=float)
    ba = payout_away.to_numpy(dtype=float)

    home_mask = (bet_side == "HOME").to_numpy(dtype=bool)
    away_mask = (bet_side == "AWAY").to_numpy(dtype=bool)

    k_home = np.where(home_mask, ((ph * (bh + 1.0) - 1.0) / bh), 0.0)
    p_away = 1.0 - ph
    k_away = np.where(away_mask, ((p_away * (ba + 1.0) - 1.0) / ba), 0.0)

    return pd.Series(np.clip(k_home + k_away, 0.0, 1.0), index=prob_home.index)


def _objective_score(edge: pd.Series, ev_side: pd.Series, risk: pd.Series, kelly: pd.Series, objective_type: str, objective_lambda: float) -> pd.Series:
    edge_abs = pd.to_numeric(edge, errors="coerce").abs().fillna(0.0)
    ev = pd.to_numeric(ev_side, errors="coerce").fillna(0.0)
    risk_series = pd.to_numeric(risk, errors="coerce").abs()
    risk_med = float(risk_series.dropna().median()) if risk_series.notna().any() else 1.0
    risk_norm = (risk_series / (risk_med if risk_med > 0 else 1.0)).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.5, 3.0)

    signal = ev.copy()
    missing_signal = signal.isna() | (signal == 0)
    signal = signal.where(~missing_signal, edge_abs)

    if objective_type == "log_growth_dd":
        penalty = 1.0 + float(objective_lambda) * risk_norm + float(objective_lambda) * pd.to_numeric(kelly, errors="coerce").fillna(0.0)
        return signal / penalty
    return signal


def _simulate_variant_for_model(model_df: pd.DataFrame, suffix: str, spec: ExperimentSpec, starting_bankroll: float) -> Dict[str, object]:
    bet_col = f"bet_side_{suffix}"
    win_col = f"bet_win_{suffix}"
    kelly_col = f"kelly_frac_{suffix}"
    edge_col = f"edge_{suffix}"
    win_prob_col = f"win_prob_home_{suffix}"
    sigma_col = f"sigma_{suffix}"
    edge_std_col = f"edge_std_{suffix}"
    ev_home_col = f"ev_home_{suffix}"
    ev_away_col = f"ev_away_{suffix}"

    required = [bet_col, win_col, kelly_col]
    if any(c not in model_df.columns for c in required):
        return {"experiment_id": spec.experiment_id, "model": suffix, "status": "missing_columns"}

    work = model_df.copy()
    work["GAME_DATE"] = pd.to_datetime(work["GAME_DATE"]).dt.normalize()
    work = work.sort_values("GAME_DATE").reset_index(drop=True)

    payout_home = pd.to_numeric(work.get("payout_home", DEFAULT_PAYOUT), errors="coerce").fillna(DEFAULT_PAYOUT)
    payout_away = pd.to_numeric(work.get("payout_away", DEFAULT_PAYOUT), errors="coerce").fillna(DEFAULT_PAYOUT)
    payout_home = payout_home.where(payout_home > 0, DEFAULT_PAYOUT)
    payout_away = payout_away.where(payout_away > 0, DEFAULT_PAYOUT)

    if win_prob_col in work.columns:
        prob_home_raw = pd.to_numeric(work[win_prob_col], errors="coerce").clip(1e-4, 1 - 1e-4)
    else:
        prob_home_raw = pd.Series(np.nan, index=work.index, dtype=float)

    if prob_home_raw.isna().all():
        edge_std = pd.to_numeric(work.get(edge_std_col, 0.0), errors="coerce").fillna(0.0)
        prob_home_raw = pd.Series(1.0 / (1.0 + np.exp(-edge_std.to_numpy())), index=work.index)

    home_cover = _home_cover_series(work)
    prob_home = _calibrate_home_prob(prob_home_raw, home_cover) if spec.use_prob_calibration else prob_home_raw

    bet_side = work[bet_col].fillna("NO BET")
    kelly_base = _kelly_selected_side(prob_home, bet_side, payout_home, payout_away)
    kelly_adj = np.clip(kelly_base * spec.kelly_scale, 0.0, spec.kelly_cap)
    work["_kelly_adj"] = kelly_adj

    candidate = bet_side.isin(["HOME", "AWAY"]) & (work["_kelly_adj"] > 0)

    if spec.min_abs_edge > 0 and edge_col in work.columns:
        edge_abs = pd.to_numeric(work[edge_col], errors="coerce").abs().fillna(0.0)
        candidate &= edge_abs >= float(spec.min_abs_edge)

    if spec.exclude_default_payout and "used_default_payout" in work.columns:
        candidate &= ~work["used_default_payout"].fillna(False).astype(bool)

    if spec.allowed_market_tiers is not None and "market_match_quality" in work.columns:
        candidate &= work["market_match_quality"].isin(spec.allowed_market_tiers)

    ev_side = _selected_side_ev(prob_home, bet_side, payout_home, payout_away)
    if ev_home_col in work.columns and ev_away_col in work.columns:
        ev_side = pd.Series(
            np.where(bet_side == "HOME", work[ev_home_col], np.where(bet_side == "AWAY", work[ev_away_col], np.nan)),
            index=work.index,
        ).fillna(ev_side)

    risk = pd.to_numeric(work.get(edge_std_col, np.nan), errors="coerce").abs()
    if risk.isna().all() and sigma_col in work.columns:
        risk = pd.to_numeric(work[sigma_col], errors="coerce").abs()

    score = _objective_score(
        edge=pd.to_numeric(work.get(edge_col, 0.0), errors="coerce").fillna(0.0),
        ev_side=ev_side,
        risk=risk,
        kelly=work["_kelly_adj"],
        objective_type=spec.objective_type,
        objective_lambda=spec.objective_lambda,
    )

    work["_rank"] = np.nan
    work.loc[candidate, "_rank"] = (
        pd.Series(score, index=work.index)
        .loc[candidate]
        .groupby(work.loc[candidate, "GAME_DATE"])
        .rank(method="first", ascending=False)
    )

    selected = candidate & (work["_rank"] <= int(spec.top_n_bets_per_day))

    pre_exposure = pd.Series(np.where(selected, work["_kelly_adj"], 0.0), index=work.index).groupby(work["GAME_DATE"]).transform("sum")
    scale = np.where(pre_exposure > spec.daily_max_exposure, spec.daily_max_exposure / pre_exposure.replace(0, np.nan), 1.0)
    scale = pd.Series(scale, index=work.index).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    work["_kelly_final"] = np.clip(np.where(selected, work["_kelly_adj"] * scale, 0.0), 0.0, spec.kelly_cap)

    wins = work[win_col].fillna(0).astype(int) == 1
    home_mask = bet_side == "HOME"
    away_mask = bet_side == "AWAY"

    pnl = np.zeros(len(work), dtype=float)
    home_win = selected & home_mask & wins
    away_win = selected & away_mask & wins
    losses = selected & (~wins)

    pnl[home_win] = work.loc[home_win, "_kelly_final"].to_numpy() * payout_home.loc[home_win].to_numpy()
    pnl[away_win] = work.loc[away_win, "_kelly_final"].to_numpy() * payout_away.loc[away_win].to_numpy()
    pnl[losses] = -work.loc[losses, "_kelly_final"].to_numpy()

    if spec.max_daily_loss is not None:
        pnl_adj = pnl.copy()
        for _, idx in work.groupby("GAME_DATE").groups.items():
            day_idx = sorted(
                list(idx),
                key=lambda i: (
                    float(work.loc[i, "_rank"]) if pd.notna(work.loc[i, "_rank"]) else np.inf,
                    int(i),
                ),
            )
            day_cum = 0.0
            for i in day_idx:
                if day_cum <= spec.max_daily_loss:
                    pnl_adj[i] = 0.0
                day_cum += pnl_adj[i]
        pnl = pnl_adj

    growth = np.clip(1.0 + pnl, 1e-12, None)
    bankroll = pd.Series(growth).cumprod() * float(starting_bankroll)
    drawdown = bankroll / bankroll.cummax().replace(0, np.nan) - 1.0

    bet_count = int(selected.sum())
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
        "min_abs_edge": spec.min_abs_edge,
        "bets_placed": bet_count,
        "home_bets": int((selected & home_mask).sum()),
        "away_bets": int((selected & away_mask).sum()),
        "win_rate": float(work.loc[selected, win_col].mean()) if bet_count else np.nan,
        "total_pnl": float(np.sum(pnl)),
        "total_pnl_return_sum": float(np.sum(pnl)),
        "avg_pnl_per_bet": float(np.mean(pnl[selected])) if bet_count else np.nan,
        "avg_kelly": float(np.mean(work.loc[selected, "_kelly_final"])) if bet_count else 0.0,
        "ending_bankroll": float(bankroll.iloc[-1]) if len(bankroll) else float(starting_bankroll),
        "roi": float((bankroll.iloc[-1] - starting_bankroll) / starting_bankroll) if len(bankroll) else 0.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
    }


def run_experiment_matrix(
    model_df: pd.DataFrame,
    suffixes: List[str],
    starting_bankroll: float,
    selected_models: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    models = selected_models if selected_models else suffixes

    rows: List[Dict[str, object]] = []
    for spec in DEFAULT_EXPERIMENT_MATRIX:
        for suffix in models:
            rows.append(_simulate_variant_for_model(model_df, suffix, spec, starting_bankroll))

    exp_df = pd.DataFrame(rows)
    ok = exp_df[exp_df["status"] == "ok"].copy() if len(exp_df) else pd.DataFrame()
    if len(ok) == 0:
        return exp_df, pd.DataFrame()

    best = (
        ok.sort_values(["experiment_id", "roi"], ascending=[True, False])
        .groupby("experiment_id", as_index=False)
        .head(1)
        .rename(columns={"model": "best_model", "roi": "best_roi", "max_drawdown": "best_model_max_drawdown"})
    )

    agg = (
        ok.groupby("experiment_id", as_index=False)
        .agg(
            avg_roi=("roi", "mean"),
            median_roi=("roi", "median"),
            avg_max_drawdown=("max_drawdown", "mean"),
            total_bets=("bets_placed", "sum"),
            avg_win_rate=("win_rate", "mean"),
        )
    )

    summary = agg.merge(best[["experiment_id", "best_model", "best_roi", "best_model_max_drawdown"]], on="experiment_id", how="left")
    summary = summary.sort_values("avg_roi", ascending=False)
    return exp_df, summary


def _plot_experiment_matrix(exp_df: pd.DataFrame, out_dir: Path) -> None:
    if len(exp_df) == 0:
        return
    ok = exp_df[exp_df["status"] == "ok"].copy()
    if len(ok) == 0:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    best = ok.sort_values(["experiment_id", "roi"], ascending=[True, False]).groupby("experiment_id", as_index=False).head(1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(best["experiment_id"], best["roi"], color="tab:green", alpha=0.85)
    axes[0].set_title("Best Model ROI by Experiment")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(best["experiment_id"], best["max_drawdown"], color="tab:red", alpha=0.85)
    axes[1].set_title("Best Model Drawdown by Experiment")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "experiment_matrix_comparison.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def build_time_regime_summaries(
    model_df: pd.DataFrame,
    suffixes: List[str],
    vol_window: int,
    vol_min_periods: int,
    vol_q1: float,
    vol_q2: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly_rows: List[Dict[str, object]] = []
    season_rows: List[Dict[str, object]] = []
    vol_daily_rows: List[Dict[str, object]] = []
    vol_summary_rows: List[Dict[str, object]] = []

    df = model_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    if "season" not in df.columns:
        df["season"] = df["GAME_DATE"].dt.year
    df["month"] = df["GAME_DATE"].dt.to_period("M").astype(str)

    for suffix in suffixes:
        bcol = f"bet_side_{suffix}"
        pcol = f"pnl_kelly_{suffix}"
        wcol = f"bet_win_{suffix}"
        if bcol not in df.columns or pcol not in df.columns:
            continue

        bets = df[df[bcol].fillna("NO BET") != "NO BET"].copy()
        if len(bets) == 0:
            continue

        for m, g in bets.groupby("month"):
            monthly_rows.append(
                {
                    "model": suffix,
                    "month": m,
                    "bets": int(len(g)),
                    "total_pnl": float(g[pcol].sum()),
                    "avg_pnl_per_bet": float(g[pcol].mean()),
                    "win_rate": float(g[wcol].mean()) if wcol in g.columns else np.nan,
                }
            )

        for s, g in bets.groupby("season"):
            season_rows.append(
                {
                    "model": suffix,
                    "season": s,
                    "bets": int(len(g)),
                    "total_pnl": float(g[pcol].sum()),
                    "avg_pnl_per_bet": float(g[pcol].mean()),
                    "win_rate": float(g[wcol].mean()) if wcol in g.columns else np.nan,
                }
            )

        bets = bets.copy()
        bets["date"] = pd.to_datetime(bets["GAME_DATE"]).dt.normalize()

        if wcol in bets.columns:
            daily = (
                bets.groupby("date", as_index=False)
                .agg(
                    daily_pnl=(pcol, "sum"),
                    daily_bets=(pcol, "size"),
                    daily_win_rate=(wcol, "mean"),
                )
                .sort_values("date")
            )
        else:
            daily = (
                bets.groupby("date", as_index=False)
                .agg(
                    daily_pnl=(pcol, "sum"),
                    daily_bets=(pcol, "size"),
                )
                .sort_values("date")
            )
            daily["daily_win_rate"] = np.nan

        daily["vol_rolling_28d"] = daily["daily_pnl"].rolling(window=vol_window, min_periods=vol_min_periods).std()
        valid = daily["vol_rolling_28d"].dropna()

        if len(valid) >= 30:
            ql = float(valid.quantile(vol_q1))
            qh = float(valid.quantile(vol_q2))

            def _bucket(v: float) -> str:
                if pd.isna(v):
                    return "unknown_vol"
                if v <= ql:
                    return "low_vol"
                if v <= qh:
                    return "mid_vol"
                return "high_vol"

            daily["vol_bucket"] = daily["vol_rolling_28d"].map(_bucket)
        else:
            daily["vol_bucket"] = "unknown_vol"

        daily["model"] = suffix
        vol_daily_rows.extend(daily.to_dict(orient="records"))

        for bucket, g in daily.groupby("vol_bucket"):
            vol_summary_rows.append(
                {
                    "model": suffix,
                    "vol_bucket": bucket,
                    "days": int(len(g)),
                    "bets": int(g["daily_bets"].sum()),
                    "total_pnl": float(g["daily_pnl"].sum()),
                    "avg_daily_pnl": float(g["daily_pnl"].mean()),
                    "avg_pnl_per_bet": float(g["daily_pnl"].sum() / g["daily_bets"].sum()) if g["daily_bets"].sum() > 0 else np.nan,
                    "avg_daily_win_rate": float(g["daily_win_rate"].mean()) if "daily_win_rate" in g.columns else np.nan,
                }
            )

    return (
        pd.DataFrame(monthly_rows).sort_values(["model", "month"]).reset_index(drop=True) if len(monthly_rows) else pd.DataFrame(),
        pd.DataFrame(season_rows).sort_values(["model", "season"]).reset_index(drop=True) if len(season_rows) else pd.DataFrame(),
        pd.DataFrame(vol_daily_rows).sort_values(["model", "date"]).reset_index(drop=True) if len(vol_daily_rows) else pd.DataFrame(),
        pd.DataFrame(vol_summary_rows).sort_values(["model", "vol_bucket"]).reset_index(drop=True) if len(vol_summary_rows) else pd.DataFrame(),
    )


def build_regime_to_profile_switch_plan(
    vol_daily_df: pd.DataFrame,
    primary_model: str,
    high_vol_profile: str,
    default_profile: str,
    exp_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(vol_daily_df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    v = vol_daily_df.copy()
    v = v[v["model"] == primary_model].copy()
    if len(v) == 0:
        return pd.DataFrame(), pd.DataFrame()

    v["date"] = pd.to_datetime(v["date"], errors="coerce")
    v = v.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    v["recommended_experiment_id"] = np.where(v["vol_bucket"] == "high_vol", high_vol_profile, default_profile)
    v["switch_reason"] = np.where(
        v["vol_bucket"] == "high_vol",
        "high_vol -> growth profile",
        "non_high_vol -> risk profile",
    )

    keep_cols = [
        "date",
        "model",
        "vol_bucket",
        "daily_pnl",
        "daily_bets",
        "daily_win_rate",
        "vol_rolling_28d",
        "recommended_experiment_id",
        "switch_reason",
    ]
    daily_plan = v[keep_cols].copy()

    if exp_df is not None and len(exp_df) > 0:
        ok = exp_df[(exp_df["status"] == "ok") & (exp_df["model"] == primary_model)].copy()
        if len(ok) > 0:
            ok = ok[["experiment_id", "roi", "max_drawdown", "win_rate", "avg_pnl_per_bet"]].rename(
                columns={
                    "experiment_id": "recommended_experiment_id",
                    "roi": "profile_roi",
                    "max_drawdown": "profile_max_drawdown",
                    "win_rate": "profile_win_rate",
                    "avg_pnl_per_bet": "profile_avg_pnl_per_bet",
                }
            )
            daily_plan = daily_plan.merge(ok, on="recommended_experiment_id", how="left")

    summary = (
        daily_plan.groupby(["model", "vol_bucket", "recommended_experiment_id"], as_index=False)
        .agg(
            days=("date", "count"),
            total_bets=("daily_bets", "sum"),
            avg_daily_bets=("daily_bets", "mean"),
            avg_daily_win_rate=("daily_win_rate", "mean"),
            avg_daily_pnl=("daily_pnl", "mean"),
        )
        .sort_values(["model", "vol_bucket", "recommended_experiment_id"])
        .reset_index(drop=True)
    )

    return daily_plan, summary


def validate_time_aggregations(monthly_df: pd.DataFrame, season_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if len(monthly_df) == 0 or len(season_df) == 0:
        return pd.DataFrame(
            [
                {
                    "check": "monthly_vs_season",
                    "status": "skipped",
                    "details": "monthly or season summary missing",
                }
            ]
        )

    m = monthly_df.copy()
    s = season_df.copy()

    # Primary reconciliation: totals by model should match exactly across monthly vs season summaries.
    m_model = m.groupby("model", as_index=False).agg(monthly_bets=("bets", "sum"), monthly_total_pnl=("total_pnl", "sum"))
    s_model = s.groupby("model", as_index=False).agg(season_bets=("bets", "sum"), season_total_pnl=("total_pnl", "sum"))
    model_cmp = m_model.merge(s_model, on="model", how="outer")
    model_cmp["bets_diff"] = (model_cmp["monthly_bets"].fillna(0) - model_cmp["season_bets"].fillna(0)).abs()
    model_cmp["pnl_diff"] = (model_cmp["monthly_total_pnl"].fillna(0.0) - model_cmp["season_total_pnl"].fillna(0.0)).abs()

    bets_ok = bool((model_cmp["bets_diff"] <= 0).all())
    pnl_ok = bool((model_cmp["pnl_diff"] <= 1e-9).all())

    rows.append(
        {
            "check": "monthly_bets_reconcile",
            "status": "pass" if bets_ok else "warn",
            "details": f"max_bets_diff={float(model_cmp['bets_diff'].max()) if len(model_cmp) else 0.0}",
        }
    )
    rows.append(
        {
            "check": "monthly_pnl_reconcile",
            "status": "pass" if pnl_ok else "warn",
            "details": f"max_pnl_diff={float(model_cmp['pnl_diff'].max()) if len(model_cmp) else 0.0}",
        }
    )

    # Secondary informational check: season labels may be domain-specific (NBA season spans years).
    m["inferred_year"] = pd.to_datetime(m["month"] + "-01", errors="coerce").dt.year
    m = m.dropna(subset=["inferred_year"])
    m["inferred_year"] = m["inferred_year"].astype(int)
    s["season"] = pd.to_numeric(s["season"], errors="coerce").astype("Int64")
    s = s.dropna(subset=["season"])
    s["season"] = s["season"].astype(int)

    year_keys_m = set(m[["model", "inferred_year"]].drop_duplicates().itertuples(index=False, name=None))
    year_keys_s = set(s[["model", "season"]].drop_duplicates().itertuples(index=False, name=None))
    only_monthly = len(year_keys_m - year_keys_s)
    only_season = len(year_keys_s - year_keys_m)
    rows.append(
        {
            "check": "season_label_alignment_info",
            "status": "info",
            "details": f"monthly_only_keys={only_monthly}; season_only_keys={only_season}",
        }
    )

    return pd.DataFrame(rows)


def _determine_run_folder(outputs_dir: Path, selected_inputs: Dict[str, Path], run_date: Optional[str]) -> Path:
    if run_date:
        return outputs_dir / f"copilot_analysis_odds_v{run_date}"
    parsed = _parse_version_date(selected_inputs["model_output"])
    if parsed is not None:
        return outputs_dir / f"copilot_analysis_odds_v{parsed.strftime('%Y_%m_%d')}"
    return outputs_dir / "copilot_analysis_latest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhanced analysis of copilot model outputs.")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--starting-bankroll", type=float, default=10000.0)
    parser.add_argument("--run-date", default=None)

    parser.add_argument("--prefer-odds", dest="prefer_odds", action="store_true", default=True)
    parser.add_argument("--no-prefer-odds", dest="prefer_odds", action="store_false")
    parser.add_argument("--allow-fallback", dest="allow_fallback", action="store_true", default=True)
    parser.add_argument("--no-allow-fallback", dest="allow_fallback", action="store_false")

    parser.add_argument("--strict-schema", dest="strict_schema", action="store_true", default=True)
    parser.add_argument("--no-strict-schema", dest="strict_schema", action="store_false")

    parser.add_argument("--run-experiment-matrix", dest="run_experiment_matrix", action="store_true", default=True)
    parser.add_argument("--no-run-experiment-matrix", dest="run_experiment_matrix", action="store_false")

    parser.add_argument("--tuning-models", default="linear_regression,elasticnet")
    parser.add_argument("--primary-model-for-switch", default="linear_regression")
    parser.add_argument("--high-vol-profile", default="R6_stability_tier_b_c")
    parser.add_argument("--default-profile", default="R5_stability_tier_b")

    parser.add_argument("--vol-window", type=int, default=28)
    parser.add_argument("--vol-min-periods", type=int, default=14)
    parser.add_argument("--vol-quantiles", default="0.33,0.67")

    parser.add_argument("--enable-feature-diagnostics", dest="enable_feature_diagnostics", action="store_true", default=True)
    parser.add_argument("--no-enable-feature-diagnostics", dest="enable_feature_diagnostics", action="store_false")
    parser.add_argument("--feature-source", default=DEFAULT_FEATURE_SOURCE)
    parser.add_argument("--feature-join-key", default="GAME_ID")
    parser.add_argument("--min-group-bets", type=int, default=25)
    parser.add_argument("--feature-quantiles", type=int, default=5)
    parser.add_argument("--top-factor-levels", type=int, default=10)

    parser.add_argument("--input-model-output", default=None)
    parser.add_argument("--input-model-rmse", default=None)
    parser.add_argument("--input-join-audit", default=None)
    parser.add_argument("--input-objective-sweep", default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)

    selected_inputs, manifest_df = discover_inputs(outputs_dir, prefer_odds=args.prefer_odds, allow_fallback=args.allow_fallback)

    overrides = {
        "model_output": args.input_model_output,
        "model_rmse": args.input_model_rmse,
        "join_audit": args.input_join_audit,
        "objective_sweep": args.input_objective_sweep,
    }
    _validate_override_paths(overrides)
    for k, v in overrides.items():
        if v:
            selected_inputs[k] = Path(v)
            manifest_df.loc[manifest_df["input_key"] == k, "selected_path"] = str(v)
            manifest_df.loc[manifest_df["input_key"] == k, "selection_source"] = "manual_override"

    run_folder = _determine_run_folder(outputs_dir, selected_inputs, args.run_date)
    run_folder.mkdir(parents=True, exist_ok=True)

    model_df = pd.read_csv(selected_inputs["model_output"])
    schema_df = validate_schema(model_df, strict_schema=args.strict_schema)

    feature_join_df = model_df.copy()
    feature_join_quality_df = pd.DataFrame()
    feature_manifest_df = pd.DataFrame()
    if args.enable_feature_diagnostics:
        feature_join_df, feature_join_quality_df, feature_manifest_df = _build_feature_join(
            model_df=model_df,
            feature_source=Path(args.feature_source),
            feature_join_key=str(args.feature_join_key),
        )

    sim_df, summary_df, bet_log_df, suffixes = build_bankroll_outputs(model_df, starting_bankroll=args.starting_bankroll)
    outcome_df = build_bet_outcome_summary(bet_log_df, suffixes)
    side_df = build_side_counts_summary(outcome_df)
    profitable_df = _rank_profitable_models(summary_df)

    bets_long_df = _build_model_bets_long(model_df=feature_join_df, suffixes=suffixes)
    deep_diag = _deep_diagnostics(
        bets_long_df=bets_long_df,
        model_df_for_features=feature_join_df,
        min_group_bets=args.min_group_bets,
        feature_quantiles=args.feature_quantiles,
        top_factor_levels=args.top_factor_levels,
    )

    market_default_df, market_tier_df, market_2d_df = build_market_quality_diagnostics(model_df, suffixes)

    vol_q1, vol_q2 = _parse_vol_quantiles(args.vol_quantiles)

    monthly_df, season_df, vol_daily_df, vol_summary_df = build_time_regime_summaries(
        model_df=model_df,
        suffixes=suffixes,
        vol_window=args.vol_window,
        vol_min_periods=args.vol_min_periods,
        vol_q1=vol_q1,
        vol_q2=vol_q2,
    )
    time_validation_df = validate_time_aggregations(monthly_df, season_df)

    requested_models = [s.strip() for s in str(args.tuning_models).split(",") if s.strip()]
    tuning_models = _resolve_tuning_models(suffixes, requested_models)
    shortlist_df = pd.DataFrame({"selected_tuning_models": tuning_models})

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
    side_df.to_csv(run_folder / "side_bets_wins_by_model.csv", index=False)
    profitable_df.to_csv(run_folder / "profitable_models_summary.csv", index=False)

    market_default_df.to_csv(run_folder / "market_default_split_summary.csv", index=False)
    market_tier_df.to_csv(run_folder / "market_tier_split_summary.csv", index=False)
    market_2d_df.to_csv(run_folder / "market_quality_2d_split_summary.csv", index=False)

    deep_diag.get("bet_type", pd.DataFrame()).to_csv(run_folder / "bet_type_factor_summary.csv", index=False)
    deep_diag.get("time", pd.DataFrame()).to_csv(run_folder / "time_factor_summary.csv", index=False)
    deep_diag.get("confidence", pd.DataFrame()).to_csv(run_folder / "confidence_factor_summary.csv", index=False)
    deep_diag.get("data_quality", pd.DataFrame()).to_csv(run_folder / "data_quality_factor_summary.csv", index=False)
    deep_diag.get("team_context", pd.DataFrame()).to_csv(run_folder / "team_context_factor_summary.csv", index=False)

    feature_join_quality_df.to_csv(run_folder / "feature_join_quality_report.csv", index=False)
    feature_manifest_df.to_csv(run_folder / "feature_diagnostics_manifest.csv", index=False)

    monthly_df.to_csv(run_folder / "monthly_summary.csv", index=False)
    season_df.to_csv(run_folder / "season_summary.csv", index=False)
    vol_daily_df.to_csv(run_folder / "volatility_regime_daily.csv", index=False)
    vol_summary_df.to_csv(run_folder / "volatility_regime_summary.csv", index=False)
    time_validation_df.to_csv(run_folder / "time_aggregation_validation.csv", index=False)

    shortlist_df.to_csv(run_folder / "model_shortlist_summary.csv", index=False)
    manifest_df.to_csv(run_folder / "input_selection_manifest.csv", index=False)
    schema_df.to_csv(run_folder / "schema_validation_report.csv", index=False)

    _plot_bankrolls(sim_df, suffixes, run_folder / "bet_plots")
    _plot_bet_outcomes(bet_log_df, suffixes, run_folder / "bet_outcome_histograms")
    _plot_model_comparison(summary_df, run_folder)
    _plot_home_away_bets_wins_by_model(side_df, run_folder)

    if args.run_experiment_matrix:
        exp_df, exp_summary_df = run_experiment_matrix(
            model_df=model_df,
            suffixes=suffixes,
            starting_bankroll=args.starting_bankroll,
            selected_models=tuning_models,
        )
        exp_df.to_csv(run_folder / "experiment_matrix_results.csv", index=False)
        exp_summary_df.to_csv(run_folder / "experiment_matrix_summary.csv", index=False)
        _plot_experiment_matrix(exp_df, run_folder)

        if len(exp_df) > 0:
            prod = exp_df[(exp_df["experiment_id"].isin(PRODUCTION_PROFILE_EXPERIMENT_IDS)) & (exp_df["status"] == "ok")].copy()
            prod.to_csv(run_folder / "production_profile_results_combined.csv", index=False)

            for exp_id in PRODUCTION_PROFILE_EXPERIMENT_IDS:
                per_profile = prod[prod["experiment_id"] == exp_id].copy()
                per_profile.to_csv(run_folder / f"production_profile_results_{exp_id}.csv", index=False)

            # Backward-compatible alias
            prod_r5 = prod[prod["experiment_id"] == PRODUCTION_PROFILE_EXPERIMENT_IDS[0]].copy()
            prod_r5.to_csv(run_folder / "production_profile_results.csv", index=False)

            regime_plan_df, regime_plan_summary_df = build_regime_to_profile_switch_plan(
                vol_daily_df=vol_daily_df,
                primary_model=args.primary_model_for_switch,
                high_vol_profile=args.high_vol_profile,
                default_profile=args.default_profile,
                exp_df=exp_df,
            )
            regime_plan_df.to_csv(run_folder / "regime_to_profile_daily.csv", index=False)
            regime_plan_summary_df.to_csv(run_folder / "regime_to_profile_summary.csv", index=False)

    print(f"Enhanced analysis complete. Artifacts written to: {run_folder}")
    print(f"Models analyzed: {', '.join(suffixes)}")
    print(f"Tuning shortlist used: {', '.join(tuning_models)}")
    if len(profitable_df) > 0:
        top_model = profitable_df.iloc[0]
        print(
            "Top profitable model: "
            f"{top_model['model']} | bankroll_growth={float(top_model['bankroll_growth']):.4f} | "
            f"ending_bankroll={float(top_model['ending_bankroll']):.2f}"
        )


if __name__ == "__main__":
    main()
