"""Create analytics, detailed findings, and recommended next steps from a copilot analysis output folder."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path("outputs/copilot_analysis_odds_v2026_02_19")
DEFAULT_OUTPUT_ROOT = Path("outputs")
OUTPUT_PREFIX = "copilot_feedback_v"


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _to_float(value: object) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _to_bool(value: object) -> Optional[bool]:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "y"}:
        return True
    if raw in {"0", "false", "no", "n"}:
        return False
    return None


def _format_pct(value: Optional[float], decimals: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.{decimals}f}%"


def _format_float(value: Optional[float], decimals: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}"


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        resolved = lower_map.get(name.lower())
        if resolved is not None:
            return resolved
    return None


def _pick_best_model(copilot_summary: Optional[pd.DataFrame], profitable_summary: Optional[pd.DataFrame]) -> Optional[str]:
    if profitable_summary is not None and not profitable_summary.empty:
        if "profit_rank" in profitable_summary.columns:
            ranked = profitable_summary.sort_values("profit_rank", ascending=True)
            if "model" in ranked.columns:
                return str(ranked.iloc[0]["model"])
        if "roi" in profitable_summary.columns and "model" in profitable_summary.columns:
            ranked = profitable_summary.sort_values("roi", ascending=False)
            return str(ranked.iloc[0]["model"])

    if copilot_summary is not None and not copilot_summary.empty:
        model_col = _find_col(copilot_summary, ["model"])
        roi_col = _find_col(copilot_summary, ["roi"])
        if model_col and roi_col:
            ranked = copilot_summary.sort_values(roi_col, ascending=False)
            return str(ranked.iloc[0][model_col])
    return None


def load_inputs(input_dir: Path) -> Tuple[Dict[str, Optional[pd.DataFrame]], List[str], List[str]]:
    files = {
        "copilot_betting_summary": "copilot_betting_summary.csv",
        "profitable_models_summary": "profitable_models_summary.csv",
        "bet_outcomes_summary": "bet_outcomes_summary.csv",
        "market_tier_split_summary": "market_tier_split_summary.csv",
        "monthly_summary": "monthly_summary.csv",
        "volatility_regime_summary": "volatility_regime_summary.csv",
        "regime_to_profile_summary": "regime_to_profile_summary.csv",
        "confidence_factor_summary": "confidence_factor_summary.csv",
        "data_quality_factor_summary": "data_quality_factor_summary.csv",
        "bet_type_factor_summary": "bet_type_factor_summary.csv",
        "team_context_factor_summary": "team_context_factor_summary.csv",
        "schema_validation_report": "schema_validation_report.csv",
        "feature_join_quality_report": "feature_join_quality_report.csv",
        "model_rmse_selected": "model_rmse_selected.csv",
        "feature_diagnostics_manifest": "feature_diagnostics_manifest.csv",
    }

    loaded: Dict[str, Optional[pd.DataFrame]] = {}
    present: List[str] = []
    missing: List[str] = []

    for key, name in files.items():
        full_path = input_dir / name
        df = _safe_read_csv(full_path)
        loaded[key] = df
        if df is None:
            missing.append(name)
        else:
            present.append(name)
    return loaded, present, missing


def build_analytics_summary(inputs: Dict[str, Optional[pd.DataFrame]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    copilot_summary = inputs.get("copilot_betting_summary")
    profitable_summary = inputs.get("profitable_models_summary")
    outcomes_summary = inputs.get("bet_outcomes_summary")
    market_summary = inputs.get("market_tier_split_summary")
    monthly_summary = inputs.get("monthly_summary")
    vol_summary = inputs.get("volatility_regime_summary")
    schema_summary = inputs.get("schema_validation_report")
    join_quality = inputs.get("feature_join_quality_report")
    feature_manifest = inputs.get("feature_diagnostics_manifest")

    best_model = _pick_best_model(copilot_summary, profitable_summary)
    if best_model:
        rows.append(
            {
                "category": "portfolio",
                "metric": "best_model",
                "value": best_model,
                "details": "top-ranked by available profitability/roi ordering",
            }
        )

    if copilot_summary is not None and not copilot_summary.empty:
        roi_col = _find_col(copilot_summary, ["roi"])
        num_bets_col = _find_col(copilot_summary, ["num_bets", "bets"])
        win_col = _find_col(copilot_summary, ["win_rate"])
        if roi_col:
            rows.append(
                {
                    "category": "portfolio",
                    "metric": "median_model_roi",
                    "value": float(pd.to_numeric(copilot_summary[roi_col], errors="coerce").median()),
                    "details": "median across models in copilot_betting_summary",
                }
            )
        if num_bets_col:
            rows.append(
                {
                    "category": "portfolio",
                    "metric": "total_model_bets",
                    "value": int(pd.to_numeric(copilot_summary[num_bets_col], errors="coerce").sum()),
                    "details": "sum of model-level bet counts",
                }
            )
        if win_col:
            rows.append(
                {
                    "category": "portfolio",
                    "metric": "avg_model_win_rate",
                    "value": float(pd.to_numeric(copilot_summary[win_col], errors="coerce").mean()),
                    "details": "mean of model-level win rates",
                }
            )

    if outcomes_summary is not None and not outcomes_summary.empty:
        pnl_col = _find_col(outcomes_summary, ["total_pnl"])
        model_col = _find_col(outcomes_summary, ["model"])
        if pnl_col:
            pnl_series = pd.to_numeric(outcomes_summary[pnl_col], errors="coerce")
            rows.append(
                {
                    "category": "outcomes",
                    "metric": "total_pnl_all_models",
                    "value": float(pnl_series.sum()),
                    "details": "sum across model rows",
                }
            )
            rows.append(
                {
                    "category": "outcomes",
                    "metric": "positive_pnl_models",
                    "value": int((pnl_series > 0).sum()),
                    "details": "count of model rows with positive total_pnl",
                }
            )
        if model_col:
            rows.append(
                {
                    "category": "outcomes",
                    "metric": "model_count",
                    "value": int(outcomes_summary[model_col].nunique()),
                    "details": "distinct models in bet_outcomes_summary",
                }
            )

    if market_summary is not None and not market_summary.empty:
        avg_col = _find_col(market_summary, ["avg_pnl_per_bet", "roi_per_bet"])
        tier_col = _find_col(market_summary, ["market_match_quality", "market_tier", "tier"])
        if avg_col and tier_col:
            market_tmp = market_summary.copy()
            market_tmp[avg_col] = pd.to_numeric(market_tmp[avg_col], errors="coerce")
            best_tier = market_tmp.sort_values(avg_col, ascending=False).iloc[0]
            worst_tier = market_tmp.sort_values(avg_col, ascending=True).iloc[0]
            rows.append(
                {
                    "category": "market",
                    "metric": "best_market_bucket",
                    "value": best_tier[tier_col],
                    "details": f"avg_pnl_per_bet={best_tier[avg_col]:.6f}",
                }
            )
            rows.append(
                {
                    "category": "market",
                    "metric": "worst_market_bucket",
                    "value": worst_tier[tier_col],
                    "details": f"avg_pnl_per_bet={worst_tier[avg_col]:.6f}",
                }
            )

    if monthly_summary is not None and not monthly_summary.empty:
        pnl_col = _find_col(monthly_summary, ["total_pnl"])
        month_col = _find_col(monthly_summary, ["month"])
        if pnl_col and month_col:
            grouped = monthly_summary.copy()
            grouped[pnl_col] = pd.to_numeric(grouped[pnl_col], errors="coerce")
            monthly_agg = grouped.groupby(month_col, dropna=False)[pnl_col].sum().reset_index()
            rows.append(
                {
                    "category": "time",
                    "metric": "positive_months",
                    "value": int((monthly_agg[pnl_col] > 0).sum()),
                    "details": "months with positive aggregate pnl",
                }
            )
            rows.append(
                {
                    "category": "time",
                    "metric": "negative_months",
                    "value": int((monthly_agg[pnl_col] < 0).sum()),
                    "details": "months with negative aggregate pnl",
                }
            )

    if vol_summary is not None and not vol_summary.empty:
        pnl_col = _find_col(vol_summary, ["avg_pnl_per_bet", "total_pnl"])
        bucket_col = _find_col(vol_summary, ["vol_bucket"])
        if pnl_col and bucket_col:
            tmp = vol_summary.copy()
            tmp[pnl_col] = pd.to_numeric(tmp[pnl_col], errors="coerce")
            agg = tmp.groupby(bucket_col, dropna=False)[pnl_col].mean().reset_index()
            best = agg.sort_values(pnl_col, ascending=False).iloc[0]
            worst = agg.sort_values(pnl_col, ascending=True).iloc[0]
            rows.append(
                {
                    "category": "regime",
                    "metric": "best_vol_bucket",
                    "value": best[bucket_col],
                    "details": f"mean_{pnl_col}={best[pnl_col]:.6f}",
                }
            )
            rows.append(
                {
                    "category": "regime",
                    "metric": "worst_vol_bucket",
                    "value": worst[bucket_col],
                    "details": f"mean_{pnl_col}={worst[pnl_col]:.6f}",
                }
            )

    if schema_summary is not None and not schema_summary.empty:
        severity_col = _find_col(schema_summary, ["severity"])
        present_col = _find_col(schema_summary, ["present"])
        if severity_col and present_col:
            tmp = schema_summary.copy()
            is_error = tmp[severity_col].astype(str).str.lower() == "error"
            present_flags = tmp[present_col].map(_to_bool)
            missing_hard = tmp[is_error & (present_flags == False)]
            rows.append(
                {
                    "category": "quality",
                    "metric": "missing_hard_required_columns",
                    "value": int(len(missing_hard)),
                    "details": "from schema_validation_report",
                }
            )

    if join_quality is not None and not join_quality.empty:
        match_col = _find_col(join_quality, ["feature_match_rate"])
        if match_col:
            rows.append(
                {
                    "category": "quality",
                    "metric": "feature_match_rate",
                    "value": float(pd.to_numeric(join_quality[match_col], errors="coerce").iloc[0]),
                    "details": "from feature_join_quality_report",
                }
            )

    if feature_manifest is not None and not feature_manifest.empty:
        include_col = _find_col(feature_manifest, ["include_for_factor_slicing"])
        leaky_col = _find_col(feature_manifest, ["is_leaky"])
        if include_col:
            include_count = int(feature_manifest[include_col].map(_to_bool).fillna(False).sum())
            rows.append(
                {
                    "category": "features",
                    "metric": "factor_slice_features",
                    "value": include_count,
                    "details": "count of columns included for factor slicing",
                }
            )
        if leaky_col:
            leaky_count = int(feature_manifest[leaky_col].map(_to_bool).fillna(False).sum())
            rows.append(
                {
                    "category": "features",
                    "metric": "flagged_leaky_features",
                    "value": leaky_count,
                    "details": "count of features flagged as leaky",
                }
            )

    return pd.DataFrame(rows)


def build_feedback_findings(inputs: Dict[str, Optional[pd.DataFrame]], best_model: Optional[str]) -> pd.DataFrame:
    findings: List[Dict[str, object]] = []

    copilot_summary = inputs.get("copilot_betting_summary")
    market_summary = inputs.get("market_tier_split_summary")
    monthly_summary = inputs.get("monthly_summary")
    vol_summary = inputs.get("volatility_regime_summary")
    confidence_summary = inputs.get("confidence_factor_summary")
    schema_summary = inputs.get("schema_validation_report")
    join_quality = inputs.get("feature_join_quality_report")

    if copilot_summary is not None and not copilot_summary.empty:
        model_col = _find_col(copilot_summary, ["model"])
        roi_col = _find_col(copilot_summary, ["roi"])
        win_col = _find_col(copilot_summary, ["win_rate"])
        if roi_col and model_col:
            tmp = copilot_summary.copy()
            tmp[roi_col] = pd.to_numeric(tmp[roi_col], errors="coerce")
            best_row = tmp.sort_values(roi_col, ascending=False).iloc[0]
            worst_row = tmp.sort_values(roi_col, ascending=True).iloc[0]
            findings.append(
                {
                    "severity": "positive",
                    "area": "model_ranking",
                    "finding": "Top model identified",
                    "metric": "roi",
                    "value": best_row[roi_col],
                    "threshold": "max",
                    "recommendation": f"Use {best_row[model_col]} as baseline model for constrained retuning.",
                    "source_file": "copilot_betting_summary.csv",
                }
            )
            findings.append(
                {
                    "severity": "high",
                    "area": "portfolio_health",
                    "finding": "Worst model underperformance",
                    "metric": "roi",
                    "value": worst_row[roi_col],
                    "threshold": "roi < -0.20",
                    "recommendation": f"De-prioritize or disable model {worst_row[model_col]} until recalibrated.",
                    "source_file": "copilot_betting_summary.csv",
                }
            )

        if win_col:
            avg_win = pd.to_numeric(copilot_summary[win_col], errors="coerce").mean()
            severity = "medium" if avg_win < 0.50 else "positive"
            findings.append(
                {
                    "severity": severity,
                    "area": "portfolio_health",
                    "finding": "Portfolio average win rate",
                    "metric": "win_rate",
                    "value": avg_win,
                    "threshold": "target >= 0.50",
                    "recommendation": "Tighten edge thresholds and calibration if win rate remains below 50%.",
                    "source_file": "copilot_betting_summary.csv",
                }
            )

    if market_summary is not None and not market_summary.empty:
        avg_col = _find_col(market_summary, ["avg_pnl_per_bet", "roi_per_bet"])
        tier_col = _find_col(market_summary, ["market_match_quality", "market_tier", "tier"])
        bets_col = _find_col(market_summary, ["bets", "num_bets"])
        if avg_col and tier_col:
            tmp = market_summary.copy()
            tmp[avg_col] = pd.to_numeric(tmp[avg_col], errors="coerce")
            if bets_col:
                tmp[bets_col] = pd.to_numeric(tmp[bets_col], errors="coerce")
                tmp["_safe_weight"] = np.maximum(tmp[bets_col], 1.0)
                tmp["_weighted_value"] = tmp[avg_col] * tmp["_safe_weight"]
                grouped = (
                    tmp.groupby(tier_col, dropna=False)[["_weighted_value", "_safe_weight"]]
                    .sum()
                    .reset_index()
                )
                grouped["weighted_avg"] = grouped["_weighted_value"] / grouped["_safe_weight"]
                weighted_df = grouped[[tier_col, "weighted_avg"]]
                best = weighted_df.sort_values("weighted_avg", ascending=False).iloc[0]
                worst = weighted_df.sort_values("weighted_avg", ascending=True).iloc[0]
                best_value = float(best["weighted_avg"])
                worst_value = float(worst["weighted_avg"])
                best_bucket = best[tier_col]
                worst_bucket = worst[tier_col]
            else:
                agg = tmp.groupby(tier_col, dropna=False)[avg_col].mean().reset_index(name="avg_metric")
                best = agg.sort_values("avg_metric", ascending=False).iloc[0]
                worst = agg.sort_values("avg_metric", ascending=True).iloc[0]
                best_value = float(best["avg_metric"])
                worst_value = float(worst["avg_metric"])
                best_bucket = best[tier_col]
                worst_bucket = worst[tier_col]

            findings.append(
                {
                    "severity": "positive" if best_value > 0 else "medium",
                    "area": "market_segmentation",
                    "finding": "Best market bucket",
                    "metric": "avg_pnl_per_bet",
                    "value": best_value,
                    "threshold": "> 0",
                    "recommendation": f"Increase selectivity toward {best_bucket} where feasible.",
                    "source_file": "market_tier_split_summary.csv",
                }
            )
            findings.append(
                {
                    "severity": "high" if worst_value < 0 else "medium",
                    "area": "market_segmentation",
                    "finding": "Worst market bucket",
                    "metric": "avg_pnl_per_bet",
                    "value": worst_value,
                    "threshold": ">= 0",
                    "recommendation": f"Reduce exposure in {worst_bucket} until feature quality improves.",
                    "source_file": "market_tier_split_summary.csv",
                }
            )

    if monthly_summary is not None and not monthly_summary.empty:
        pnl_col = _find_col(monthly_summary, ["total_pnl"])
        month_col = _find_col(monthly_summary, ["month"])
        if pnl_col and month_col:
            tmp = monthly_summary.copy()
            tmp[pnl_col] = pd.to_numeric(tmp[pnl_col], errors="coerce")
            monthly_agg = tmp.groupby(month_col, dropna=False)[pnl_col].sum().reset_index()
            negative_share = float((monthly_agg[pnl_col] < 0).mean()) if len(monthly_agg) else np.nan
            findings.append(
                {
                    "severity": "high" if negative_share > 0.60 else "medium",
                    "area": "time_stability",
                    "finding": "Negative month frequency",
                    "metric": "negative_month_share",
                    "value": negative_share,
                    "threshold": "<= 0.50",
                    "recommendation": "Add regime-aware stake throttling during drawdown months.",
                    "source_file": "monthly_summary.csv",
                }
            )

    if vol_summary is not None and not vol_summary.empty:
        bucket_col = _find_col(vol_summary, ["vol_bucket"])
        pnl_col = _find_col(vol_summary, ["avg_pnl_per_bet", "avg_daily_pnl"])
        if bucket_col and pnl_col:
            tmp = vol_summary.copy()
            tmp[pnl_col] = pd.to_numeric(tmp[pnl_col], errors="coerce")
            agg = tmp.groupby(bucket_col, dropna=False)[pnl_col].mean().reset_index()
            high_row = agg[agg[bucket_col].astype(str).str.lower() == "high_vol"]
            low_row = agg[agg[bucket_col].astype(str).str.lower() == "low_vol"]
            if not high_row.empty and not low_row.empty:
                high_val = float(high_row.iloc[0][pnl_col])
                low_val = float(low_row.iloc[0][pnl_col])
                findings.append(
                    {
                        "severity": "positive" if high_val > low_val else "medium",
                        "area": "regime_behavior",
                        "finding": "High-vol vs low-vol differential",
                        "metric": pnl_col,
                        "value": high_val - low_val,
                        "threshold": "> 0",
                        "recommendation": "Keep separate profile routing by volatility and retune low-vol profile.",
                        "source_file": "volatility_regime_summary.csv",
                    }
                )

    if confidence_summary is not None and not confidence_summary.empty:
        model_col = _find_col(confidence_summary, ["model"])
        group_col = _find_col(confidence_summary, ["factor_group"])
        name_col = _find_col(confidence_summary, ["factor_name"])
        level_col = _find_col(confidence_summary, ["factor_level"])
        roi_col = _find_col(confidence_summary, ["roi_per_bet", "avg_pnl_per_bet"])
        if model_col and group_col and name_col and level_col and roi_col:
            subset = confidence_summary.copy()
            if best_model:
                subset = subset[subset[model_col].astype(str) == str(best_model)]
            subset = subset[
                (subset[group_col].astype(str).str.lower() == "confidence")
                & (subset[name_col].astype(str).str.lower() == "edge_decile")
            ]
            if not subset.empty:
                subset[roi_col] = pd.to_numeric(subset[roi_col], errors="coerce")
                top = subset.sort_values(roi_col, ascending=False).iloc[0]
                bot = subset.sort_values(roi_col, ascending=True).iloc[0]
                findings.append(
                    {
                        "severity": "positive" if float(top[roi_col]) > 0 else "medium",
                        "area": "calibration",
                        "finding": "Best edge decile performance",
                        "metric": "roi_per_bet",
                        "value": float(top[roi_col]),
                        "threshold": "> 0",
                        "recommendation": f"Favor edge bucket {top[level_col]} for {best_model or 'top model'}.",
                        "source_file": "confidence_factor_summary.csv",
                    }
                )
                findings.append(
                    {
                        "severity": "high" if float(bot[roi_col]) < 0 else "medium",
                        "area": "calibration",
                        "finding": "Worst edge decile performance",
                        "metric": "roi_per_bet",
                        "value": float(bot[roi_col]),
                        "threshold": ">= 0",
                        "recommendation": f"Downweight or filter edge bucket {bot[level_col]} for {best_model or 'top model'}.",
                        "source_file": "confidence_factor_summary.csv",
                    }
                )

    if schema_summary is not None and not schema_summary.empty:
        severity_col = _find_col(schema_summary, ["severity"])
        present_col = _find_col(schema_summary, ["present"])
        if severity_col and present_col:
            tmp = schema_summary.copy()
            is_error = tmp[severity_col].astype(str).str.lower() == "error"
            present_flags = tmp[present_col].map(_to_bool)
            hard_missing = tmp[is_error & (present_flags == False)]
            findings.append(
                {
                    "severity": "high" if len(hard_missing) > 0 else "positive",
                    "area": "data_quality",
                    "finding": "Hard-required schema coverage",
                    "metric": "missing_hard_required_columns",
                    "value": int(len(hard_missing)),
                    "threshold": "= 0",
                    "recommendation": "Block deployment if any hard-required column is missing.",
                    "source_file": "schema_validation_report.csv",
                }
            )

    if join_quality is not None and not join_quality.empty:
        rate_col = _find_col(join_quality, ["feature_match_rate"])
        if rate_col:
            rate = _to_float(pd.to_numeric(join_quality[rate_col], errors="coerce").iloc[0])
            findings.append(
                {
                    "severity": "positive" if (rate is not None and rate >= 0.99) else "medium",
                    "area": "data_quality",
                    "finding": "Feature join match rate",
                    "metric": "feature_match_rate",
                    "value": rate,
                    "threshold": ">= 0.99",
                    "recommendation": "Monitor join coverage drift in daily runs.",
                    "source_file": "feature_join_quality_report.csv",
                }
            )

    findings_df = pd.DataFrame(findings)
    if findings_df.empty:
        return findings_df
    severity_rank = {"high": 0, "medium": 1, "positive": 2}
    findings_df["_rank"] = findings_df["severity"].map(severity_rank).fillna(9)
    findings_df = findings_df.sort_values(["_rank", "area", "finding"]).drop(columns=["_rank"])
    return findings_df


def build_model_statistics(inputs: Dict[str, Optional[pd.DataFrame]]) -> pd.DataFrame:
    base = inputs.get("copilot_betting_summary")
    if base is None or base.empty:
        return pd.DataFrame()

    model_col = _find_col(base, ["model"])
    if not model_col:
        return pd.DataFrame()

    keep_cols = [
        model_col,
        _find_col(base, ["roi"]),
        _find_col(base, ["win_rate"]),
        _find_col(base, ["num_bets", "bets"]),
        _find_col(base, ["avg_edge"]),
        _find_col(base, ["avg_ev_selected_side"]),
        _find_col(base, ["total_pnl"]),
        _find_col(base, ["ending_bankroll"]),
        _find_col(base, ["max_drawdown"]),
    ]
    keep_cols = [c for c in keep_cols if c is not None]
    stats = base[keep_cols].copy().rename(columns={model_col: "model"})

    for col in ["roi", "win_rate", "num_bets", "avg_edge", "avg_ev_selected_side", "total_pnl", "ending_bankroll", "max_drawdown"]:
        if col in stats.columns:
            stats[col] = pd.to_numeric(stats[col], errors="coerce")

    outcomes = inputs.get("bet_outcomes_summary")
    if outcomes is not None and not outcomes.empty:
        m_col = _find_col(outcomes, ["model"])
        home_col = _find_col(outcomes, ["home_avg_pnl", "home_pnl"])
        away_col = _find_col(outcomes, ["away_avg_pnl", "away_pnl"])
        if m_col and home_col and away_col:
            tmp = outcomes[[m_col, home_col, away_col]].copy()
            tmp[home_col] = pd.to_numeric(tmp[home_col], errors="coerce")
            tmp[away_col] = pd.to_numeric(tmp[away_col], errors="coerce")
            tmp["home_away_pnl_gap"] = tmp[home_col] - tmp[away_col]
            stats = stats.merge(tmp[[m_col, "home_away_pnl_gap"]].rename(columns={m_col: "model"}), on="model", how="left")

    rmse = inputs.get("model_rmse_selected")
    if rmse is not None and not rmse.empty:
        model_id_col = _find_col(rmse, ["model_id", "model"])
        group_col = _find_col(rmse, ["group_type"])
        rmse_mean_col = _find_col(rmse, ["rmse_mean", "rmse"])
        rmse_std_col = _find_col(rmse, ["rmse_std"])
        if model_id_col and group_col and rmse_mean_col:
            overall = rmse[rmse[group_col].astype(str).str.lower() == "overall"].copy()
            overall = overall.dropna(subset=[model_id_col])
            overall[rmse_mean_col] = pd.to_numeric(overall[rmse_mean_col], errors="coerce")
            if rmse_std_col:
                overall[rmse_std_col] = pd.to_numeric(overall[rmse_std_col], errors="coerce")
            cols = [model_id_col, rmse_mean_col] + ([rmse_std_col] if rmse_std_col else [])
            overall = overall[cols].drop_duplicates(subset=[model_id_col])
            rename_map = {model_id_col: "model", rmse_mean_col: "rmse_mean"}
            if rmse_std_col:
                rename_map[rmse_std_col] = "rmse_std"
            overall = overall.rename(columns=rename_map)
            stats = stats.merge(overall, on="model", how="left")

    vol = inputs.get("volatility_regime_summary")
    if vol is not None and not vol.empty:
        m_col = _find_col(vol, ["model"])
        b_col = _find_col(vol, ["vol_bucket"])
        p_col = _find_col(vol, ["avg_pnl_per_bet", "avg_daily_pnl"])
        if m_col and b_col and p_col:
            tmp = vol[[m_col, b_col, p_col]].copy()
            tmp[p_col] = pd.to_numeric(tmp[p_col], errors="coerce")
            pivot = tmp.pivot_table(index=m_col, columns=b_col, values=p_col, aggfunc="mean").reset_index()
            high_name = next((c for c in pivot.columns if str(c).lower() == "high_vol"), None)
            low_name = next((c for c in pivot.columns if str(c).lower() == "low_vol"), None)
            if high_name is not None and low_name is not None:
                pivot["vol_high_minus_low"] = pivot[high_name] - pivot[low_name]
                stats = stats.merge(
                    pivot[[m_col, "vol_high_minus_low"]].rename(columns={m_col: "model"}),
                    on="model",
                    how="left",
                )

    market = inputs.get("market_tier_split_summary")
    if market is not None and not market.empty:
        m_col = _find_col(market, ["model"])
        q_col = _find_col(market, ["market_match_quality", "market_tier", "tier"])
        p_col = _find_col(market, ["avg_pnl_per_bet", "roi_per_bet"])
        if m_col and q_col and p_col:
            tmp = market[[m_col, q_col, p_col]].copy()
            tmp[p_col] = pd.to_numeric(tmp[p_col], errors="coerce")
            pivot = tmp.pivot_table(index=m_col, columns=q_col, values=p_col, aggfunc="mean").reset_index()
            b_name = next((c for c in pivot.columns if str(c).lower() == "tier_b_moneyline"), None)
            d_name = next((c for c in pivot.columns if str(c).lower() == "tier_d_default"), None)
            if b_name is not None and d_name is not None:
                pivot["tier_b_minus_tier_d"] = pivot[b_name] - pivot[d_name]
                stats = stats.merge(
                    pivot[[m_col, "tier_b_minus_tier_d"]].rename(columns={m_col: "model"}),
                    on="model",
                    how="left",
                )

    sort_col = "roi" if "roi" in stats.columns else "model"
    return stats.sort_values(sort_col, ascending=False)


def build_feature_importance_proxy(
    inputs: Dict[str, Optional[pd.DataFrame]],
    top_n_per_model: int = 6,
    min_bets: int = 100,
) -> pd.DataFrame:
    factor_keys = [
        "confidence_factor_summary",
        "data_quality_factor_summary",
        "bet_type_factor_summary",
        "team_context_factor_summary",
    ]

    tables: List[pd.DataFrame] = []
    for key in factor_keys:
        df = inputs.get(key)
        if df is None or df.empty:
            continue
        model_col = _find_col(df, ["model"])
        group_col = _find_col(df, ["factor_group"])
        name_col = _find_col(df, ["factor_name"])
        level_col = _find_col(df, ["factor_level"])
        roi_col = _find_col(df, ["roi_per_bet", "avg_pnl_per_bet"])
        bets_col = _find_col(df, ["bets", "num_bets"])
        if not (model_col and group_col and name_col and level_col and roi_col and bets_col):
            continue

        tmp = df[[model_col, group_col, name_col, level_col, roi_col, bets_col]].copy()
        tmp = tmp.rename(
            columns={
                model_col: "model",
                group_col: "factor_group",
                name_col: "factor_name",
                level_col: "factor_level",
                roi_col: "roi_per_bet",
                bets_col: "bets",
            }
        )
        tmp["roi_per_bet"] = pd.to_numeric(tmp["roi_per_bet"], errors="coerce")
        tmp["bets"] = pd.to_numeric(tmp["bets"], errors="coerce")
        tmp = tmp.dropna(subset=["model", "factor_name", "factor_level", "roi_per_bet", "bets"])
        tmp = tmp[tmp["bets"] >= float(min_bets)]
        tables.append(tmp)

    if not tables:
        return pd.DataFrame()

    all_factors = pd.concat(tables, ignore_index=True)
    rows: List[Dict[str, object]] = []
    for (model, factor_group, factor_name), group in all_factors.groupby(["model", "factor_group", "factor_name"], dropna=False):
        if len(group) < 2:
            continue
        top = group.sort_values("roi_per_bet", ascending=False).iloc[0]
        bottom = group.sort_values("roi_per_bet", ascending=True).iloc[0]
        rows.append(
            {
                "model": model,
                "factor_group": factor_group,
                "factor_name": factor_name,
                "top_level": top["factor_level"],
                "top_roi_per_bet": float(top["roi_per_bet"]),
                "bottom_level": bottom["factor_level"],
                "bottom_roi_per_bet": float(bottom["roi_per_bet"]),
                "impact_spread": float(top["roi_per_bet"] - bottom["roi_per_bet"]),
                "level_count": int(group["factor_level"].nunique()),
                "total_bets": int(group["bets"].sum()),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_impact_spread"] = out["impact_spread"].abs()
    out = out.sort_values(["model", "abs_impact_spread"], ascending=[True, False])
    out = out.groupby("model", as_index=False).head(top_n_per_model)
    out = out.drop(columns=["abs_impact_spread"])
    return out


def build_next_steps(findings_df: pd.DataFrame) -> pd.DataFrame:
    if findings_df is None or findings_df.empty:
        return pd.DataFrame(
            [
                {
                    "priority": 1,
                    "action": "Collect additional diagnostics",
                    "rationale": "No findings could be generated from available files.",
                    "target_metric": "n/a",
                    "owner": "research",
                }
            ]
        )

    rows: List[Dict[str, object]] = []
    high = findings_df[findings_df["severity"] == "high"]
    medium = findings_df[findings_df["severity"] == "medium"]
    priority = 1

    for _, finding in high.head(4).iterrows():
        rows.append(
            {
                "priority": priority,
                "action": f"Address {finding['area']} risk: {finding['finding']}",
                "rationale": str(finding["recommendation"]),
                "target_metric": str(finding["metric"]),
                "owner": "modeling",
            }
        )
        priority += 1

    for _, finding in medium.head(3).iterrows():
        rows.append(
            {
                "priority": priority,
                "action": f"Improve {finding['area']}: {finding['finding']}",
                "rationale": str(finding["recommendation"]),
                "target_metric": str(finding["metric"]),
                "owner": "research",
            }
        )
        priority += 1

    if not rows:
        rows.append(
            {
                "priority": 1,
                "action": "Scale current profile cautiously",
                "rationale": "Findings are mostly positive; continue with guardrails and monitoring.",
                "target_metric": "portfolio_roi",
                "owner": "trading",
            }
        )

    return pd.DataFrame(rows)


def _build_output_dir(output_root: Path, generation_dt: datetime) -> Path:
    base_name = f"{OUTPUT_PREFIX}{generation_dt.strftime('%Y_%m_%d')}"
    output_dir = output_root / base_name
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
        return output_dir
    suffix = generation_dt.strftime("%H%M%S")
    output_dir = output_root / f"{base_name}_{suffix}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def write_brief(
    output_dir: Path,
    input_dir: Path,
    present_files: List[str],
    missing_files: List[str],
    best_model: Optional[str],
    analytics_df: pd.DataFrame,
    findings_df: pd.DataFrame,
    next_steps_df: pd.DataFrame,
    model_stats_df: pd.DataFrame,
    feature_proxy_df: pd.DataFrame,
    native_feature_importance_files: List[str],
) -> None:
    def _lookup(metric: str) -> Optional[object]:
        if analytics_df.empty:
            return None
        subset = analytics_df[analytics_df["metric"] == metric]
        if subset.empty:
            return None
        return subset.iloc[0]["value"]

    median_roi = _to_float(_lookup("median_model_roi"))
    avg_win = _to_float(_lookup("avg_model_win_rate"))
    feature_match = _to_float(_lookup("feature_match_rate"))
    neg_months = _to_float(_lookup("negative_months"))
    pos_months = _to_float(_lookup("positive_months"))
    slice_feature_count = _to_float(_lookup("factor_slice_features"))

    high_findings = findings_df[findings_df["severity"] == "high"] if not findings_df.empty else pd.DataFrame()
    medium_findings = findings_df[findings_df["severity"] == "medium"] if not findings_df.empty else pd.DataFrame()
    pos_findings = findings_df[findings_df["severity"] == "positive"] if not findings_df.empty else pd.DataFrame()

    lines = [
        "# Copilot Analysis Feedback Brief (Detailed)",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Source folder: {input_dir.as_posix()}",
        f"- Output folder: {output_dir.as_posix()}",
        f"- Files present: {len(present_files)}",
        f"- Files missing: {len(missing_files)}",
        "",
        "## Executive Snapshot",
        f"- Best model by available ranking: {best_model or 'n/a'}",
        f"- Median model ROI: {_format_pct(median_roi, 3)}",
        f"- Average model win rate: {_format_pct(avg_win, 2)}",
        f"- Feature join match rate: {_format_pct(feature_match, 2)}",
        f"- Aggregate positive vs negative months: {int(pos_months) if pos_months is not None else 'n/a'} / {int(neg_months) if neg_months is not None else 'n/a'}",
        f"- Feature columns eligible for factor slicing: {int(slice_feature_count) if slice_feature_count is not None else 'n/a'}",
        "",
        "## Detailed Findings",
    ]

    if high_findings.empty and medium_findings.empty:
        lines.append("- No high/medium findings were produced from the available artifacts.")
    else:
        if not high_findings.empty:
            lines.append("### High Severity")
            for _, row in high_findings.iterrows():
                lines.append(
                    f"- [{row['area']}] {row['finding']}: value={row['value']} vs threshold {row['threshold']}; action={row['recommendation']} (source={row['source_file']})"
                )
        if not medium_findings.empty:
            lines.append("### Medium Severity")
            for _, row in medium_findings.iterrows():
                lines.append(
                    f"- [{row['area']}] {row['finding']}: value={row['value']} vs threshold {row['threshold']}; action={row['recommendation']} (source={row['source_file']})"
                )

    lines.extend(["", "## Strengths"])
    if pos_findings.empty:
        lines.append("- No clear positive findings identified from available summaries.")
    else:
        for _, row in pos_findings.head(8).iterrows():
            lines.append(f"- [{row['area']}] {row['finding']}: {row['metric']}={row['value']} (source={row['source_file']})")

    lines.extend(["", "## Model Statistics"])
    if model_stats_df.empty:
        lines.append("- Model-level statistics could not be built from available files.")
    else:
        for _, row in model_stats_df.iterrows():
            model = row.get("model", "n/a")
            roi = _format_pct(_to_float(row.get("roi")), 3)
            win = _format_pct(_to_float(row.get("win_rate")), 2)
            bets = int(row.get("num_bets")) if _to_float(row.get("num_bets")) is not None else "n/a"
            rmse_mean = _format_float(_to_float(row.get("rmse_mean")), 3)
            drawdown = _format_pct(_to_float(row.get("max_drawdown")), 2)
            home_away_gap = _format_float(_to_float(row.get("home_away_pnl_gap")), 4)
            vol_gap = _format_float(_to_float(row.get("vol_high_minus_low")), 5)
            market_gap = _format_float(_to_float(row.get("tier_b_minus_tier_d")), 5)
            lines.append(
                f"- {model}: roi={roi}, win_rate={win}, bets={bets}, rmse_mean={rmse_mean}, max_drawdown={drawdown}, home_away_gap={home_away_gap}, high_minus_low_vol={vol_gap}, tier_b_minus_tier_d={market_gap}"
            )

    lines.extend(["", "## Feature Importance by Model"])
    if native_feature_importance_files:
        lines.append(f"- Native feature-importance files detected: {', '.join(native_feature_importance_files)}")
    else:
        lines.append("- No native feature-importance artifact found in this run folder; using factor-level ROI impact proxy.")

    if feature_proxy_df.empty:
        lines.append("- Feature-importance proxy table is empty (insufficient factor variation after filters).")
    else:
        for model, group in feature_proxy_df.groupby("model", dropna=False):
            lines.append(f"### {model}")
            for _, row in group.sort_values("impact_spread", ascending=False).head(4).iterrows():
                lines.append(
                    f"- {row['factor_group']}.{row['factor_name']}: impact_spread={_format_float(_to_float(row['impact_spread']), 5)}; best={row['top_level']}({_format_float(_to_float(row['top_roi_per_bet']), 5)}), worst={row['bottom_level']}({_format_float(_to_float(row['bottom_roi_per_bet']), 5)}), levels={int(row['level_count'])}, total_bets={int(row['total_bets'])}"
                )

    lines.extend(["", "## Suggested Next Steps"])
    if next_steps_df.empty:
        lines.append("- No prioritized actions generated.")
    else:
        for _, row in next_steps_df.sort_values("priority").head(8).iterrows():
            lines.append(f"- P{int(row['priority'])}: {row['action']} -> {row['rationale']} (target={row['target_metric']}, owner={row['owner']})")

    if missing_files:
        lines.extend(["", "## Missing Inputs", *[f"- {name}" for name in missing_files]])

    (output_dir / "brief.md").write_text("\n".join(lines), encoding="utf-8")


def run(input_dir: Path, output_root: Path) -> Path:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    now = datetime.now()
    output_dir = _build_output_dir(output_root=output_root, generation_dt=now)
    inputs, present_files, missing_files = load_inputs(input_dir)
    best_model = _pick_best_model(inputs.get("copilot_betting_summary"), inputs.get("profitable_models_summary"))

    analytics_df = build_analytics_summary(inputs)
    findings_df = build_feedback_findings(inputs, best_model=best_model)
    next_steps_df = build_next_steps(findings_df)
    model_stats_df = build_model_statistics(inputs)
    feature_proxy_df = build_feature_importance_proxy(inputs)

    analytics_df.to_csv(output_dir / "analytics_summary.csv", index=False)
    findings_df.to_csv(output_dir / "feedback_findings.csv", index=False)
    next_steps_df.to_csv(output_dir / "next_steps.csv", index=False)
    model_stats_df.to_csv(output_dir / "model_statistics.csv", index=False)
    feature_proxy_df.to_csv(output_dir / "feature_importance_proxy.csv", index=False)

    native_feature_importance_files = [p.name for p in sorted(input_dir.glob("*importance*.csv"))]

    manifest_df = pd.DataFrame(
        [
            {
                "generated_at": now.isoformat(timespec="seconds"),
                "source_dir": input_dir.as_posix(),
                "output_dir": output_dir.as_posix(),
                "present_file_count": len(present_files),
                "missing_file_count": len(missing_files),
                "present_files": "|".join(sorted(present_files)),
                "missing_files": "|".join(sorted(missing_files)),
                "native_feature_importance_files": "|".join(native_feature_importance_files),
            }
        ]
    )
    manifest_df.to_csv(output_dir / "run_manifest.csv", index=False)

    write_brief(
        output_dir=output_dir,
        input_dir=input_dir,
        present_files=present_files,
        missing_files=missing_files,
        best_model=best_model,
        analytics_df=analytics_df,
        findings_df=findings_df,
        next_steps_df=next_steps_df,
        model_stats_df=model_stats_df,
        feature_proxy_df=feature_proxy_df,
        native_feature_importance_files=native_feature_importance_files,
    )

    print("Feedback analysis completed.")
    print(f"Source: {input_dir.as_posix()}")
    print(f"Output: {output_dir.as_posix()}")
    print(f"Files present: {len(present_files)}")
    print(f"Files missing: {len(missing_files)}")
    print(f"Model statistics rows: {len(model_stats_df)}")
    print(f"Feature-importance proxy rows: {len(feature_proxy_df)}")

    if not findings_df.empty:
        print("Top findings:")
        for _, row in findings_df.head(3).iterrows():
            print(f"  - [{row['severity']}] {row['area']}: {row['finding']}")

    if not next_steps_df.empty:
        print("Top next steps:")
        for _, row in next_steps_df.sort_values("priority").head(3).iterrows():
            print(f"  - P{int(row['priority'])}: {row['action']}")

    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize copilot analysis outputs into concise feedback artifacts.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="Path to existing copilot analysis folder to summarize.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory where dated feedback folder will be created.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(input_dir=Path(args.input_dir), output_root=Path(args.output_root))


if __name__ == "__main__":
    main()
