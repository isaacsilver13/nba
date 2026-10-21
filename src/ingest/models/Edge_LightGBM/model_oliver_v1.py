"""model_oliver_v1.py

Self-contained NBA spread prediction ensemble incorporating Dean Oliver's
Basketball on Paper principles:
  - Four Factors (eFG%, TOV%, OREB%, FTR) as first-class features
  - Oliver weighted composite score
  - Pythagorean Win% differential (true team quality signal)

Reads:  data/processed/df_model_3.csv
Writes: outputs/model_oliver_v1_predictions.csv
        outputs/model_oliver_v1_rmse.csv
        outputs/model_oliver_v1_audit.csv
        outputs/model_oliver_v1_objective_sweep.csv

Bloat removed vs. model_ensemble_odds_v2026_02_19.py:
  - No config system imports (reads CSV directly)
  - Leakage columns excluded: abs_edge, edge_rank_today,
    edge_percentile_last_30_days, model_error_roll_last20
  - Redundant Elo transforms removed: abs_elo_diff, elo_diff_squared,
    sigmoid_elo_diff
  - Manual interaction features removed: rest_diff_elo_diff,
    home_elo_roll_2Y_rest_advantage, pace_diff_rest_diff,
    net_rating_diff_rest_diff (trees find these natively)
  - Pace collinearity removed: combined_pace, pace_interaction
  - Redundant margin features removed: home_margin_last5, away_margin_last5,
    pts_diff_last5, rolling_margin_diff_5
  - Redundant squares removed: rest_advantage_sq, streak_length_squared
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

INPUT_PATH = "data/processed/df_model_3.csv"
ODDS_PATH = "data/odds/nba_2008-2025.csv"
PROCESSED_GAMES_PATH = "data/processed/nba_games_with_game_id_processed.csv"

OUTPUT_PRED_PATH = "outputs/model_oliver_v1_predictions.csv"
OUTPUT_RMSE_PATH = "outputs/model_oliver_v1_rmse.csv"
OUTPUT_AUDIT_PATH = "outputs/model_oliver_v1_audit.csv"
OUTPUT_OBJECTIVE_SWEEP_PATH = "outputs/model_oliver_v1_objective_sweep.csv"

EDGE_STANDARDIZATION_MODE = "standardized"  # "standardized" or "raw"

TRAIN_SIZE = 2000
TEST_SIZE = 300

DEFAULT_AMERICAN_ODDS = -110.0
DEFAULT_PAYOUT = 100.0 / abs(DEFAULT_AMERICAN_ODDS)
EV_THRESHOLD = 0.02
FRACTIONAL_KELLY = 0.5
MAX_KELLY = 0.1

APPLY_MARKET_MATCH_PENALTY_TO_KELLY = True
MARKET_MATCH_QUALITY_PENALTY_WEIGHTS = {
    "tier_a_market": 1.00,
    "tier_b_moneyline": 0.92,
    "tier_c_mirrored": 0.80,
    "tier_d_default": 0.65,
    "tier_e_unknown": 0.75,
}

CONFIDENCE_SCORE_MODE = "abs_edge"
TOP_N_BETS_PER_DAY = 5
TOP_N_SWEEP = [2, 3, 5, 7, 10]
DAILY_MAX_EXPOSURE = 0.25
OBJECTIVE_LAMBDAS = [0.25, 0.5, 1.0]

SPREAD_BUCKETS = [
    (0, 2, "0-2"),
    (2, 5, "2-5"),
    (5, 10, "5-10"),
    (10, 15, "10-15"),
    (15, np.inf, "15+"),
]

# Oliver / Pythagorean constant
PYTH_EXPONENT = 16.5

# Oliver Four Factors weights (empirically derived in Basketball on Paper)
FF_WEIGHTS = {"efg": 0.40, "tov": 0.25, "oreb": 0.20, "ftr": 0.15}

TEAM_HOME_CANDIDATES = ["home", "home_team", "home_team_abbr", "home_abbr", "home_team_id"]
TEAM_AWAY_CANDIDATES = ["away", "away_team", "away_team_abbr", "away_abbr", "away_team_id"]

DYNAMIC_PREFIXES = (
    "home_team_adv_", "away_team_adv_",
    "home_team_ff_",  "away_team_ff_",
    "home_team_sc_",  "away_team_sc_",
    "home_team_usg_", "away_team_usg_",
    "home_team_misc_","away_team_misc_",
    "home_team_ptrk_","away_team_ptrk_",
)
DYNAMIC_SUFFIXES = ("_r5", "_r10", "_r20", "_ewm5", "_ewm10", "_ewm20", "_s2d")

TEAM_MAP = {
    "atl": "ATL", "bkn": "BKN", "bos": "BOS", "cha": "CHA", "chi": "CHI",
    "cle": "CLE", "dal": "DAL", "den": "DEN", "det": "DET", "gs":  "GSW",
    "hou": "HOU", "ind": "IND", "lac": "LAC", "lal": "LAL", "mem": "MEM",
    "mia": "MIA", "mil": "MIL", "min": "MIN", "no":  "NOP", "ny":  "NYK",
    "okc": "OKC", "orl": "ORL", "phi": "PHI", "phx": "PHX", "por": "POR",
    "sa":  "SAS", "sac": "SAC", "tor": "TOR", "utah":"UTA", "wsh": "WAS",
}

# Columns that must never be features (leakage, targets, identifiers)
DENY_EXACT = {
    # Targets
    "home_margin", "away_margin",
    # Result labels
    "did_home_cover", "home_cover", "favorite_cover",
    "favorite_cover_manual", "team_that_covered",
    # Leakage: computed by a prior model run
    "abs_edge", "edge_rank_today", "edge_percentile_last_30_days",
    "model_error_roll_last20",
    # Redundant Elo monotonic transforms (trees don't benefit)
    "abs_elo_diff", "elo_diff_squared", "sigmoid_elo_diff",
    # Manual interactions (trees find these automatically)
    "rest_diff_elo_diff", "home_elo_roll_2Y_rest_advantage",
    "pace_diff_rest_diff", "net_rating_diff_rest_diff",
    # Pace collinearity
    "combined_pace", "pace_interaction",
    # Redundant margin signals (rolling_margin_diff_10 captures the same info)
    "home_margin_last5", "away_margin_last5", "pts_diff_last5",
    "rolling_margin_diff_5",
    # Squared transforms (tree handles nonlinearity natively)
    "rest_advantage_sq", "streak_length_squared",
    # Raw spread (use signed version only)
    "spread",
}

DENY_SUBSTRINGS = (
    "model_error", "market_error", "result", "outcome", "postgame",
    "pred_", "sigma_", "error_", "abs_error", "edge_",
    "kelly_", "pnl_", "cum_pnl", "bet_",
)

# ---------------------------------------------------------------------------
# Static (non-dynamic) feature allowlist — curated with Oliver principles
# ---------------------------------------------------------------------------

STATIC_FEATURES = [
    # Spread / market context
    "spread_signed",
    "is_home_favorite",
    # Rest / schedule
    "home_b2b", "away_b2b",
    "rest_diff",
    "rest_advantage",
    "rest_weighted_load",
    "games_last_3_days", "games_last_5_days", "games_last_7_days",
    # Travel / fatigue
    "travel_diff_3d", "travel_diff_7d", "travel_diff_10d",
    "travel_1d_z", "travel_3d_z", "travel_7d_z", "travel_10d_z",
    "fatigue_diff",
    # Elo quality signals
    "elo_diff", "home_elo_roll_2Y", "away_elo_roll_2Y",
    # Recent form
    "margin_diff_last5",
    "rolling_margin_diff_10",
    "win_streak", "loss_streak",
    "after_ot_flag",
    "blowout_pct_last20", "blowout_pct_season",
    "ats_roll_last20",
    # Variance / consistency
    "margin_std_last_10", "home_margin_std_last_10", "away_margin_std_last_10",
    # Schedule load
    "top8_points_diff",
    # Injury / roster
    "missing_top1_flag", "missing_top2_flag",
    "minutes_lost_last_game", "usage_lost_last_game",
    # Ratings (short-window, non-dynamic)
    "bench_net_rating_last_5",
    "off_rating_last_5", "def_rating_last_5",
    "net_rating_last_5", "net_rating_trend",
    "net_rating_diff",
    # Style matchup
    "team_style_vs_opponent_style",
    # Pace (single predictor, no collinear variants)
    "pace_diff",
    # Rolling net box score r5
    "home_team_net_points_r5",   "home_team_net_rebounds_r5",  "home_team_net_assists_r5",
    "home_team_net_STL_r5",      "home_team_net_BLK_r5",       "home_team_net_turnovers_r5",
    "home_team_net_OREB_r5",     "home_team_net_fgm_r5",       "home_team_net_fga_r5",
    "home_team_net_three_pm_r5", "home_team_net_three_pa_r5",  "home_team_net_ftm_r5",
    "home_team_net_fta_r5",
    # Rolling net box score r10
    "home_team_net_points_r10",   "home_team_net_rebounds_r10", "home_team_net_assists_r10",
    "home_team_net_STL_r10",      "home_team_net_BLK_r10",      "home_team_net_turnovers_r10",
    "home_team_net_OREB_r10",     "home_team_net_fgm_r10",      "home_team_net_fga_r10",
    "home_team_net_three_pm_r10", "home_team_net_three_pa_r10", "home_team_net_ftm_r10",
    "home_team_net_fta_r10",
    "away_team_net_points_r5",   "away_team_net_rebounds_r5",  "away_team_net_assists_r5",
    "away_team_net_STL_r5",      "away_team_net_BLK_r5",       "away_team_net_turnovers_r5",
    "away_team_net_OREB_r5",     "away_team_net_fgm_r5",       "away_team_net_fga_r5",
    "away_team_net_three_pm_r5", "away_team_net_three_pa_r5",  "away_team_net_ftm_r5",
    "away_team_net_fta_r5",
    "away_team_net_points_r10",   "away_team_net_rebounds_r10", "away_team_net_assists_r10",
    "away_team_net_STL_r10",      "away_team_net_BLK_r10",      "away_team_net_turnovers_r10",
    "away_team_net_OREB_r10",     "away_team_net_fgm_r10",      "away_team_net_fga_r10",
    "away_team_net_three_pm_r10", "away_team_net_three_pa_r10", "away_team_net_ftm_r10",
    "away_team_net_fta_r10",
    # Payout info (used in EV / Kelly, not as a predictive signal directly)
    "payout_home", "payout_away",
    # --- Oliver derived features (added by add_oliver_features) ---
    "efg_diff",            # Four Factors: Factor 1 — Shooting
    "tov_pct_diff",        # Four Factors: Factor 2 — Turnovers (lower is better)
    "oreb_pct_diff",       # Four Factors: Factor 3 — Offensive rebounding
    "ftr_diff",            # Four Factors: Factor 4 — Free throw rate
    "oliver_ff_composite", # Weighted composite of all four factors
    "pyth_win_pct_home",   # Pythagorean W% home team (true quality)
    "pyth_win_pct_away",   # Pythagorean W% away team
    "pyth_diff",           # Pythagorean quality differential (mean-reversion signal)
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    model_id: str
    model: object
    needs_imputer: bool


# ---------------------------------------------------------------------------
# Oliver feature engineering
# ---------------------------------------------------------------------------

def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return column as float series, or NaN series if missing."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _pyth_win_pct(ortg: pd.Series, drtg: pd.Series, exp: float = PYTH_EXPONENT) -> pd.Series:
    """Oliver Pythagorean win probability: ortg^exp / (ortg^exp + drtg^exp)."""
    o = ortg.clip(lower=1.0)
    d = drtg.clip(lower=1.0)
    o_exp = o ** exp
    d_exp = d ** exp
    return o_exp / (o_exp + d_exp)


def add_oliver_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Dean Oliver Basketball on Paper features inline.

    Uses 10-game rolling Four Factors (r10 window) and advanced ratings.
    Falls back to NaN gracefully when source columns are absent.
    """
    df = df.copy()

    # --- Four Factors differentials (home minus away) ---
    h_efg  = _safe_col(df, "home_team_ff_effective_fg_pct_r10")
    a_efg  = _safe_col(df, "away_team_ff_effective_fg_pct_r10")
    h_tov  = _safe_col(df, "home_team_ff_turnover_pct_r10")
    a_tov  = _safe_col(df, "away_team_ff_turnover_pct_r10")
    h_oreb = _safe_col(df, "home_team_ff_off_reb_pct_r10")
    a_oreb = _safe_col(df, "away_team_ff_off_reb_pct_r10")
    h_ftr  = _safe_col(df, "home_team_ff_free_throw_rate_r10")
    a_ftr  = _safe_col(df, "away_team_ff_free_throw_rate_r10")

    df["efg_diff"]      = h_efg  - a_efg        # Higher = home shooting advantage
    df["tov_pct_diff"]  = h_tov  - a_tov        # Higher = home turns it over more (bad)
    df["oreb_pct_diff"] = h_oreb - a_oreb        # Higher = home crashes boards more
    df["ftr_diff"]      = h_ftr  - a_ftr         # Higher = home gets to line more

    # Oliver composite: +efg, -tov, +oreb, +ftr  (all from home perspective)
    df["oliver_ff_composite"] = (
        FF_WEIGHTS["efg"]  *  df["efg_diff"]
        - FF_WEIGHTS["tov"]  *  df["tov_pct_diff"]
        + FF_WEIGHTS["oreb"] *  df["oreb_pct_diff"]
        + FF_WEIGHTS["ftr"]  *  df["ftr_diff"]
    )

    # --- Pythagorean Win% (Oliver formula, exponent 16.5) ---
    h_ortg = _safe_col(df, "home_team_adv_off_rating_r10")
    h_drtg = _safe_col(df, "home_team_adv_def_rating_r10")
    a_ortg = _safe_col(df, "away_team_adv_off_rating_r10")
    a_drtg = _safe_col(df, "away_team_adv_def_rating_r10")

    df["pyth_win_pct_home"] = _pyth_win_pct(h_ortg, h_drtg)
    df["pyth_win_pct_away"] = _pyth_win_pct(a_ortg, a_drtg)
    df["pyth_diff"]         = df["pyth_win_pct_home"] - df["pyth_win_pct_away"]

    return df


# ---------------------------------------------------------------------------
# Odds bridge utilities
# ---------------------------------------------------------------------------

def canonical_game_id(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})

    def _canon(v):
        if pd.isna(v):
            return np.nan
        vv = str(v).lstrip("0")
        return vv if vv else "0"

    return s.map(_canon)


def normalize_team_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.strip().map(TEAM_MAP)


def to_payout_from_american(american: pd.Series) -> pd.Series:
    vals = pd.to_numeric(american, errors="coerce")
    payout = pd.Series(np.nan, index=vals.index, dtype=float)
    neg_mask = vals < 0
    pos_mask = vals > 0
    payout.loc[neg_mask] = 100.0 / vals.loc[neg_mask].abs()
    payout.loc[pos_mask] = vals.loc[pos_mask] / 100.0
    return payout


def infer_market_match_quality(payout_source: str, used_default_payout: bool) -> str:
    if bool(used_default_payout):
        return "tier_d_default"
    src = str(payout_source).lower()
    if "mirrored" in src:
        return "tier_c_mirrored"
    if "moneyline" in src:
        return "tier_b_moneyline"
    if "spread_or_market" in src:
        return "tier_a_market"
    return "tier_e_unknown"


def find_team_columns(df: pd.DataFrame) -> Tuple[str, str]:
    home_col = next((c for c in TEAM_HOME_CANDIDATES if c in df.columns), None)
    away_col = next((c for c in TEAM_AWAY_CANDIDATES if c in df.columns), None)
    if not home_col or not away_col:
        raise ValueError(
            "Home/away team columns not found. Expected one of: "
            f"{TEAM_HOME_CANDIDATES} / {TEAM_AWAY_CANDIDATES}"
        )
    return home_col, away_col


def extract_primary_side_prices(
    odds_df: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    home_candidates = [
        "spread_odds_home", "home_spread_odds", "spread_home_odds",
        "home_spread_price", "spread_price_home",
    ]
    away_candidates = [
        "spread_odds_away", "away_spread_odds", "spread_away_odds",
        "away_spread_price", "spread_price_away",
    ]

    home_price = pd.Series(np.nan, index=odds_df.index, dtype=float)
    away_price = pd.Series(np.nan, index=odds_df.index, dtype=float)
    home_src   = pd.Series("",    index=odds_df.index, dtype=object)
    away_src   = pd.Series("",    index=odds_df.index, dtype=object)

    for col in home_candidates:
        if col in odds_df.columns:
            vals = pd.to_numeric(odds_df[col], errors="coerce")
            mask = home_price.isna() & vals.notna()
            home_price.loc[mask] = vals.loc[mask]
            home_src.loc[mask]   = col

    for col in away_candidates:
        if col in odds_df.columns:
            vals = pd.to_numeric(odds_df[col], errors="coerce")
            mask = away_price.isna() & vals.notna()
            away_price.loc[mask] = vals.loc[mask]
            away_src.loc[mask]   = col

    if "moneyline_home" in odds_df.columns:
        vals = pd.to_numeric(odds_df["moneyline_home"], errors="coerce")
        mask = home_price.isna() & vals.notna()
        home_price.loc[mask] = vals.loc[mask]
        home_src.loc[mask]   = "moneyline_home"

    if "moneyline_away" in odds_df.columns:
        vals = pd.to_numeric(odds_df["moneyline_away"], errors="coerce")
        mask = away_price.isna() & vals.notna()
        away_price.loc[mask] = vals.loc[mask]
        away_src.loc[mask]   = "moneyline_away"

    return home_price, away_price, home_src, away_src


def build_odds_bridge() -> Tuple[pd.DataFrame, Dict]:
    empty = pd.DataFrame(columns=["GAME_ID_KEY", "home_american", "away_american", "home_src", "away_src"])

    if not os.path.exists(ODDS_PATH):
        return empty, {"odds_file_found": 0, "odds_rows_raw": 0, "odds_duplicate_collision_count": 0}

    odds = pd.read_csv(ODDS_PATH, parse_dates=["date"])
    odds["date"]  = pd.to_datetime(odds["date"]).dt.normalize()
    odds["home"]  = normalize_team_code(odds["home"])
    odds["away"]  = normalize_team_code(odds["away"])

    home_american, away_american, home_src, away_src = extract_primary_side_prices(odds)
    odds["home_american"] = home_american
    odds["away_american"] = away_american
    odds["home_src"]      = home_src
    odds["away_src"]      = away_src

    odds_key = ["date", "home", "away"]
    dup_sizes = odds.groupby(odds_key, dropna=False).size()
    duplicate_collision_count = int(dup_sizes[dup_sizes > 1].sum() - len(dup_sizes[dup_sizes > 1]))

    odds_dedup = odds.sort_values("date").drop_duplicates(odds_key, keep="last")

    if not os.path.exists(PROCESSED_GAMES_PATH):
        return empty, {
            "odds_file_found": 1,
            "odds_rows_raw": int(len(odds)),
            "odds_rows_dedup": int(len(odds_dedup)),
            "odds_duplicate_collision_count": duplicate_collision_count,
            "processed_games_found": 0,
        }

    games = pd.read_csv(PROCESSED_GAMES_PATH, usecols=["GAME_ID", "date", "home", "away"])
    games["date"]        = pd.to_datetime(games["date"]).dt.normalize()
    games["GAME_ID_KEY"] = canonical_game_id(games["GAME_ID"])

    odds_gid = odds_dedup.merge(games, on=["date", "home", "away"], how="inner", validate="m:1")
    cols = ["GAME_ID_KEY", "home_american", "away_american", "home_src", "away_src", "date", "home", "away"]
    odds_gid = odds_gid[cols].sort_values("date").drop_duplicates("GAME_ID_KEY", keep="last")

    return odds_gid, {
        "odds_file_found": 1,
        "processed_games_found": 1,
        "odds_rows_raw": int(len(odds)),
        "odds_rows_dedup": int(len(odds_dedup)),
        "odds_duplicate_collision_count": duplicate_collision_count,
        "odds_rows_with_game_id": int(len(odds_gid)),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_feature_table() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(INPUT_PATH, parse_dates=["date_x"])
    df = df.rename(columns={"date_x": "GAME_DATE"})
    df["GAME_DATE"]    = pd.to_datetime(df["GAME_DATE"]).dt.normalize()
    df["GAME_ID_KEY"]  = canonical_game_id(df["GAME_ID"]) if "GAME_ID" in df.columns else np.nan

    # Merge to get home/away team name columns (needed for RMSE breakdown)
    if os.path.exists(PROCESSED_GAMES_PATH):
        games = pd.read_csv(PROCESSED_GAMES_PATH, usecols=["GAME_ID", "date", "home", "away"])
        games["GAME_ID_KEY"] = canonical_game_id(games["GAME_ID"])
        games["date"]        = pd.to_datetime(games["date"]).dt.normalize()
        games = games.sort_values("date").drop_duplicates("GAME_ID_KEY", keep="last")

        if "GAME_ID_KEY" in df.columns:
            df = df.merge(
                games[["GAME_ID_KEY", "home", "away", "date"]],
                on="GAME_ID_KEY",
                how="left",
                validate="m:1",
            )

    # --- Odds bridge ---
    odds_gid, odds_meta = build_odds_bridge()

    df = df.merge(
        odds_gid[["GAME_ID_KEY", "home_american", "away_american", "home_src", "away_src"]],
        on="GAME_ID_KEY",
        how="left",
    )

    primary_match_mask = df[["home_american", "away_american"]].notna().any(axis=1)
    primary_match_count = int(primary_match_mask.sum())
    fallback_match_count = 0

    # Fallback: match by date + team names when GAME_ID_KEY join missed
    if {"home", "away", "GAME_DATE"}.issubset(df.columns) and len(odds_gid):
        fallback_key = odds_gid[["date", "home", "away", "home_american", "away_american", "home_src", "away_src"]].copy()
        fallback = df.loc[~primary_match_mask, ["GAME_DATE", "home", "away"]].merge(
            fallback_key,
            left_on=["GAME_DATE", "home", "away"],
            right_on=["date", "home", "away"],
            how="left",
        )
        if len(fallback):
            has_fallback = fallback[["home_american", "away_american"]].notna().any(axis=1).to_numpy()
            fallback_match_count = int(has_fallback.sum())

            target_idx       = df.index[~primary_match_mask]
            fallback_home    = pd.Series(fallback["home_american"].to_numpy(), index=target_idx)
            fallback_away    = pd.Series(fallback["away_american"].to_numpy(), index=target_idx)
            home_src_cur     = df.loc[target_idx, "home_src"].fillna("").to_numpy()
            away_src_cur     = df.loc[target_idx, "away_src"].fillna("").to_numpy()
            home_src_fill    = fallback["home_src"].fillna("").to_numpy()
            away_src_fill    = fallback["away_src"].fillna("").to_numpy()

            df.loc[target_idx, "home_american"] = df.loc[target_idx, "home_american"].combine_first(fallback_home)
            df.loc[target_idx, "away_american"] = df.loc[target_idx, "away_american"].combine_first(fallback_away)
            df.loc[target_idx, "home_src"]      = np.where(home_src_cur == "", home_src_fill, home_src_cur)
            df.loc[target_idx, "away_src"]      = np.where(away_src_cur == "", away_src_fill, away_src_cur)

    # Coerce odds to numeric; mirror and default fill
    df["home_american"] = pd.to_numeric(df["home_american"], errors="coerce")
    df["away_american"] = pd.to_numeric(df["away_american"], errors="coerce")

    mirror_home = df["home_american"].isna() & df["away_american"].notna()
    mirror_away = df["away_american"].isna() & df["home_american"].notna()
    df.loc[mirror_home, "home_american"] = df.loc[mirror_home, "away_american"]
    df.loc[mirror_away, "away_american"] = df.loc[mirror_away, "home_american"]

    default_home = df["home_american"].isna()
    default_away = df["away_american"].isna()
    df.loc[default_home, "home_american"] = DEFAULT_AMERICAN_ODDS
    df.loc[default_away, "away_american"] = DEFAULT_AMERICAN_ODDS

    df["payout_home"] = to_payout_from_american(df["home_american"])
    df["payout_away"] = to_payout_from_american(df["away_american"])

    invalid_home = df["payout_home"].isna() | (df["payout_home"] <= 0)
    invalid_away = df["payout_away"].isna() | (df["payout_away"] <= 0)
    df.loc[invalid_home, "payout_home"] = DEFAULT_PAYOUT
    df.loc[invalid_away, "payout_away"] = DEFAULT_PAYOUT

    def _src_label(row):
        h = row.get("home_src", "") or ""
        a = row.get("away_src", "") or ""
        h_tag = "default_-110" if h == "" and row.get("home_american") == DEFAULT_AMERICAN_ODDS else (
            "mirrored" if h == "" else ("moneyline" if h.startswith("moneyline") else "spread_or_market")
        )
        a_tag = "default_-110" if a == "" and row.get("away_american") == DEFAULT_AMERICAN_ODDS else (
            "mirrored" if a == "" else ("moneyline" if a.startswith("moneyline") else "spread_or_market")
        )
        return f"home:{h_tag}|away:{a_tag}"

    df["used_default_payout"]      = default_home | default_away | invalid_home | invalid_away
    df["payout_source"]            = df.apply(_src_label, axis=1)
    df["market_match_quality"]     = [
        infer_market_match_quality(src, used)
        for src, used in zip(df["payout_source"], df["used_default_payout"])
    ]
    df["market_match_penalty_weight"] = (
        df["market_match_quality"]
        .map(MARKET_MATCH_QUALITY_PENALTY_WEIGHTS)
        .fillna(MARKET_MATCH_QUALITY_PENALTY_WEIGHTS["tier_e_unknown"])
    )

    final_match_mask = (~default_home) | (~default_away)

    # --- Oliver features ---
    df = add_oliver_features(df)

    audit = {
        **odds_meta,
        "feature_rows": int(len(df)),
        "primary_match_count": primary_match_count,
        "fallback_match_count": fallback_match_count,
        "final_market_match_count": int(final_match_mask.sum()),
        "final_market_match_pct": float(final_match_mask.mean()) if len(df) else np.nan,
        "mirror_home_count": int(mirror_home.sum()),
        "mirror_away_count": int(mirror_away.sum()),
        "default_home_count": int(default_home.sum()),
        "default_away_count": int(default_away.sum()),
        "invalid_home_payout_count": int(invalid_home.sum()),
        "invalid_away_payout_count": int(invalid_away.sum()),
        "tier_a_market_count": int((df["market_match_quality"] == "tier_a_market").sum()),
        "tier_b_moneyline_count": int((df["market_match_quality"] == "tier_b_moneyline").sum()),
        "tier_c_mirrored_count": int((df["market_match_quality"] == "tier_c_mirrored").sum()),
        "tier_d_default_count": int((df["market_match_quality"] == "tier_d_default").sum()),
        "tier_e_unknown_count": int((df["market_match_quality"] == "tier_e_unknown").sum()),
    }

    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    return df, pd.DataFrame([audit])


# ---------------------------------------------------------------------------
# Feature set construction
# ---------------------------------------------------------------------------

def build_feature_sets(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    identifiers = [c for c in ["season", "GAME_ID", "GAME_ID_KEY", "GAME_DATE"] if c in df.columns]
    team_cols   = [c for c in ["home", "away", "date"] if c in df.columns]
    targets     = [c for c in ["home_margin", "away_margin"] if c in df.columns]

    # Dynamic columns from DYNAMIC_PREFIXES × DYNAMIC_SUFFIXES
    dynamic_cols = [
        c for c in df.columns
        if c.startswith(DYNAMIC_PREFIXES) and c.endswith(DYNAMIC_SUFFIXES)
    ]

    # Combine static allowlist + dynamic columns, filter to what's actually present
    candidates = sorted(set(
        [c for c in STATIC_FEATURES if c in df.columns] + dynamic_cols
    ))

    exclude = set(identifiers) | set(team_cols) | set(targets) | DENY_EXACT

    features = []
    for c in candidates:
        if c in exclude:
            continue
        cl = c.lower()
        if any(tok in cl for tok in DENY_SUBSTRINGS):
            continue
        features.append(c)

    # ensure spread_signed is present
    if "spread_signed" in df.columns and "spread_signed" not in features:
        features.append("spread_signed")

    features = sorted(set(features))
    edge_features = [c for c in features if "edge" in c.lower()]
    return features, edge_features, identifiers


# ---------------------------------------------------------------------------
# Edge standardization
# ---------------------------------------------------------------------------

def standardize_edge_cols(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    edge_cols: List[str],
    mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    if mode != "standardized" or not edge_cols:
        return X_train, X_test, {}

    stats = {}
    X_train = X_train.copy()
    X_test  = X_test.copy()

    for col in edge_cols:
        mean = X_train[col].mean()
        std  = X_train[col].std()
        if std == 0 or np.isnan(std):
            std = 1.0
        X_train[col] = (X_train[col] - mean) / std
        X_test[col]  = (X_test[col]  - mean) / std
        stats[col]   = (mean, std)

    return X_train, X_test, stats


# ---------------------------------------------------------------------------
# Walk-forward splitter
# ---------------------------------------------------------------------------

def walk_forward_splits(df: pd.DataFrame, time_col: str, train_size: int, test_size: int):
    df    = df.sort_values(time_col).reset_index(drop=True)
    start = 0
    while True:
        train_end = start + train_size
        test_end  = train_end + test_size
        if test_end > len(df):
            break
        yield df.iloc[start:train_end], df.iloc[train_end:test_end]
        start += test_size


# ---------------------------------------------------------------------------
# Model specs
# ---------------------------------------------------------------------------

def build_model_specs() -> List[ModelSpec]:
    specs = []

    for idx, params in enumerate([
        {"n_estimators": 800,  "learning_rate": 0.05, "max_depth": 4, "subsample": 0.8,
         "colsample_bytree": 0.8, "objective": "reg:squarederror", "random_state": 42},
        {"n_estimators": 1200, "learning_rate": 0.02, "max_depth": 6, "subsample": 0.8,
         "colsample_bytree": 0.8, "objective": "reg:squarederror", "random_state": 42},
    ], start=1):
        specs.append(ModelSpec(f"xgboost_v{idx}", XGBRegressor(**params), needs_imputer=False))

    for idx, params in enumerate([
        {"n_estimators": 1200, "learning_rate": 0.02, "max_depth": 4, "num_leaves": 31,
         "min_child_samples": 30, "subsample": 0.8, "colsample_bytree": 0.8,
         "random_state": 42, "verbose": -1},
        {"n_estimators": 800,  "learning_rate": 0.05, "max_depth": 6, "num_leaves": 63,
         "min_child_samples": 20, "subsample": 0.9, "colsample_bytree": 0.9,
         "random_state": 42, "verbose": -1},
    ], start=1):
        specs.append(ModelSpec(f"lightgbm_v{idx}", lgb.LGBMRegressor(**params), needs_imputer=False))

    specs.append(ModelSpec("linear_regression", LinearRegression(), needs_imputer=True))
    specs.append(ModelSpec(
        "elasticnet",
        ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=20000,
                   tol=1e-3, selection="random"),
        needs_imputer=True,
    ))
    specs.append(ModelSpec(
        "random_forest",
        RandomForestRegressor(n_estimators=600, max_depth=12, min_samples_leaf=5, random_state=42),
        needs_imputer=True,
    ))
    specs.append(ModelSpec(
        "gbr",
        GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42),
        needs_imputer=True,
    ))

    return specs


# ---------------------------------------------------------------------------
# Sigma helper
# ---------------------------------------------------------------------------

def compute_sigma(y_true: pd.Series, y_pred: np.ndarray) -> float:
    residuals = y_true - y_pred
    sigma = float(np.nanstd(residuals))
    return sigma if sigma > 0 else 1.0


# ---------------------------------------------------------------------------
# Betting metrics
# ---------------------------------------------------------------------------

def apply_betting_metrics(
    df: pd.DataFrame,
    pred_col: str,
    sigma_col: str,
    suffix: str,
) -> pd.DataFrame:
    edge_col        = f"edge_{suffix}"
    edge_std_col    = f"edge_std_{suffix}"
    win_prob_col    = f"win_prob_home_{suffix}"
    ev_home_col     = f"ev_home_{suffix}"
    ev_away_col     = f"ev_away_{suffix}"
    bet_side_col    = f"bet_side_{suffix}"
    kelly_col       = f"kelly_frac_{suffix}"
    kelly_raw_col   = f"kelly_frac_raw_{suffix}"
    kelly_pen_col   = f"kelly_penalty_weight_{suffix}"
    bet_win_col     = f"bet_win_{suffix}"
    pnl_col         = f"pnl_kelly_{suffix}"
    cum_pnl_col     = f"cum_pnl_{suffix}"

    new_cols = {}

    edge     = df[pred_col] - df["spread_signed"]
    edge_std = edge / df[sigma_col].replace(0, np.nan)
    win_prob = norm.cdf(edge_std.fillna(0.0))

    payout_home = pd.to_numeric(df.get("payout_home", DEFAULT_PAYOUT), errors="coerce").fillna(DEFAULT_PAYOUT)
    payout_away = pd.to_numeric(df.get("payout_away", DEFAULT_PAYOUT), errors="coerce").fillna(DEFAULT_PAYOUT)
    payout_home = payout_home.where(payout_home > 0, DEFAULT_PAYOUT)
    payout_away = payout_away.where(payout_away > 0, DEFAULT_PAYOUT)

    new_cols[edge_col]     = edge
    new_cols[edge_std_col] = edge_std
    new_cols[win_prob_col] = win_prob

    ev_home = win_prob * payout_home - (1 - win_prob)
    ev_away = (1 - win_prob) * payout_away - win_prob
    new_cols[ev_home_col] = ev_home
    new_cols[ev_away_col] = ev_away

    bet_side = np.where(
        ev_home > EV_THRESHOLD, "HOME",
        np.where(ev_away > EV_THRESHOLD, "AWAY", "NO BET"),
    )
    new_cols[bet_side_col] = bet_side

    kelly     = np.zeros(len(df))
    home_mask = bet_side == "HOME"
    away_mask = bet_side == "AWAY"

    kelly_home = ((win_prob[home_mask] * (payout_home[home_mask] + 1) - 1) / payout_home[home_mask]).clip(lower=0)
    kelly_away = (((1 - win_prob[away_mask]) * (payout_away[away_mask] + 1) - 1) / payout_away[away_mask]).clip(lower=0)
    kelly[home_mask] = kelly_home
    kelly[away_mask] = kelly_away
    kelly     = np.clip(kelly, 0, 1)
    kelly     = np.minimum(kelly * FRACTIONAL_KELLY, MAX_KELLY)
    kelly_raw = kelly.copy()

    kelly_penalty = pd.to_numeric(df.get("market_match_penalty_weight", 1.0), errors="coerce").fillna(1.0)
    if APPLY_MARKET_MATCH_PENALTY_TO_KELLY:
        kelly = np.clip(kelly * kelly_penalty.to_numpy(), 0, MAX_KELLY)

    new_cols[kelly_raw_col] = kelly_raw
    new_cols[kelly_pen_col] = kelly_penalty
    new_cols[kelly_col]     = kelly

    home_margin_needed = -df["spread_signed"]
    if "home_margin_needed" not in df.columns:
        new_cols["home_margin_needed"] = home_margin_needed

    did_home_cover = df["home_margin"] > home_margin_needed
    if "did_home_cover" not in df.columns:
        new_cols["did_home_cover"] = did_home_cover
    if "team_that_covered" not in df.columns:
        new_cols["team_that_covered"] = np.where(did_home_cover, "HOME", "AWAY")

    team_that_covered = new_cols.get("team_that_covered", df.get("team_that_covered"))
    bet_win = (bet_side == team_that_covered).astype(int)
    new_cols[bet_win_col] = bet_win

    pnl       = np.zeros(len(df))
    home_win  = (bet_win == 1) & home_mask
    away_win  = (bet_win == 1) & away_mask
    losses    = (bet_win == 0) & (bet_side != "NO BET")

    pnl[home_win] = kelly[home_win] * payout_home[home_win]
    pnl[away_win] = kelly[away_win] * payout_away[away_win]
    pnl[losses]   = -kelly[losses]

    new_cols[pnl_col]     = pnl
    new_cols[cum_pnl_col] = pd.Series(pnl, index=df.index).fillna(0).cumsum()

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# ---------------------------------------------------------------------------
# Daily top-N and exposure controls
# ---------------------------------------------------------------------------

def apply_daily_topn_and_exposure_controls(
    df: pd.DataFrame,
    suffix: str,
    top_n: int,
    daily_max_exposure: float,
) -> pd.DataFrame:
    work = df.copy()

    edge_col        = f"edge_{suffix}"
    bet_side_col    = f"bet_side_{suffix}"
    bet_win_col     = f"bet_win_{suffix}"
    kelly_col       = f"kelly_frac_{suffix}"
    pnl_col         = f"pnl_kelly_{suffix}"
    cum_pnl_col     = f"cum_pnl_{suffix}"
    rank_col        = f"edge_rank_day_{suffix}"
    confidence_col  = f"confidence_abs_edge_{suffix}"
    selected_col    = f"selected_topn_{suffix}"
    exposure_pre    = f"daily_exposure_pre_{suffix}"
    exposure_scale  = f"daily_exposure_scale_{suffix}"

    payout_home = pd.to_numeric(work.get("payout_home", DEFAULT_PAYOUT), errors="coerce").fillna(DEFAULT_PAYOUT)
    payout_away = pd.to_numeric(work.get("payout_away", DEFAULT_PAYOUT), errors="coerce").fillna(DEFAULT_PAYOUT)

    work[confidence_col] = pd.to_numeric(work[edge_col], errors="coerce").abs()
    is_candidate         = work[bet_side_col].isin(["HOME", "AWAY"])
    work[rank_col]       = np.nan
    work.loc[is_candidate, rank_col] = (
        work.loc[is_candidate]
        .groupby("GAME_DATE")[confidence_col]
        .rank(method="first", ascending=False)
    )
    work[selected_col]   = is_candidate & (work[rank_col] <= top_n)

    kelly_selected = np.where(work[selected_col], pd.to_numeric(work[kelly_col], errors="coerce").fillna(0.0), 0.0)
    work[exposure_pre]   = pd.Series(kelly_selected, index=work.index).groupby(work["GAME_DATE"]).transform("sum")

    daily_scale = np.where(
        work[exposure_pre] > daily_max_exposure,
        daily_max_exposure / work[exposure_pre].replace(0, np.nan),
        1.0,
    )
    daily_scale = pd.Series(daily_scale, index=work.index).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    work[exposure_scale] = daily_scale
    work[kelly_col]      = np.clip(kelly_selected * work[exposure_scale], 0, MAX_KELLY)

    home_mask = work[bet_side_col] == "HOME"
    away_mask = work[bet_side_col] == "AWAY"
    bet_win   = work[bet_win_col].fillna(0).astype(int)

    pnl       = np.zeros(len(work))
    home_win  = (bet_win == 1) & home_mask & work[selected_col]
    away_win  = (bet_win == 1) & away_mask & work[selected_col]
    losses    = (bet_win == 0) & work[selected_col]

    pnl[home_win] = work.loc[home_win, kelly_col].to_numpy() * payout_home.loc[home_win].to_numpy()
    pnl[away_win] = work.loc[away_win, kelly_col].to_numpy() * payout_away.loc[away_win].to_numpy()
    pnl[losses]   = -work.loc[losses, kelly_col].to_numpy()

    work[pnl_col]     = pnl
    work[cum_pnl_col] = pd.Series(pnl, index=work.index).fillna(0).cumsum()
    return work


# ---------------------------------------------------------------------------
# Objective sweep helper
# ---------------------------------------------------------------------------

def summarize_strategy_variant(
    df: pd.DataFrame,
    suffix: str,
    top_n: int,
    daily_max_exposure: float,
    objective_type: str,
    objective_lambda: float,
) -> Dict:
    trial     = apply_daily_topn_and_exposure_controls(df, suffix, top_n=top_n, daily_max_exposure=daily_max_exposure)
    kelly_col = f"kelly_frac_{suffix}"
    pnl_col   = f"pnl_kelly_{suffix}"

    active   = trial[kelly_col] > 0
    pnl      = pd.to_numeric(trial[pnl_col], errors="coerce").fillna(0.0)
    growth   = (1.0 + pnl).clip(lower=1e-12)
    bankroll = growth.cumprod()
    drawdown = bankroll / bankroll.cummax().replace(0, np.nan) - 1.0

    max_dd        = float(drawdown.min()) if len(drawdown) else 0.0
    log_growth    = float(np.log(growth).mean()) if len(growth) else 0.0
    ending_br     = float(bankroll.iloc[-1]) if len(bankroll) else 1.0
    roi           = ending_br - 1.0

    objective_score = (
        log_growth - objective_lambda * abs(max_dd)
        if objective_type == "log_growth_dd"
        else roi
    )

    return {
        "objective_type":     objective_type,
        "objective_lambda":   objective_lambda,
        "top_n_bets_per_day": int(top_n),
        "daily_max_exposure": float(daily_max_exposure),
        "bets_placed":        int(active.sum()),
        "win_rate":           float(trial.loc[active, f"bet_win_{suffix}"].mean()) if active.any() else np.nan,
        "avg_kelly":          float(trial.loc[active, kelly_col].mean()) if active.any() else 0.0,
        "avg_abs_edge":       float(trial.loc[active, f"edge_{suffix}"].abs().mean()) if active.any() else 0.0,
        "total_pnl":          float(pnl.sum()),
        "ending_bankroll":    ending_br,
        "roi":                float(roi),
        "max_drawdown":       max_dd,
        "log_growth_mean":    log_growth,
        "objective_score":    float(objective_score),
    }


# ---------------------------------------------------------------------------
# RMSE summary
# ---------------------------------------------------------------------------

def assign_spread_bucket(spread_signed: float) -> str:
    spread_abs = abs(spread_signed)
    for lo, hi, label in SPREAD_BUCKETS:
        if lo <= spread_abs < hi:
            return label
    return "15+"


def build_rmse_summary(
    df: pd.DataFrame,
    models: List[str],
    home_team_col: str,
    away_team_col: str,
) -> pd.DataFrame:
    rows = []
    df   = df.copy()
    df["spread_bucket"] = df["spread_signed"].apply(assign_spread_bucket)
    df["home_away"]     = np.where(df["home_margin"] > 0, "HOME_WIN", "AWAY_WIN")

    group_specs = [
        ("overall",    None),
        ("spread_bucket", "spread_bucket"),
        ("home_team",  home_team_col),
        ("away_team",  away_team_col),
        ("home_away",  "home_away"),
    ]

    for model_id in models:
        error_col = f"error_{model_id}"
        for group_type, group_col in group_specs:
            grp = [("all", df)] if group_col is None else df.groupby(group_col)
            for group_value, gdf in grp:
                rmse = float(np.sqrt(mean_squared_error(gdf["home_margin"], gdf[f"pred_{model_id}"])))
                mae  = float(np.mean(np.abs(gdf[error_col])))
                rows.append({
                    "group_type":  group_type,
                    "group_value": group_value,
                    "model_id":    model_id,
                    "rmse":        rmse,
                    "mae":         mae,
                    "count":       len(gdf),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading feature table…")
    df, audit_df = load_feature_table()

    features, edge_features, identifiers = build_feature_sets(df)
    print(f"  Features selected: {len(features)}")

    oliver_present = [f for f in ["efg_diff", "tov_pct_diff", "oreb_pct_diff", "ftr_diff",
                                   "oliver_ff_composite", "pyth_win_pct_home", "pyth_win_pct_away",
                                   "pyth_diff"] if f in features]
    print(f"  Oliver features active: {oliver_present}")

    home_team_col, away_team_col = find_team_columns(df)

    target   = "home_margin"
    time_col = "GAME_DATE"

    # Coerce all numeric columns
    for col in features + [target, "spread_signed", "payout_home", "payout_away"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    min_required = TRAIN_SIZE + TEST_SIZE
    if len(df) < min_required:
        train_size = int(len(df) * 0.7)
        test_size  = max(int(len(df) * 0.15), 1)
        print(f"  Small dataset — using train={train_size}, test={test_size}")
    else:
        train_size = TRAIN_SIZE
        test_size  = TEST_SIZE

    model_specs  = build_model_specs()
    pred_store   = {s.model_id: pd.Series(index=df.index, dtype=float) for s in model_specs}
    sigma_store  = {s.model_id: pd.Series(index=df.index, dtype=float) for s in model_specs}
    rmse_by_model = {s.model_id: [] for s in model_specs}

    for fold, (train_df, test_df) in enumerate(
        walk_forward_splits(df, time_col, train_size, test_size), start=1
    ):
        print(f"  Fold {fold}: train={len(train_df)}, test={len(test_df)}")

        X_train = train_df[features]
        y_train = train_df[target]
        X_test  = test_df[features]
        y_test  = test_df[target]

        non_all_nan     = X_train.columns[X_train.notna().any()].tolist()
        X_train         = X_train[non_all_nan]
        X_test          = X_test[non_all_nan]
        edge_fold       = [c for c in edge_features if c in non_all_nan]
        X_train, X_test, _ = standardize_edge_cols(X_train, X_test, edge_fold, EDGE_STANDARDIZATION_MODE)

        for spec in model_specs:
            X_tr = X_train
            X_te = X_test

            if spec.needs_imputer:
                imp  = SimpleImputer(strategy="median")
                X_tr = pd.DataFrame(imp.fit_transform(X_tr), columns=non_all_nan, index=X_train.index)
                X_te = pd.DataFrame(imp.transform(X_te),     columns=non_all_nan, index=X_test.index)

            spec.model.fit(X_tr, y_train)
            pred_test  = spec.model.predict(X_te)
            pred_train = spec.model.predict(X_tr)

            sigma = compute_sigma(y_train, pred_train)
            rmse  = float(np.sqrt(mean_squared_error(y_test, pred_test)))
            rmse_by_model[spec.model_id].append(rmse)

            pred_store[spec.model_id].loc[test_df.index]  = pred_test
            sigma_store[spec.model_id].loc[test_df.index] = sigma

    # Assemble output
    pred_indices = pd.concat(pred_store.values(), axis=1).dropna(how="all").index
    output_df    = df.loc[pred_indices].copy()

    keep_output = identifiers + [
        "home_margin", "spread_signed",
        home_team_col, away_team_col,
        "payout_home", "payout_away",
        "payout_source", "used_default_payout",
        "market_match_quality", "market_match_penalty_weight",
        # Include Oliver features in output for inspection
        "efg_diff", "tov_pct_diff", "oreb_pct_diff", "ftr_diff",
        "oliver_ff_composite", "pyth_win_pct_home", "pyth_win_pct_away", "pyth_diff",
    ]
    keep_output = [c for c in keep_output if c in output_df.columns]
    output_df   = output_df[keep_output]
    output_df["edge_standardization_mode"] = EDGE_STANDARDIZATION_MODE

    for spec in model_specs:
        mid       = spec.model_id
        pred_col  = f"pred_{mid}"
        sigma_col = f"sigma_{mid}"
        vals = {
            pred_col:             pred_store[mid].loc[pred_indices].values,
            sigma_col:            sigma_store[mid].loc[pred_indices].values,
        }
        vals[f"error_{mid}"]     = vals[pred_col] - output_df["home_margin"].values
        vals[f"abs_error_{mid}"] = np.abs(vals[f"error_{mid}"])
        output_df = pd.concat([output_df, pd.DataFrame(vals, index=output_df.index)], axis=1)
        output_df = apply_betting_metrics(output_df, pred_col, sigma_col, mid)

    # RMSE summary
    rmse_rows = []
    for mid, rmses in rmse_by_model.items():
        rmse_rows.append({
            "model_id":   mid,
            "rmse_mean":  float(np.mean(rmses)) if rmses else np.nan,
            "rmse_std":   float(np.std(rmses))  if rmses else np.nan,
            "folds":      len(rmses),
        })
    rmse_summary_df = pd.DataFrame(rmse_rows)

    # RMSE-weighted ensemble
    weights    = rmse_summary_df.set_index("model_id")["rmse_mean"].to_dict()
    weights    = {k: 1.0 / v for k, v in weights.items() if v and not np.isnan(v)}
    weight_sum = sum(weights.values())
    weights    = {k: v / weight_sum for k, v in weights.items()} if weight_sum else {}

    objective_sweep_df = pd.DataFrame()

    if weights:
        ensemble_pred = sum(output_df[f"pred_{mid}"] * w for mid, w in weights.items())
        ensemble_cols = {
            "pred_ensemble":  ensemble_pred,
            "sigma_ensemble": output_df[[f"sigma_{m}" for m in weights]].mean(axis=1),
        }
        ensemble_cols["error_ensemble"]     = ensemble_cols["pred_ensemble"] - output_df["home_margin"].values
        ensemble_cols["abs_error_ensemble"] = np.abs(ensemble_cols["error_ensemble"])
        output_df = pd.concat([output_df, pd.DataFrame(ensemble_cols, index=output_df.index)], axis=1)
        output_df = apply_betting_metrics(output_df, "pred_ensemble", "sigma_ensemble", "ensemble")

        output_for_sweep = output_df.copy()
        output_df = apply_daily_topn_and_exposure_controls(
            output_df, suffix="ensemble", top_n=TOP_N_BETS_PER_DAY, daily_max_exposure=DAILY_MAX_EXPOSURE
        )

        sweep_rows = []
        for top_n in TOP_N_SWEEP:
            sweep_rows.append(summarize_strategy_variant(
                output_for_sweep, "ensemble", top_n, DAILY_MAX_EXPOSURE, "roi", 0.0
            ))
            for lam in OBJECTIVE_LAMBDAS:
                sweep_rows.append(summarize_strategy_variant(
                    output_for_sweep, "ensemble", top_n, DAILY_MAX_EXPOSURE, "log_growth_dd", float(lam)
                ))
        objective_sweep_df = pd.DataFrame(sweep_rows)

    output_df = output_df.sort_values("GAME_DATE").reset_index(drop=True)

    # Detailed RMSE breakdown
    all_models    = [s.model_id for s in model_specs] + (["ensemble"] if "pred_ensemble" in output_df.columns else [])
    rmse_detailed = build_rmse_summary(output_df, all_models, home_team_col, away_team_col)
    rmse_out      = pd.concat(
        [rmse_summary_df.assign(group_type="summary"), rmse_detailed],
        axis=0, ignore_index=True,
    )

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PRED_PATH), exist_ok=True)
    output_df.to_csv(OUTPUT_PRED_PATH, index=False)
    rmse_out.to_csv(OUTPUT_RMSE_PATH, index=False)
    audit_df.to_csv(OUTPUT_AUDIT_PATH, index=False)
    objective_sweep_df.to_csv(OUTPUT_OBJECTIVE_SWEEP_PATH, index=False)

    print(f"\nSaved predictions  → {OUTPUT_PRED_PATH}")
    print(f"Saved RMSE summary → {OUTPUT_RMSE_PATH}")
    print(f"Saved audit        → {OUTPUT_AUDIT_PATH}")
    print(f"Saved obj sweep    → {OUTPUT_OBJECTIVE_SWEEP_PATH}")

    # Quick sanity print
    if "pred_ensemble" in output_df.columns and "home_margin" in output_df.columns:
        overall_rmse = float(np.sqrt(mean_squared_error(
            output_df["home_margin"], output_df["pred_ensemble"]
        )))
        print(f"\nEnsemble RMSE (all folds): {overall_rmse:.4f}")


if __name__ == "__main__":
    main()
