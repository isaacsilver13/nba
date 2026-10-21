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

import sys
from pathlib import Path

try:
    from ingest.dataPrep.config import get_config, resolve_step_config
except ImportError:
    try:
        from dataPrep.config import get_config, resolve_step_config
    except ImportError:
        src_root = Path(__file__).resolve().parents[3]
        if src_root.exists() and str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))
        from ingest.dataPrep.config import get_config, resolve_step_config


OUTPUT_PRED_PATH = "outputs/copilot_model_output_odds_v2026_02_19.csv"
OUTPUT_RMSE_PATH = "outputs/copilot_model_rmse_odds_v2026_02_19.csv"
OUTPUT_AUDIT_PATH = "outputs/copilot_model_join_audit_odds_v2026_02_19.csv"
OUTPUT_OBJECTIVE_SWEEP_PATH = "outputs/copilot_objective_sweep_odds_v2026_02_19.csv"

ODDS_PATH = "data/odds/nba_2008-2025.csv"
PROCESSED_GAMES_PATH = "data/processed/nba_games_with_game_id_processed.csv"

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

TEAM_HOME_CANDIDATES = [
    "home",
    "home_team",
    "home_team_abbr",
    "home_abbr",
    "home_team_id",
]
TEAM_AWAY_CANDIDATES = [
    "away",
    "away_team",
    "away_team_abbr",
    "away_abbr",
    "away_team_id",
]

DYNAMIC_PREFIXES = (
    "home_team_adv_",
    "away_team_adv_",
    "home_team_ff_",
    "away_team_ff_",
    "home_team_sc_",
    "away_team_sc_",
    "home_team_usg_",
    "away_team_usg_",
    "home_team_misc_",
    "away_team_misc_",
    "home_team_ptrk_",
    "away_team_ptrk_",
)
DYNAMIC_SUFFIXES = ("_r5", "_r10", "_r20", "_ewm5", "_ewm10", "_ewm20", "_s2d")

TEAM_MAP = {
    "atl": "ATL", "bkn": "BKN", "bos": "BOS", "cha": "CHA", "chi": "CHI", "cle": "CLE", "dal": "DAL",
    "den": "DEN", "det": "DET", "gs": "GSW", "hou": "HOU", "ind": "IND", "lac": "LAC", "lal": "LAL",
    "mem": "MEM", "mia": "MIA", "mil": "MIL", "min": "MIN", "no": "NOP", "ny": "NYK", "okc": "OKC",
    "orl": "ORL", "phi": "PHI", "phx": "PHX", "por": "POR", "sa": "SAS", "sac": "SAC", "tor": "TOR",
    "utah": "UTA", "wsh": "WAS",
}

MODERATE_DENY_EXACT = {
    "home_margin",
    "away_margin",
    "did_home_cover",
    "home_cover",
    "favorite_cover",
    "favorite_cover_manual",
    "team_that_covered",
    "abs_edge",
    "edge_rank_today",
    "edge_percentile_last_30_days",
    "model_error_roll_last20",
}

MODERATE_DENY_SUBSTRINGS = (
    "model_error",
    "market_error",
    "result",
    "outcome",
    "postgame",
    "pred_",
    "sigma_",
    "error_",
    "abs_error",
    "edge_",
    "kelly_",
    "pnl_",
    "cum_pnl",
    "bet_",
)


@dataclass
class ModelSpec:
    model_id: str
    model: object
    needs_imputer: bool


def get_step6a_output_path() -> str:
    cfg = resolve_step_config(get_config(), "step6a")
    return cfg["paths"]["output_model"]


def get_step6a_columns() -> List[str]:
    cfg = resolve_step_config(get_config(), "step6a")
    return cfg["columns"]["keep"]


def canonical_game_id(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})

    def _canon(v):
        if pd.isna(v):
            return np.nan
        vv = str(v)
        vv = vv.lstrip("0")
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
            "Home/away team identifiers were not found. Expected columns like "
            "'home'/'away' or 'home_team'/'away_team'."
        )
    return home_col, away_col


def extract_primary_side_prices(odds_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    home_candidates = [
        "spread_odds_home", "home_spread_odds", "spread_home_odds", "home_spread_price", "spread_price_home"
    ]
    away_candidates = [
        "spread_odds_away", "away_spread_odds", "spread_away_odds", "away_spread_price", "spread_price_away"
    ]

    home_src = pd.Series("", index=odds_df.index, dtype=object)
    away_src = pd.Series("", index=odds_df.index, dtype=object)

    home_price = pd.Series(np.nan, index=odds_df.index, dtype=float)
    away_price = pd.Series(np.nan, index=odds_df.index, dtype=float)

    for col in home_candidates:
        if col in odds_df.columns:
            col_vals = pd.to_numeric(odds_df[col], errors="coerce")
            use_mask = home_price.isna() & col_vals.notna()
            home_price.loc[use_mask] = col_vals.loc[use_mask]
            home_src.loc[use_mask] = col

    for col in away_candidates:
        if col in odds_df.columns:
            col_vals = pd.to_numeric(odds_df[col], errors="coerce")
            use_mask = away_price.isna() & col_vals.notna()
            away_price.loc[use_mask] = col_vals.loc[use_mask]
            away_src.loc[use_mask] = col

    if "moneyline_home" in odds_df.columns:
        mlh = pd.to_numeric(odds_df["moneyline_home"], errors="coerce")
        use_mask = home_price.isna() & mlh.notna()
        home_price.loc[use_mask] = mlh.loc[use_mask]
        home_src.loc[use_mask] = "moneyline_home"

    if "moneyline_away" in odds_df.columns:
        mla = pd.to_numeric(odds_df["moneyline_away"], errors="coerce")
        use_mask = away_price.isna() & mla.notna()
        away_price.loc[use_mask] = mla.loc[use_mask]
        away_src.loc[use_mask] = "moneyline_away"

    return home_price, away_price, home_src, away_src


def build_odds_bridge() -> Tuple[pd.DataFrame, Dict[str, float]]:
    if not os.path.exists(ODDS_PATH):
        return pd.DataFrame(columns=["GAME_ID_KEY", "payout_home", "payout_away", "payout_source"]), {
            "odds_file_found": 0,
            "odds_rows_raw": 0,
            "odds_duplicate_collision_count": 0,
        }

    odds = pd.read_csv(ODDS_PATH, parse_dates=["date"])
    odds["date"] = pd.to_datetime(odds["date"]).dt.normalize()
    odds["home"] = normalize_team_code(odds["home"])
    odds["away"] = normalize_team_code(odds["away"])

    home_american, away_american, home_src, away_src = extract_primary_side_prices(odds)

    odds["home_american"] = home_american
    odds["away_american"] = away_american
    odds["home_src"] = home_src
    odds["away_src"] = away_src

    odds_key = ["date", "home", "away"]
    dup_sizes = odds.groupby(odds_key, dropna=False).size()
    duplicate_collision_count = int(dup_sizes[dup_sizes > 1].sum() - len(dup_sizes[dup_sizes > 1]))

    odds_dedup = odds.sort_values("date").drop_duplicates(odds_key, keep="last")

    if not os.path.exists(PROCESSED_GAMES_PATH):
        return pd.DataFrame(columns=["GAME_ID_KEY", "payout_home", "payout_away", "payout_source"]), {
            "odds_file_found": 1,
            "odds_rows_raw": int(len(odds)),
            "odds_rows_dedup": int(len(odds_dedup)),
            "odds_duplicate_collision_count": duplicate_collision_count,
            "processed_games_found": 0,
        }

    games = pd.read_csv(PROCESSED_GAMES_PATH, usecols=["GAME_ID", "date", "home", "away"])
    games["date"] = pd.to_datetime(games["date"]).dt.normalize()
    games["GAME_ID_KEY"] = canonical_game_id(games["GAME_ID"])

    odds_gid = odds_dedup.merge(games, on=["date", "home", "away"], how="inner", validate="m:1")

    odds_gid = odds_gid[["GAME_ID_KEY", "home_american", "away_american", "home_src", "away_src", "date", "home", "away"]].copy()
    odds_gid = odds_gid.sort_values("date").drop_duplicates("GAME_ID_KEY", keep="last")

    return odds_gid, {
        "odds_file_found": 1,
        "processed_games_found": 1,
        "odds_rows_raw": int(len(odds)),
        "odds_rows_dedup": int(len(odds_dedup)),
        "odds_duplicate_collision_count": duplicate_collision_count,
        "odds_rows_with_game_id": int(len(odds_gid)),
    }


def load_feature_table() -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = get_step6a_output_path()
    df = pd.read_csv(path, parse_dates=["date_x"])
    if "date_x" in df.columns:
        df = df.rename(columns={"date_x": "GAME_DATE"})

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.normalize()
    df["GAME_ID_KEY"] = canonical_game_id(df["GAME_ID"]) if "GAME_ID" in df.columns else np.nan

    games = None
    if os.path.exists(PROCESSED_GAMES_PATH):
        games = pd.read_csv(PROCESSED_GAMES_PATH, usecols=["GAME_ID", "date", "home", "away"])
        games["GAME_ID_KEY"] = canonical_game_id(games["GAME_ID"])
        games["date"] = pd.to_datetime(games["date"]).dt.normalize()
        games = games.sort_values("date").drop_duplicates("GAME_ID_KEY", keep="last")

        if "GAME_ID_KEY" in df.columns:
            df = df.merge(
                games[["GAME_ID_KEY", "home", "away", "date"]],
                on="GAME_ID_KEY",
                how="left",
                validate="m:1",
            )

    odds_gid, odds_meta = build_odds_bridge()

    df = df.merge(
        odds_gid[["GAME_ID_KEY", "home_american", "away_american", "home_src", "away_src"]],
        on="GAME_ID_KEY",
        how="left",
    )

    primary_match_mask = df[["home_american", "away_american"]].notna().any(axis=1)
    primary_match_count = int(primary_match_mask.sum())

    fallback_match_count = 0
    if {"home", "away", "GAME_DATE"}.issubset(df.columns):
        fallback_key_df = odds_gid[["date", "home", "away", "home_american", "away_american", "home_src", "away_src"]].copy()
        fallback = df.loc[~primary_match_mask, ["GAME_DATE", "home", "away"]].merge(
            fallback_key_df,
            left_on=["GAME_DATE", "home", "away"],
            right_on=["date", "home", "away"],
            how="left",
        )

        if len(fallback):
            has_fallback = fallback[["home_american", "away_american"]].notna().any(axis=1).to_numpy()
            fallback_match_count = int(has_fallback.sum())

            target_idx = df.index[~primary_match_mask]
            fallback_home = pd.Series(fallback["home_american"].to_numpy(), index=target_idx)
            fallback_away = pd.Series(fallback["away_american"].to_numpy(), index=target_idx)

            df.loc[target_idx, "home_american"] = df.loc[target_idx, "home_american"].combine_first(fallback_home)
            df.loc[target_idx, "away_american"] = df.loc[target_idx, "away_american"].combine_first(fallback_away)

            home_src_current = df.loc[target_idx, "home_src"].fillna("").to_numpy()
            away_src_current = df.loc[target_idx, "away_src"].fillna("").to_numpy()
            home_src_fill = fallback["home_src"].fillna("").to_numpy()
            away_src_fill = fallback["away_src"].fillna("").to_numpy()
            df.loc[target_idx, "home_src"] = np.where(home_src_current == "", home_src_fill, home_src_current)
            df.loc[target_idx, "away_src"] = np.where(away_src_current == "", away_src_fill, away_src_current)

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
        home_src = row.get("home_src", "") or ""
        away_src = row.get("away_src", "") or ""
        home_tag = "spread_or_market"
        away_tag = "spread_or_market"

        if home_src == "":
            if row.get("home_american", np.nan) == DEFAULT_AMERICAN_ODDS:
                home_tag = "default_-110"
            else:
                home_tag = "mirrored"
        elif home_src.startswith("moneyline"):
            home_tag = "moneyline"

        if away_src == "":
            if row.get("away_american", np.nan) == DEFAULT_AMERICAN_ODDS:
                away_tag = "default_-110"
            else:
                away_tag = "mirrored"
        elif away_src.startswith("moneyline"):
            away_tag = "moneyline"

        return f"home:{home_tag}|away:{away_tag}"

    df["used_default_payout"] = default_home | default_away | invalid_home | invalid_away
    df["payout_source"] = df.apply(_src_label, axis=1)
    df["market_match_quality"] = [
        infer_market_match_quality(src, used_default)
        for src, used_default in zip(df["payout_source"], df["used_default_payout"])
    ]
    df["market_match_penalty_weight"] = (
        df["market_match_quality"].map(MARKET_MATCH_QUALITY_PENALTY_WEIGHTS).fillna(MARKET_MATCH_QUALITY_PENALTY_WEIGHTS["tier_e_unknown"])
    )

    final_match_mask = (~default_home) | (~default_away)

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
    audit_df = pd.DataFrame([audit])
    return df, audit_df


def build_feature_sets(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    base_keep = [c for c in get_step6a_columns() if c in df.columns]
    dynamic_cols = [
        c
        for c in df.columns
        if c.startswith(DYNAMIC_PREFIXES) and c.endswith(DYNAMIC_SUFFIXES)
    ]

    identifiers = [c for c in ["season", "GAME_ID", "GAME_ID_KEY", "GAME_DATE"] if c in df.columns]
    team_cols = [c for c in ["home", "away"] if c in df.columns]
    targets = [c for c in ["home_margin", "away_margin"] if c in df.columns]

    candidates = sorted(set(base_keep + dynamic_cols + ["payout_home", "payout_away"]))

    features = []
    for c in candidates:
        cl = c.lower()
        if c in identifiers or c in targets or c in team_cols:
            continue
        if c in MODERATE_DENY_EXACT:
            continue
        if any(tok in cl for tok in MODERATE_DENY_SUBSTRINGS):
            continue
        if c == "spread":
            continue
        features.append(c)

    if "spread_signed" in df.columns and "spread_signed" not in features:
        features.append("spread_signed")

    edge_features = [c for c in features if "edge" in c.lower()]
    return sorted(set(features)), edge_features, identifiers


def standardize_edge_cols(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    edge_cols: List[str],
    mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Tuple[float, float]]]:
    if mode != "standardized" or not edge_cols:
        return X_train, X_test, {}

    stats = {}
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in edge_cols:
        mean = X_train[col].mean()
        std = X_train[col].std()
        if std == 0 or np.isnan(std):
            std = 1.0
        X_train[col] = (X_train[col] - mean) / std
        X_test[col] = (X_test[col] - mean) / std
        stats[col] = (mean, std)

    return X_train, X_test, stats


def walk_forward_splits(
    df: pd.DataFrame,
    time_col: str,
    train_size: int,
    test_size: int,
):
    df = df.sort_values(time_col).reset_index(drop=True)
    start = 0
    while True:
        train_end = start + train_size
        test_end = train_end + test_size
        if test_end > len(df):
            break
        yield df.iloc[start:train_end], df.iloc[train_end:test_end]
        start += test_size


def build_model_specs() -> List[ModelSpec]:
    specs = []

    xgb_params = [
        {
            "n_estimators": 800,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": 42,
        },
        {
            "n_estimators": 1200,
            "learning_rate": 0.02,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": 42,
        },
    ]

    for idx, params in enumerate(xgb_params, start=1):
        specs.append(ModelSpec(model_id=f"xgboost_v{idx}", model=XGBRegressor(**params), needs_imputer=False))

    lgb_params = [
        {
            "n_estimators": 1200,
            "learning_rate": 0.02,
            "max_depth": 4,
            "num_leaves": 31,
            "min_child_samples": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1,
        },
        {
            "n_estimators": 800,
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 63,
            "min_child_samples": 20,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "verbose": -1,
        },
    ]

    for idx, params in enumerate(lgb_params, start=1):
        specs.append(ModelSpec(model_id=f"lightgbm_v{idx}", model=lgb.LGBMRegressor(**params), needs_imputer=False))

    specs.append(ModelSpec(model_id="linear_regression", model=LinearRegression(), needs_imputer=True))
    specs.append(
        ModelSpec(
            model_id="elasticnet",
            model=ElasticNet(
                alpha=1.0,
                l1_ratio=0.5,
                random_state=42,
                max_iter=20000,
                tol=1e-3,
                selection="random",
            ),
            needs_imputer=True,
        )
    )
    specs.append(
        ModelSpec(
            model_id="random_forest",
            model=RandomForestRegressor(n_estimators=600, max_depth=12, min_samples_leaf=5, random_state=42),
            needs_imputer=True,
        )
    )
    specs.append(
        ModelSpec(
            model_id="gbr",
            model=GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42),
            needs_imputer=True,
        )
    )

    return specs


def compute_sigma(y_true: pd.Series, y_pred: np.ndarray) -> float:
    residuals = y_true - y_pred
    sigma = float(np.nanstd(residuals))
    return sigma if sigma > 0 else 1.0


def apply_betting_metrics(
    df: pd.DataFrame,
    pred_col: str,
    sigma_col: str,
    suffix: str,
) -> pd.DataFrame:
    edge_col = f"edge_{suffix}"
    edge_std_col = f"edge_std_{suffix}"
    win_prob_col = f"win_prob_home_{suffix}"
    ev_home_col = f"ev_home_{suffix}"
    ev_away_col = f"ev_away_{suffix}"
    bet_side_col = f"bet_side_{suffix}"
    kelly_col = f"kelly_frac_{suffix}"
    kelly_raw_col = f"kelly_frac_raw_{suffix}"
    kelly_penalty_col = f"kelly_penalty_weight_{suffix}"
    bet_win_col = f"bet_win_{suffix}"
    pnl_col = f"pnl_kelly_{suffix}"
    cum_pnl_col = f"cum_pnl_{suffix}"

    new_cols = {}

    edge = df[pred_col] - df["spread_signed"]
    edge_std = edge / df[sigma_col].replace(0, np.nan)
    win_prob = norm.cdf(edge_std.fillna(0.0))

    payout_home = pd.to_numeric(df.get("payout_home", DEFAULT_PAYOUT), errors="coerce").fillna(DEFAULT_PAYOUT)
    payout_away = pd.to_numeric(df.get("payout_away", DEFAULT_PAYOUT), errors="coerce").fillna(DEFAULT_PAYOUT)
    payout_home = payout_home.where(payout_home > 0, DEFAULT_PAYOUT)
    payout_away = payout_away.where(payout_away > 0, DEFAULT_PAYOUT)

    new_cols[edge_col] = edge
    new_cols[edge_std_col] = edge_std
    new_cols[win_prob_col] = win_prob

    ev_home = win_prob * payout_home - (1 - win_prob)
    ev_away = (1 - win_prob) * payout_away - win_prob
    new_cols[ev_home_col] = ev_home
    new_cols[ev_away_col] = ev_away

    bet_side = np.where(
        ev_home > EV_THRESHOLD,
        "HOME",
        np.where(ev_away > EV_THRESHOLD, "AWAY", "NO BET"),
    )
    new_cols[bet_side_col] = bet_side

    kelly = np.zeros(len(df))
    home_mask = bet_side == "HOME"
    away_mask = bet_side == "AWAY"

    kelly_home = ((win_prob[home_mask] * (payout_home[home_mask] + 1) - 1) / payout_home[home_mask]).clip(lower=0)
    kelly_away = (((1 - win_prob[away_mask]) * (payout_away[away_mask] + 1) - 1) / payout_away[away_mask]).clip(lower=0)

    kelly[home_mask] = kelly_home
    kelly[away_mask] = kelly_away
    kelly = np.clip(kelly, 0, 1)
    kelly = np.minimum(kelly * FRACTIONAL_KELLY, MAX_KELLY)
    kelly_raw = kelly.copy()

    kelly_penalty = pd.to_numeric(df.get("market_match_penalty_weight", 1.0), errors="coerce").fillna(1.0)
    if APPLY_MARKET_MATCH_PENALTY_TO_KELLY:
        kelly = np.clip(kelly * kelly_penalty.to_numpy(), 0, MAX_KELLY)

    new_cols[kelly_raw_col] = kelly_raw
    new_cols[kelly_penalty_col] = kelly_penalty
    new_cols[kelly_col] = kelly

    if "home_margin_needed" not in df.columns:
        new_cols["home_margin_needed"] = -df["spread_signed"]

    if "home_margin_needed" in new_cols:
        home_margin_needed = new_cols["home_margin_needed"]
    elif "home_margin_needed" in df.columns:
        home_margin_needed = df["home_margin_needed"]
    else:
        home_margin_needed = -df["spread_signed"]

    did_home_cover = df["home_margin"] > home_margin_needed
    if "did_home_cover" not in df.columns:
        new_cols["did_home_cover"] = did_home_cover

    if "team_that_covered" not in df.columns:
        new_cols["team_that_covered"] = np.where(did_home_cover, "HOME", "AWAY")

    team_that_covered = new_cols.get("team_that_covered", df.get("team_that_covered"))
    bet_win = (bet_side == team_that_covered).astype(int)
    new_cols[bet_win_col] = bet_win

    pnl = np.zeros(len(df))
    home_win = (bet_win == 1) & home_mask
    away_win = (bet_win == 1) & away_mask
    losses = (bet_win == 0) & (bet_side != "NO BET")

    pnl[home_win] = kelly[home_win] * payout_home[home_win]
    pnl[away_win] = kelly[away_win] * payout_away[away_win]
    pnl[losses] = -kelly[losses]

    new_cols[pnl_col] = pnl
    new_cols[cum_pnl_col] = pd.Series(pnl, index=df.index).fillna(0).cumsum()

    new_df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return new_df


def apply_daily_topn_and_exposure_controls(
    df: pd.DataFrame,
    suffix: str,
    top_n: int,
    daily_max_exposure: float,
) -> pd.DataFrame:
    work = df.copy()

    edge_col = f"edge_{suffix}"
    bet_side_col = f"bet_side_{suffix}"
    bet_win_col = f"bet_win_{suffix}"
    kelly_col = f"kelly_frac_{suffix}"
    pnl_col = f"pnl_kelly_{suffix}"
    cum_pnl_col = f"cum_pnl_{suffix}"

    rank_col = f"edge_rank_day_{suffix}"
    confidence_col = f"confidence_abs_edge_{suffix}"
    selected_col = f"selected_topn_{suffix}"
    exposure_pre_col = f"daily_exposure_pre_{suffix}"
    exposure_scale_col = f"daily_exposure_scale_{suffix}"

    payout_home = pd.to_numeric(work.get("payout_home", DEFAULT_PAYOUT), errors="coerce").fillna(DEFAULT_PAYOUT)
    payout_away = pd.to_numeric(work.get("payout_away", DEFAULT_PAYOUT), errors="coerce").fillna(DEFAULT_PAYOUT)

    work[confidence_col] = pd.to_numeric(work[edge_col], errors="coerce").abs() if CONFIDENCE_SCORE_MODE == "abs_edge" else 0.0

    is_candidate = work[bet_side_col].isin(["HOME", "AWAY"])
    work[rank_col] = np.nan
    work.loc[is_candidate, rank_col] = (
        work.loc[is_candidate]
        .groupby("GAME_DATE")[confidence_col]
        .rank(method="first", ascending=False)
    )

    work[selected_col] = is_candidate & (work[rank_col] <= top_n)

    kelly_selected = np.where(work[selected_col], pd.to_numeric(work[kelly_col], errors="coerce").fillna(0.0), 0.0)
    work[exposure_pre_col] = pd.Series(kelly_selected, index=work.index).groupby(work["GAME_DATE"]).transform("sum")

    daily_scale = np.where(
        work[exposure_pre_col] > daily_max_exposure,
        daily_max_exposure / work[exposure_pre_col].replace(0, np.nan),
        1.0,
    )
    daily_scale = pd.Series(daily_scale, index=work.index).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    work[exposure_scale_col] = daily_scale

    work[kelly_col] = np.clip(kelly_selected * work[exposure_scale_col], 0, MAX_KELLY)

    home_mask = work[bet_side_col] == "HOME"
    away_mask = work[bet_side_col] == "AWAY"
    bet_win = work[bet_win_col].fillna(0).astype(int)

    pnl = np.zeros(len(work))
    home_win = (bet_win == 1) & home_mask & work[selected_col]
    away_win = (bet_win == 1) & away_mask & work[selected_col]
    losses = (bet_win == 0) & work[selected_col]

    pnl[home_win] = work.loc[home_win, kelly_col].to_numpy() * payout_home.loc[home_win].to_numpy()
    pnl[away_win] = work.loc[away_win, kelly_col].to_numpy() * payout_away.loc[away_win].to_numpy()
    pnl[losses] = -work.loc[losses, kelly_col].to_numpy()

    work[pnl_col] = pnl
    work[cum_pnl_col] = pd.Series(pnl, index=work.index).fillna(0).cumsum()
    return work


def summarize_strategy_variant(
    df: pd.DataFrame,
    suffix: str,
    top_n: int,
    daily_max_exposure: float,
    objective_type: str,
    objective_lambda: float,
) -> Dict[str, float]:
    trial = apply_daily_topn_and_exposure_controls(df, suffix, top_n=top_n, daily_max_exposure=daily_max_exposure)

    kelly_col = f"kelly_frac_{suffix}"
    pnl_col = f"pnl_kelly_{suffix}"

    active = trial[kelly_col] > 0
    pnl = pd.to_numeric(trial[pnl_col], errors="coerce").fillna(0.0)
    growth = 1.0 + pnl
    growth = growth.clip(lower=1e-12)
    bankroll = growth.cumprod()
    running_max = bankroll.cummax().replace(0, np.nan)
    drawdown = bankroll / running_max - 1.0

    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    log_growth_mean = float(np.log(growth).mean()) if len(growth) else 0.0
    ending_bankroll = float(bankroll.iloc[-1]) if len(bankroll) else 1.0
    roi = ending_bankroll - 1.0

    if objective_type == "log_growth_dd":
        objective_score = log_growth_mean - objective_lambda * abs(max_drawdown)
    else:
        objective_score = roi

    return {
        "objective_type": objective_type,
        "objective_lambda": objective_lambda,
        "top_n_bets_per_day": int(top_n),
        "daily_max_exposure": float(daily_max_exposure),
        "bets_placed": int(active.sum()),
        "win_rate": float(trial.loc[active, f"bet_win_{suffix}"].mean()) if active.any() else np.nan,
        "avg_kelly": float(trial.loc[active, kelly_col].mean()) if active.any() else 0.0,
        "avg_abs_edge": float(trial.loc[active, f"edge_{suffix}"].abs().mean()) if active.any() else 0.0,
        "total_pnl": float(pnl.sum()),
        "ending_bankroll": ending_bankroll,
        "roi": float(roi),
        "max_drawdown": max_drawdown,
        "log_growth_mean": log_growth_mean,
        "objective_score": float(objective_score),
    }


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

    df = df.copy()
    df["spread_bucket"] = df["spread_signed"].apply(assign_spread_bucket)
    df["home_away"] = np.where(df["home_margin"] > 0, "HOME_WIN", "AWAY_WIN")

    group_specs = [
        ("overall", None),
        ("spread_bucket", "spread_bucket"),
        ("home_team", home_team_col),
        ("away_team", away_team_col),
        ("home_away", "home_away"),
    ]

    for model_id in models:
        error_col = f"error_{model_id}"
        for group_type, group_col in group_specs:
            if group_col is None:
                grp = [("all", df)]
            else:
                grp = df.groupby(group_col)
            for group_value, gdf in grp:
                rmse = float(np.sqrt(mean_squared_error(gdf["home_margin"], gdf[f"pred_{model_id}"])))
                mae = np.mean(np.abs(gdf[error_col]))
                rows.append(
                    {
                        "group_type": group_type,
                        "group_value": group_value,
                        "model_id": model_id,
                        "rmse": rmse,
                        "mae": mae,
                        "count": len(gdf),
                    }
                )

    return pd.DataFrame(rows)


def main():
    df, audit_df = load_feature_table()
    features, edge_features, identifiers = build_feature_sets(df)

    home_team_col, away_team_col = find_team_columns(df)

    target = "home_margin"
    time_col = "GAME_DATE"

    numeric_cols = [c for c in features + [target, "spread_signed", "payout_home", "payout_away"] if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    min_required = TRAIN_SIZE + TEST_SIZE
    if len(df) < min_required:
        train_size = int(len(df) * 0.7)
        test_size = max(int(len(df) * 0.15), 1)
    else:
        train_size = TRAIN_SIZE
        test_size = TEST_SIZE

    model_specs = build_model_specs()

    pred_store = {spec.model_id: pd.Series(index=df.index, dtype=float) for spec in model_specs}
    sigma_store = {spec.model_id: pd.Series(index=df.index, dtype=float) for spec in model_specs}
    rmse_by_model = {spec.model_id: [] for spec in model_specs}

    for fold, (train_df, test_df) in enumerate(walk_forward_splits(df, time_col, train_size, test_size), start=1):
        print(f"Fold {fold}: Train {len(train_df)} rows, Test {len(test_df)} rows")

        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]

        non_all_nan = X_train.columns[X_train.notna().any()].tolist()
        X_train = X_train[non_all_nan]
        X_test = X_test[non_all_nan]
        edge_features_fold = [c for c in edge_features if c in non_all_nan]

        X_train, X_test, _ = standardize_edge_cols(X_train, X_test, edge_features_fold, EDGE_STANDARDIZATION_MODE)

        for spec in model_specs:
            X_train_fold = X_train
            X_test_fold = X_test

            if spec.needs_imputer:
                imputer = SimpleImputer(strategy="median")
                X_train_fold = pd.DataFrame(imputer.fit_transform(X_train_fold), columns=non_all_nan, index=X_train.index)
                X_test_fold = pd.DataFrame(imputer.transform(X_test_fold), columns=non_all_nan, index=X_test.index)

            model = spec.model
            model.fit(X_train_fold, y_train)

            pred_test = model.predict(X_test_fold)
            pred_train = model.predict(X_train_fold)

            sigma = compute_sigma(y_train, pred_train)

            rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
            rmse_by_model[spec.model_id].append(rmse)

            pred_store[spec.model_id].loc[test_df.index] = pred_test
            sigma_store[spec.model_id].loc[test_df.index] = sigma

    pred_indices = pd.concat(pred_store.values(), axis=1).dropna(how="all").index
    output_df = df.loc[pred_indices].copy()

    keep_output = identifiers + [
        "home_margin",
        "spread_signed",
        home_team_col,
        away_team_col,
        "payout_home",
        "payout_away",
        "payout_source",
        "used_default_payout",
        "market_match_quality",
        "market_match_penalty_weight",
    ]
    keep_output = [c for c in keep_output if c in output_df.columns]
    output_df = output_df[keep_output]
    output_df["edge_standardization_mode"] = EDGE_STANDARDIZATION_MODE

    for spec in model_specs:
        model_id = spec.model_id
        pred_col = f"pred_{model_id}"
        sigma_col = f"sigma_{model_id}"

        model_cols = {
            pred_col: pred_store[model_id].loc[pred_indices].values,
            sigma_col: sigma_store[model_id].loc[pred_indices].values,
        }
        model_cols[f"error_{model_id}"] = model_cols[pred_col] - output_df["home_margin"].values
        model_cols[f"abs_error_{model_id}"] = np.abs(model_cols[f"error_{model_id}"])

        output_df = pd.concat([output_df, pd.DataFrame(model_cols, index=output_df.index)], axis=1)
        output_df = apply_betting_metrics(output_df, pred_col, sigma_col, model_id)

    rmse_summary_rows = []
    for model_id, rmses in rmse_by_model.items():
        rmse_summary_rows.append(
            {
                "model_id": model_id,
                "rmse_mean": float(np.mean(rmses)) if rmses else np.nan,
                "rmse_std": float(np.std(rmses)) if rmses else np.nan,
                "folds": len(rmses),
            }
        )

    rmse_summary_df = pd.DataFrame(rmse_summary_rows)

    weights = rmse_summary_df.set_index("model_id")["rmse_mean"].to_dict()
    weights = {k: 1.0 / v for k, v in weights.items() if v and not np.isnan(v)}
    weight_sum = sum(weights.values())
    weights = {k: v / weight_sum for k, v in weights.items()} if weight_sum else {}

    if weights:
        ensemble_pred = None
        for model_id, w in weights.items():
            pred_col = f"pred_{model_id}"
            if ensemble_pred is None:
                ensemble_pred = output_df[pred_col] * w
            else:
                ensemble_pred = ensemble_pred + output_df[pred_col] * w

        ensemble_cols = {
            "pred_ensemble": ensemble_pred,
            "sigma_ensemble": output_df[[f"sigma_{m}" for m in weights]].mean(axis=1),
        }
        ensemble_cols["error_ensemble"] = ensemble_cols["pred_ensemble"] - output_df["home_margin"].values
        ensemble_cols["abs_error_ensemble"] = np.abs(ensemble_cols["error_ensemble"])
        output_df = pd.concat([output_df, pd.DataFrame(ensemble_cols, index=output_df.index)], axis=1)
        output_df = apply_betting_metrics(output_df, "pred_ensemble", "sigma_ensemble", "ensemble")

        output_for_sweep = output_df.copy()

        output_df = apply_daily_topn_and_exposure_controls(
            output_df,
            suffix="ensemble",
            top_n=TOP_N_BETS_PER_DAY,
            daily_max_exposure=DAILY_MAX_EXPOSURE,
        )

        objective_rows = []
        for top_n in TOP_N_SWEEP:
            objective_rows.append(
                summarize_strategy_variant(
                    output_for_sweep,
                    suffix="ensemble",
                    top_n=top_n,
                    daily_max_exposure=DAILY_MAX_EXPOSURE,
                    objective_type="roi",
                    objective_lambda=0.0,
                )
            )
            for lam in OBJECTIVE_LAMBDAS:
                objective_rows.append(
                    summarize_strategy_variant(
                        output_for_sweep,
                        suffix="ensemble",
                        top_n=top_n,
                        daily_max_exposure=DAILY_MAX_EXPOSURE,
                        objective_type="log_growth_dd",
                        objective_lambda=float(lam),
                    )
                )
        objective_sweep_df = pd.DataFrame(objective_rows)
    else:
        objective_sweep_df = pd.DataFrame()

    output_df = output_df.sort_values("GAME_DATE").reset_index(drop=True)

    rmse_detailed = build_rmse_summary(
        output_df,
        [spec.model_id for spec in model_specs] + (["ensemble"] if "pred_ensemble" in output_df else []),
        home_team_col,
        away_team_col,
    )

    rmse_out = pd.concat([rmse_summary_df.assign(group_type="overall")], axis=0, ignore_index=True)
    rmse_out = pd.concat([rmse_out, rmse_detailed], axis=0, ignore_index=True)

    os.makedirs(os.path.dirname(OUTPUT_PRED_PATH), exist_ok=True)
    output_df.to_csv(OUTPUT_PRED_PATH, index=False)
    rmse_out.to_csv(OUTPUT_RMSE_PATH, index=False)
    audit_df.to_csv(OUTPUT_AUDIT_PATH, index=False)
    objective_sweep_df.to_csv(OUTPUT_OBJECTIVE_SWEEP_PATH, index=False)

    print(f"Saved predictions to {OUTPUT_PRED_PATH}")
    print(f"Saved RMSE summary to {OUTPUT_RMSE_PATH}")
    print(f"Saved odds join audit to {OUTPUT_AUDIT_PATH}")
    print(f"Saved objective sweep to {OUTPUT_OBJECTIVE_SWEEP_PATH}")


if __name__ == "__main__":
    main()
