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


OUTPUT_PRED_PATH = "outputs/copilot_model_output.csv"
OUTPUT_RMSE_PATH = "outputs/copilot_model_rmse.csv"

EDGE_STANDARDIZATION_MODE = "standardized"  # "standardized" or "raw"

TRAIN_SIZE = 2000
TEST_SIZE = 300

PAYOUT = 0.909
EV_THRESHOLD = 0.02
FRACTIONAL_KELLY = 0.5
MAX_KELLY = 0.1

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


def find_team_columns(df: pd.DataFrame) -> Tuple[str, str]:
    home_col = next((c for c in TEAM_HOME_CANDIDATES if c in df.columns), None)
    away_col = next((c for c in TEAM_AWAY_CANDIDATES if c in df.columns), None)
    if not home_col or not away_col:
        raise ValueError(
            "Home/away team identifiers were not found. Expected columns like "
            "'home'/'away' or 'home_team'/'away_team'."
        )
    return home_col, away_col


def load_feature_table() -> pd.DataFrame:
    path = get_step6a_output_path()
    df = pd.read_csv(path, parse_dates=["date_x"])
    if "date_x" in df.columns:
        df = df.rename(columns={"date_x": "GAME_DATE"})

    games_path = os.path.join("data", "processed", "nba_games_with_game_id_processed.csv")
    if os.path.exists(games_path) and "GAME_ID" in df.columns:
        games = pd.read_csv(games_path, usecols=["GAME_ID", "home", "away"])
        df = df.merge(games, on="GAME_ID", how="left", validate="m:1")

    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    return df


def build_feature_sets(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    columns_keep = [c for c in get_step6a_columns() if c in df.columns]

    identifiers = [c for c in ["season", "GAME_ID", "GAME_DATE"] if c in df.columns]
    targets = [c for c in ["home_margin", "away_margin"] if c in df.columns]

    leakage = [
        c for c in columns_keep
        if "model_error" in c.lower() or "box" in c.lower()
    ]

    market = ["spread_signed"]
    if "is_home_favorite" in df.columns:
        market.append("is_home_favorite")

    features = [
        c for c in columns_keep
        if c not in identifiers
        and c not in targets
        and c not in leakage
        and c != "spread"
    ]

    if "spread_signed" in df.columns and "spread_signed" not in features:
        features.append("spread_signed")

    features = sorted(set(features))
    edge_features = [c for c in features if "edge" in c.lower()]

    return features, edge_features, identifiers


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
        specs.append(
            ModelSpec(
                model_id=f"xgboost_v{idx}",
                model=XGBRegressor(**params),
                needs_imputer=False,
            )
        )

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
        specs.append(
            ModelSpec(
                model_id=f"lightgbm_v{idx}",
                model=lgb.LGBMRegressor(**params),
                needs_imputer=False,
            )
        )

    specs.append(
        ModelSpec(
            model_id="linear_regression",
            model=LinearRegression(),
            needs_imputer=True,
        )
    )

    specs.append(
        ModelSpec(
            model_id="elasticnet",
            model=ElasticNet(alpha=0.5, l1_ratio=0.4, random_state=42, max_iter=5000),
            needs_imputer=True,
        )
    )

    specs.append(
        ModelSpec(
            model_id="random_forest",
            model=RandomForestRegressor(
                n_estimators=600,
                max_depth=12,
                min_samples_leaf=5,
                random_state=42,
            ),
            needs_imputer=True,
        )
    )

    specs.append(
        ModelSpec(
            model_id="gbr",
            model=GradientBoostingRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            ),
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
    bet_win_col = f"bet_win_{suffix}"
    pnl_col = f"pnl_kelly_{suffix}"
    cum_pnl_col = f"cum_pnl_{suffix}"

    new_cols = {}

    edge = df[pred_col] - df["spread_signed"]
    edge_std = edge / df[sigma_col]
    win_prob = norm.cdf(edge_std)

    new_cols[edge_col] = edge
    new_cols[edge_std_col] = edge_std
    new_cols[win_prob_col] = win_prob

    ev_home = win_prob * PAYOUT - (1 - win_prob)
    ev_away = (1 - win_prob) * PAYOUT - win_prob
    new_cols[ev_home_col] = ev_home
    new_cols[ev_away_col] = ev_away

    bet_side = np.where(
        ev_home > EV_THRESHOLD,
        "HOME",
        np.where(ev_away > EV_THRESHOLD, "AWAY", "NO BET"),
    )
    new_cols[bet_side_col] = bet_side

    kelly = np.zeros(len(df))
    home = bet_side == "HOME"
    away = bet_side == "AWAY"
    kelly[home] = ((win_prob[home] * (PAYOUT + 1) - 1) / PAYOUT)
    kelly[away] = (((1 - win_prob[away]) * (PAYOUT + 1) - 1) / PAYOUT)
    kelly = np.clip(kelly, 0, 1)
    kelly = np.minimum(kelly * FRACTIONAL_KELLY, MAX_KELLY)
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
    pnl[bet_win == 1] = kelly[bet_win == 1] * PAYOUT
    pnl[bet_win == 0] = -kelly[bet_win == 0]
    new_cols[pnl_col] = pnl
    new_cols[cum_pnl_col] = pd.Series(pnl, index=df.index).fillna(0).cumsum()

    new_df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return new_df


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
    df = load_feature_table()
    features, edge_features, identifiers = build_feature_sets(df)

    home_team_col, away_team_col = find_team_columns(df)

    target = "home_margin"
    time_col = "GAME_DATE"

    numeric_cols = [c for c in features + [target, "spread_signed"] if c in df.columns]
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

    for fold, (train_df, test_df) in enumerate(
        walk_forward_splits(df, time_col, train_size, test_size), start=1
    ):
        print(f"Fold {fold}: Train {len(train_df)} rows, Test {len(test_df)} rows")

        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]

        non_all_nan = X_train.columns[X_train.notna().any()].tolist()
        X_train = X_train[non_all_nan]
        X_test = X_test[non_all_nan]
        edge_features_fold = [c for c in edge_features if c in non_all_nan]

        X_train, X_test, _ = standardize_edge_cols(
            X_train, X_test, edge_features_fold, EDGE_STANDARDIZATION_MODE
        )

        for spec in model_specs:
            X_train_fold = X_train
            X_test_fold = X_test

            if spec.needs_imputer:
                imputer = SimpleImputer(strategy="median")
                X_train_fold = pd.DataFrame(
                    imputer.fit_transform(X_train_fold),
                    columns=non_all_nan,
                    index=X_train.index,
                )
                X_test_fold = pd.DataFrame(
                    imputer.transform(X_test_fold),
                    columns=non_all_nan,
                    index=X_test.index,
                )

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

    output_df = output_df[identifiers + ["home_margin", "spread_signed", home_team_col, away_team_col]]
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

    print(f"Saved predictions to {OUTPUT_PRED_PATH}")
    print(f"Saved RMSE summary to {OUTPUT_RMSE_PATH}")


if __name__ == "__main__":
    main()
