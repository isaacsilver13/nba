"""Create rolling team statistics and moving window features."""

import os
import numpy as np
import pandas as pd

try:
    # When imported as a package module
    from ..config import get_config, resolve_step_config
    from ..helpers import load_cached_output, save_output
except ImportError:
    # When loaded dynamically or as direct import
    try:
        from ingest.dataPrep.config import get_config, resolve_step_config
        from ingest.dataPrep.helpers import load_cached_output, save_output
    except ImportError:
        # Fallback for different import contexts
        import sys
        from pathlib import Path
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from config import get_config, resolve_step_config
        from helpers import load_cached_output, save_output


def _ewm_rolling(series, window, decay):
    """Compute shifted exponentially weighted moving average.

    Parameters
    ----------
    series : pd.Series
        Input series to smooth.
    window : int
        Lookback window size.
    decay : float
        Exponential decay factor (0,1].

    Returns
    -------
    pd.Series
        EWMA series shifted by one to prevent leakage.
    """
    weights = decay ** np.arange(window - 1, -1, -1)

    def _apply(x):
        w = weights[-len(x):]
        return float(np.dot(x, w) / w.sum()) if w.sum() != 0 else 0.0

    return series.shift(1).rolling(window, min_periods=1).apply(_apply, raw=True)


def _kmeans_numpy(X, k=4, seed=42, max_iter=50):
    """Run a simple k-means clustering using NumPy only.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    k : int, default=4
        Number of clusters.
    seed : int, default=42
        RNG seed for centroid initialization.
    max_iter : int, default=50
        Maximum iterations to run.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Cluster labels and centroid matrix.
    """
    rng = np.random.default_rng(seed)
    if len(X) < k:
        labels = np.zeros(len(X), dtype=int)
        return labels, np.mean(X, axis=0, keepdims=True)

    centroids = X[rng.choice(len(X), k, replace=False)]
    for _ in range(max_iter):
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if (labels == i).any() else centroids[i]
            for i in range(k)
        ])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return labels, centroids


def _map_style_labels(centroids, feature_cols):
    """Map centroid profiles to human-readable play style labels.

    Parameters
    ----------
    centroids : np.ndarray
        Cluster centroids.
    feature_cols : list[str]
        Column names aligned to centroid dimensions.

    Returns
    -------
    tuple[dict, dict]
        Label mapping by centroid index and numeric id mapping.
    """
    centers = pd.DataFrame(centroids, columns=feature_cols)
    used = set()
    label_map = {}

    fast_idx = centers["pace"].idxmax()
    label_map[fast_idx] = "Fast offense"
    used.add(fast_idx)

    three_idx = centers.loc[~centers.index.isin(used), "three_pa_rate"].idxmax()
    label_map[three_idx] = "3pt heavy"
    used.add(three_idx)

    paint_idx = centers.loc[~centers.index.isin(used), "paint_points"].idxmax()
    label_map[paint_idx] = "Paint dominant"
    used.add(paint_idx)

    remaining = [i for i in centers.index if i not in used]
    for idx in remaining:
        label_map[idx] = "Slow defensive"

    style_ids = {"Fast offense": 1, "Slow defensive": 2, "3pt heavy": 3, "Paint dominant": 4}
    id_map = {idx: style_ids[label] for idx, label in label_map.items()}
    return label_map, id_map


def _prior_style_map(df, date_col, style_cols, min_games=5, min_teams=8):
    """Team style ids per (season, month) using only games played BEFORE that month.

    A team with fewer than `min_games` earlier games this season falls back to its previous-season averages; with
    no usable history it gets no row (style id 0 after the merge). Clusters are refit each month, so a game's style
    never depends on its own or any later result.
    """
    d = df[["team", "season", date_col] + style_cols].copy()
    d["month"] = d[date_col].dt.to_period("M")
    season_avgs = {s: g.groupby("team")[style_cols].mean() for s, g in d.groupby("season")}
    frames = []
    for season, gs in d.groupby("season", sort=True):
        prev = season_avgs.get(season - 1)
        for month in sorted(gs["month"].dropna().unique()):
            past = gs[gs[date_col] < month.to_timestamp()]
            counts = past.groupby("team").size()
            cur = past.groupby("team")[style_cols].mean()
            rows = {}
            for team in gs["team"].unique():
                if counts.get(team, 0) >= min_games:
                    rows[team] = cur.loc[team]
                elif prev is not None and team in prev.index:
                    rows[team] = prev.loc[team]
            if len(rows) < min_teams:
                continue
            avg = pd.DataFrame(rows).T
            X = avg[style_cols].values
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
            labels, centroids = _kmeans_numpy(X, k=4, seed=42)
            _, id_map = _map_style_labels(centroids, style_cols)
            frames.append(pd.DataFrame({
                "season": season, "month": month, "team": avg.index,
                "style_id": [id_map.get(l, 2) for l in labels],
            }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["season", "month", "team", "style_id"]
    )


def run(config=None, inputs=None):
    """Run step 6 to compute rolling, EWMA, and style features.

    Parameters
    ----------
    config : dict, optional
        Pipeline configuration; defaults to `get_config()` when None.
    inputs : dict, optional
        Prior step outputs; uses `team_opp_stats` and `df_model_with_travel` when provided.

    Returns
    -------
    dict
        Dictionary with `df_model_2` dataframe.
    """
    cfg = resolve_step_config(get_config(), "step6") if config is None else resolve_step_config(config, "step6")
    paths = cfg["paths"]
    rolling_cfg = cfg["rolling"]
    stats_cfg = cfg["stats"]
    ewm_cfg = cfg.get("ewm", {})
    s2d_cfg = cfg.get("season_to_date", {})
    cache_cfg = cfg.get("cache", {})

    output_path = paths["output_model"]

    # Try to load from cache
    df_model_cached, from_cache = load_cached_output(output_path, cache_cfg)
    if from_cache and not inputs:
        print(f"[step6] Cache hit: loaded features from {output_path}")
        return {"df_model_2": df_model_cached}

    if inputs and "team_opp_stats" in inputs:
        df = inputs["team_opp_stats"].copy()
    else:
        df = pd.read_csv(paths["input_team_opp"], dtype={"GAME_ID": str})

    if inputs and "df_model_with_travel" in inputs:
        df_model = inputs["df_model_with_travel"].copy()
    else:
        df_model = pd.read_csv(paths["input_model"], dtype={"GAME_ID": str})

    print(f"[step6] team_opp_stats rows: {len(df):,}")
    print(f"[step6] df_model_with_travel rows: {len(df_model):,}")

    # Ensure GAME_ID is string for consistent merging
    # one canonical 10-digit id on both sides (a CSV round trip can drop the leading zeros)
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.replace(".0", "", regex=False).str.zfill(10)
    df_model["GAME_ID"] = df_model["GAME_ID"].astype(str).str.replace(".0", "", regex=False).str.zfill(10)

    merge_cols = ["GAME_ID", "home", "away", "season"]
    date_col = "date_x" if "date_x" in df_model.columns else ("date" if "date" in df_model.columns else None)
    if date_col and date_col not in merge_cols:
        merge_cols.append(date_col)

    df = df.merge(df_model[merge_cols], on="GAME_ID", how="left")
    df["is_home"] = (df["team"] == df["home"]).astype(int)

    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(["team", date_col, "GAME_ID"]).reset_index(drop=True)
    else:
        df = df.sort_values(["team", "GAME_ID"]).reset_index(drop=True)

    roll_columns = stats_cfg["roll_columns"]
    roll_windows = rolling_cfg.get("windows", [5, 10])
    min_games = rolling_cfg.get("min_games", 3)

    # Rolling stats
    roll_features = {}
    for w in roll_windows:
        window_min = min(min_games, w)
        for stat in roll_columns:
            if stat in df.columns:
                roll_features[f"team_{stat}_r{w}"] = (
                    df.groupby("team")[stat]
                    .transform(lambda x: x.shift(1).rolling(w, min_periods=window_min).mean())
                    .fillna(0)
                )

    if roll_features:
        df = pd.concat([df, pd.DataFrame(roll_features, index=df.index)], axis=1)
    print(f"[step6] Added rolling features: {len(roll_features):,}")

    # EWMA stats
    ewm_cols = stats_cfg.get("ewm_columns", [])
    decay = ewm_cfg.get("decay", 0.8)
    ewm_windows = ewm_cfg.get("windows", [5, 10])

    ewm_features = {}
    for w in ewm_windows:
        for col in ewm_cols:
            if col in df.columns:
                ewm_features[f"team_{col}_ewm{w}"] = (
                    df.groupby("team")[col]
                    .transform(lambda x: _ewm_rolling(x, w, decay))
                    .fillna(0)
                )

    if ewm_features:
        df = pd.concat([df, pd.DataFrame(ewm_features, index=df.index)], axis=1)
    print(f"[step6] Added EWMA features: {len(ewm_features):,}")

    # Season-to-date priors (strictly lagged)
    if s2d_cfg.get("enabled", True):
        s2d_min_games = s2d_cfg.get("min_games", min_games)
        s2d_features = {}
        for stat in roll_columns:
            if stat in df.columns and "season" in df.columns:
                s2d_features[f"team_{stat}_s2d"] = (
                    df.groupby(["team", "season"])[stat]
                    .transform(lambda x: x.shift(1).expanding(min_periods=s2d_min_games).mean())
                    .fillna(0)
                )

        if s2d_features:
            df = pd.concat([df, pd.DataFrame(s2d_features, index=df.index)], axis=1)
        print(f"[step6] Added season-to-date features: {len(s2d_features):,}")

    # Team style clustering (refit monthly on earlier games only; see _prior_style_map)
    style_cols = ["pace", "three_pa_rate", "paint_points", "rebound_rate", "net_rebounds", "net_three_pa"]
    style_map = _prior_style_map(df, date_col, style_cols)

    df_home = df[df["is_home"] == 1].copy()
    df_away = df[df["is_home"] == 0].copy()

    home_cols = {
        col: f"home_{col}" for col in df_home.columns if col not in ["GAME_ID", "team", "is_home"]
    }

    away_cols = {
        col: f"away_{col}" for col in df_away.columns if col not in ["GAME_ID", "team", "is_home"]
    }

    df_home = df_home.rename(columns=home_cols)
    df_away = df_away.rename(columns=away_cols)

    df_game_from_df = df_home.merge(df_away, on="GAME_ID", how="inner")

    df_model = df_model.merge(df_game_from_df, on="GAME_ID", how="left")

    df_model["_month"] = pd.to_datetime(df_model[date_col]).dt.to_period("M")
    df_model = df_model.merge(
        style_map.rename(columns={"team": "home", "style_id": "home_style_id", "month": "_month"}),
        on=["season", "_month", "home"],
        how="left",
    )
    df_model = df_model.merge(
        style_map.rename(columns={"team": "away", "style_id": "away_style_id", "month": "_month"}),
        on=["season", "_month", "away"],
        how="left",
    ).drop(columns="_month")

    df_model["home_style_id"] = pd.to_numeric(df_model["home_style_id"], errors="coerce").fillna(0).astype(int)
    df_model["away_style_id"] = pd.to_numeric(df_model["away_style_id"], errors="coerce").fillna(0).astype(int)
    df_model["team_style_vs_opponent_style"] = df_model["home_style_id"] * 10 + df_model["away_style_id"]

    # Ratings and pace features (diffs)
    df_model["off_rating_last_5"] = df_model.get("home_team_off_rating_r5", 0) - df_model.get("away_team_off_rating_r5", 0)
    df_model["def_rating_last_5"] = df_model.get("home_team_def_rating_r5", 0) - df_model.get("away_team_def_rating_r5", 0)
    df_model["net_rating_last_5"] = df_model.get("home_team_net_rating_r5", 0) - df_model.get("away_team_net_rating_r5", 0)

    # windows are 5/10/20 (there is no r15, which silently made this feature a duplicate of net_rating_last_5)
    home_net_trend = df_model.get("home_team_net_rating_r5", 0) - df_model.get("home_team_net_rating_r20", 0)
    away_net_trend = df_model.get("away_team_net_rating_r5", 0) - df_model.get("away_team_net_rating_r20", 0)
    df_model["net_rating_trend"] = home_net_trend - away_net_trend

    home_pace = df_model.get("home_team_pace_r5", 0)
    away_pace = df_model.get("away_team_pace_r5", 0)
    df_model["pace_diff"] = home_pace - away_pace
    df_model["combined_pace"] = home_pace + away_pace
    df_model["pace_interaction"] = home_pace * away_pace

    # 10-game net rating gap (net_rating_last_5 is the 5-game one; they used to be identical columns)
    df_model["net_rating_diff"] = df_model.get("home_team_net_rating_r10", 0) - df_model.get("away_team_net_rating_r10", 0)

    df_model["rest_diff_elo_diff"] = df_model.get("rest_diff", 0) * df_model.get("elo_diff", 0)
    df_model["home_elo_roll_2Y_rest_advantage"] = df_model.get("home_elo_roll_2Y", 0) * df_model.get("rest_advantage", 0)
    df_model["pace_diff_rest_diff"] = df_model.get("pace_diff", 0) * df_model.get("rest_diff", 0)
    df_model["net_rating_diff_rest_diff"] = df_model.get("net_rating_diff", 0) * df_model.get("rest_diff", 0)

    # (the edge / edge-rank / model-error columns that used to be built here needed a `model_margin` column that
    # no step produces, so they were constants; removed)
    if date_col:
        df_model = df_model.sort_values(date_col).reset_index(drop=True)

    # Save to cache if enabled
    save_output(df_model, output_path, cache_cfg)
    print(f"[step6] Saved model features to {output_path} | rows={len(df_model):,}, cols={df_model.shape[1]:,}")

    return {"df_model_2": df_model}


if __name__ == "__main__":
    run(get_config())
