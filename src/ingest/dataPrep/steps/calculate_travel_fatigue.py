"""Calculate travel distance, fatigue, and rest-related features."""

import os
import pandas as pd
import numpy as np

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

def haversine_np(lat1, lon1, lat2, lon2):
    """Compute great-circle distance between coordinates in miles.

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : array-like
        Latitude/longitude pairs in degrees.

    Returns
    -------
    np.ndarray
        Distance in miles between each coordinate pair.
    """
    R = 3958.8  # miles
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def compute_travel_last_n_days(df, days):
    """Sum travel distance for each team over a lookback window.

    Parameters
    ----------
    df : pd.DataFrame
        Team-game dataframe with columns: team, date, travel_dist.
    days : int
        Lookback window in days.

    Returns
    -------
    np.ndarray
        Total travel distance over the window for each row.
    """
    out = np.zeros(len(df))

    for team, g in df.groupby("team", sort=False):
        dates = g["date"].values
        dists = g["travel_dist"].values
        idxs = g.index.values

        for i in range(len(g)):
            cutoff = dates[i] - np.timedelta64(days, "D")
            mask = (dates < dates[i]) & (dates >= cutoff)
            out[idxs[i]] = dists[mask].sum()

    return out

def run(config=None, inputs=None):
    """Run step 5 to add travel, altitude, and fatigue features.

    Parameters
    ----------
    config : dict, optional
        Pipeline configuration; defaults to `get_config()` when None.
    inputs : dict, optional
        Prior step outputs; uses `df_model` when provided.

    Returns
    -------
    dict
        Dictionary with `df_model_with_travel` dataframe.
    """
    cfg = resolve_step_config(get_config(), "step5") if config is None else resolve_step_config(config, "step5")
    paths = cfg["paths"]
    travel_cfg = cfg["travel"]
    altitude_cfg = cfg["altitude"]
    fatigue_cfg = cfg["fatigue"]
    cache_cfg = cfg.get("cache", {})

    # Try to load from cache
    df_cached, from_cache = load_cached_output(paths["output_model"], cache_cfg)
    if from_cache and not inputs:
        return {"df_model_with_travel": df_cached}

    if inputs and "df_model" in inputs:
        df_games = inputs["df_model"].copy()
    else:
        df_games = pd.read_csv(paths["input_model"])

    arenas = pd.read_csv(paths["arena_locations"])

    df_games = df_games.merge(
        arenas[["team_abbr", "lat", "lon"]],
        left_on="home",
        right_on="team_abbr",
        how="left",
    ).rename(columns={"lat": "arena_lat", "lon": "arena_lon"}).drop(columns="team_abbr")

    home_games = df_games[["date", "GAME_ID", "home", "arena_lat", "arena_lon"]].rename(
        columns={"home": "team"}
    )

    away_games = df_games[["date", "GAME_ID", "away", "arena_lat", "arena_lon"]].rename(
        columns={"away": "team"}
    )

    team_games = pd.concat([home_games, away_games], ignore_index=True)
    team_games["date"] = pd.to_datetime(team_games["date"])
    team_games = team_games.sort_values(["team", "date"]).reset_index(drop=True)

    team_games["prev_lat"] = team_games.groupby("team")["arena_lat"].shift(1)
    team_games["prev_lon"] = team_games.groupby("team")["arena_lon"].shift(1)
    team_games["prev_date"] = team_games.groupby("team")["date"].shift(1)

    team_games["travel_dist"] = haversine_np(
        team_games["prev_lat"],
        team_games["prev_lon"],
        team_games["arena_lat"],
        team_games["arena_lon"],
    )

    team_games["travel_dist"] = team_games["travel_dist"].fillna(0)

    windows = travel_cfg.get("windows", [1, 3, 7, 10])
    for d in windows:
        team_games[f"travel_{d}d"] = compute_travel_last_n_days(team_games, d)

    travel = team_games.copy()

    df_games = df_games.merge(
        travel,
        left_on=["GAME_ID", "home"],
        right_on=["GAME_ID", "team"],
        how="left",
    ).rename(
        columns={
            "travel_1d": "home_travel_1d",
            "travel_3d": "home_travel_3d",
            "travel_7d": "home_travel_7d",
            "travel_10d": "home_travel_10d",
        }
    )

    df_games = df_games.drop(columns=[c for c in ["team", "date"] if c in df_games.columns])

    df_games = df_games.merge(
        travel,
        left_on=["GAME_ID", "away"],
        right_on=["GAME_ID", "team"],
        how="left",
    ).rename(
        columns={
            "travel_1d": "away_travel_1d",
            "travel_3d": "away_travel_3d",
            "travel_7d": "away_travel_7d",
            "travel_10d": "away_travel_10d",
        }
    ).drop(columns="team")

    df_games["travel_diff_3d"] = df_games["home_travel_3d"] - df_games["away_travel_3d"]
    df_games["travel_diff_7d"] = df_games["home_travel_7d"] - df_games["away_travel_7d"]
    df_games["travel_diff_10d"] = df_games["home_travel_10d"] - df_games["away_travel_10d"]

    # z-score against the home team's EARLIER games this season only (mean/std over the whole season would
    # include future games). Fewer than 5 earlier games -> 0.
    date_key = "date_x" if "date_x" in df_games.columns else "date"
    ordered = df_games.sort_values([date_key, "GAME_ID"], kind="stable")
    for d in windows:
        col = f"home_travel_{d}d"
        prior = ordered.groupby(["home", "season"])[col]
        prior_mean = prior.transform(lambda x: x.shift(1).expanding(min_periods=5).mean())
        prior_std = prior.transform(lambda x: x.shift(1).expanding(min_periods=5).std())
        z = ((ordered[col] - prior_mean) / prior_std).replace([float("inf"), float("-inf")], float("nan"))
        df_games[f"travel_{d}d_z"] = z.reindex(df_games.index).fillna(0)

    altitude_teams = altitude_cfg.get("teams", {})
    altitude_threshold = altitude_cfg.get("threshold", 1000)

    df_games["away_altitude"] = df_games["home"].map(altitude_teams).fillna(0)
    df_games["altitude_travel_penalty"] = (
        df_games["away_travel_3d"] * (df_games["away_altitude"] > altitude_threshold).astype(int)
    )

    df_games["away_travel_rest_penalty"] = df_games["away_travel_3d"] / (
        df_games["away_rest"] + 1
    )

    weights = fatigue_cfg.get("weights", {"d1": 0.5, "d3": 0.3, "d7": 0.2})

    df_games["away_fatigue"] = (
        weights.get("d1", 0.5) * df_games["away_travel_1d"]
        + weights.get("d3", 0.3) * df_games["away_travel_3d"]
        + weights.get("d7", 0.2) * df_games["away_travel_7d"]
    )

    df_games["home_fatigue"] = (
        weights.get("d1", 0.5) * df_games["home_travel_1d"]
        + weights.get("d3", 0.3) * df_games["home_travel_3d"]
        + weights.get("d7", 0.2) * df_games["home_travel_7d"]
    )

    df_games["fatigue_diff"] = df_games["away_fatigue"] - df_games["home_fatigue"]

    # Save to cache if enabled
    save_output(df_games, paths["output_model"], cache_cfg)

    return {"df_model_with_travel": df_games}


if __name__ == "__main__":
    run(get_config())