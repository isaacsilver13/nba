"""Engineer advanced matchup, injury, and style features."""

import os
import pandas as pd
import numpy as np

try:
    # When imported as a package module
    from ..config import get_config, resolve_step_config
    from ..helpers import dedupe_player_games, load_cached_output, save_output
except ImportError:
    # When loaded dynamically or as direct import
    try:
        from ingest.dataPrep.config import get_config, resolve_step_config
        from ingest.dataPrep.helpers import dedupe_player_games, load_cached_output, save_output
    except ImportError:
        # Fallback for different import contexts
        import sys
        from pathlib import Path
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from config import get_config, resolve_step_config
        from helpers import dedupe_player_games, load_cached_output, save_output


def clean_player_data(df_players, df_games):
    """Clean player box scores and merge game metadata.

    Parameters
    ----------
    df_players : pd.DataFrame
        Raw player-level boxscore data.
    df_games : pd.DataFrame
        Game-level data containing schedule and team identifiers.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Cleaned player dataframe and the (possibly adjusted) games dataframe.
    """
    df_games = df_games.copy()
    df_players = df_players.copy()

    # Normalize player schema across V2/V3 payloads
    alias_map = {
        "GAME_ID": ["GAME_ID", "gameId"],
        "TEAM_ID": ["TEAM_ID", "teamId", "team_id"],
        "PLAYER_ID": ["PLAYER_ID", "playerId", "personId", "person_id"],
        "team": ["team", "TEAM_ABBREVIATION", "teamTricode", "team_tricode"],
        "points": ["points", "PTS"],
        "rebounds": ["rebounds", "REB", "reboundsTotal"],
        "assists": ["assists", "AST"],
        "STL": ["STL", "steals"],
        "BLK": ["BLK", "blocks"],
        "turnovers": ["turnovers", "TO"],
        "PF": ["PF", "foulsPersonal"],
        "PLUS_MINUS": ["PLUS_MINUS", "plusMinusPoints"],
        "fgm": ["fgm", "FGM", "fieldGoalsMade"],
        "fga": ["fga", "FGA", "fieldGoalsAttempted"],
        "three_pm": ["three_pm", "FG3M", "threePointersMade"],
        "three_pa": ["three_pa", "FG3A", "threePointersAttempted"],
        "ftm": ["ftm", "FTM", "freeThrowsMade"],
        "fta": ["fta", "FTA", "freeThrowsAttempted"],
        "OREB": ["OREB", "reboundsOffensive"],
        "DREB": ["DREB", "reboundsDefensive"],
    }

    for target, aliases in alias_map.items():
        if target in df_players.columns:
            continue
        src = next((c for c in aliases if c in df_players.columns), None)
        if src is not None:
            df_players[target] = df_players[src]

    if "GAME_ID" in df_players.columns:
        df_players["GAME_ID"] = df_players["GAME_ID"].astype(str).str.replace(".0", "", regex=False).str.zfill(10)
    if "team" in df_players.columns:
        df_players["team"] = df_players["team"].astype(str).str.upper().str.strip()

    df_players = dedupe_player_games(df_players)  # duplicated rows make a game's own stats part of its "previous games"

    # Ensure expected numeric columns exist and are numeric
    required_numeric = [
        "points", "rebounds", "assists", "STL", "BLK", "turnovers", "PF", "PLUS_MINUS",
        "fgm", "fga", "three_pm", "three_pa", "ftm", "fta", "OREB", "DREB",
    ]
    for col in required_numeric:
        if col not in df_players.columns:
            df_players[col] = 0
        df_players[col] = pd.to_numeric(df_players[col], errors="coerce").fillna(0)

    df_games["GAME_ID"] = df_games["GAME_ID"].astype(str).str.zfill(10)

    if "COMMENT" in df_players.columns:
        raw_status = df_players["COMMENT"]
    elif "comment" in df_players.columns:
        raw_status = df_players["comment"]
    else:
        raw_status = pd.Series("", index=df_players.index)

    df_players["status"] = (
        raw_status.fillna("")
        .astype(str)
        .str.upper()
        .str.replace("_", " ", regex=False)
        .str.replace("-", " ", regex=False)
        .str.strip()
    )

    if "minutes" in df_players.columns:
        raw_minutes = df_players["minutes"]
    elif "MIN" in df_players.columns:
        raw_minutes = df_players["MIN"]
    else:
        raw_minutes = pd.Series("", index=df_players.index)

    minutes_parts = raw_minutes.fillna("").astype(str).str.split(":", n=1, expand=True)
    if minutes_parts.shape[1] == 2:
        mins = pd.to_numeric(minutes_parts[0], errors="coerce").fillna(0)
        secs = pd.to_numeric(minutes_parts[1], errors="coerce").fillna(0)
        df_players["minutes_played"] = mins + secs / 60.0
    else:
        df_players["minutes_played"] = 0.0
    df_players["did_play"] = df_players["minutes_played"] > 0
    df_players["GAME_ID"] = df_players["GAME_ID"].astype(str)

    df_players["GAME_ID"] = (
        df_players["GAME_ID"].astype(str).str.replace(".0", "", regex=False).str.zfill(10)
    )

    df_players["coach_decision_absence"] = df_players["status"].str.contains("COACH", regex=True)

    # Known before tip-off: official inactives (DND/NWT) and declared injury/illness/rest/personal absences.
    # "DNP - Coach's Decision" (and a blank reason) is decided or unknown at game time, so it is never pregame.
    df_players["pregame_out"] = (
        (~df_players["did_play"]) & (df_players["status"] != "") & (~df_players["coach_decision_absence"])
    )

    df_players = df_players.merge(
        df_games[["GAME_ID", "date", "home", "away", "season"]],
        on="GAME_ID",
        how="left",
    )

    df_players = df_players.sort_values(["PLAYER_ID", "date"])

    cols_to_fill = [
        "fgm",
        "fga",
        "three_pm",
        "three_pa",
        "ftm",
        "fta",
        "OREB",
        "DREB",
        "rebounds",
        "assists",
        "STL",
        "BLK",
        "turnovers",
        "PF",
        "points",
        "PLUS_MINUS",
    ]

    existing_cols = [c for c in cols_to_fill if c in df_players.columns]
    df_players[existing_cols] = df_players[existing_cols].fillna(0)

    df_players["usage_proxy"] = (
        df_players["fga"] + 0.44 * df_players["fta"] + df_players["turnovers"]
    ) / df_players["minutes_played"]

    return df_players, df_games


def run(config=None, inputs=None):
    """Run step 4 to engineer advanced team and player features.

    Parameters
    ----------
    config : dict, optional
        Pipeline configuration; defaults to `get_config()` when None.
    inputs : dict, optional
        Prior step outputs; uses `games_processed` and `player_boxscores` when provided.

    Returns
    -------
    dict
        Dictionary with `df_model` and `team_game` dataframes.
    """
    cfg = resolve_step_config(get_config(), "step4") if config is None else resolve_step_config(config, "step4")
    paths = cfg["paths"]
    rolling_cfg = cfg["rolling"]
    cache_cfg = cfg.get("cache", {})

    output_team_game = paths["output_team_game"]
    output_df_model = paths["output_df_model"]

    # Try to load both from cache
    df_model_cached, model_cached = load_cached_output(output_df_model, cache_cfg)
    team_game_cached, team_game_from_cache = load_cached_output(output_team_game, cache_cfg)

    if model_cached and team_game_from_cache and not inputs:
        # Parse dates if present
        if "date" in df_model_cached.columns:
            df_model_cached["date"] = pd.to_datetime(df_model_cached["date"])
        if "date" in team_game_cached.columns:
            team_game_cached["date"] = pd.to_datetime(team_game_cached["date"])
        return {"df_model": df_model_cached, "team_game": team_game_cached}

    if inputs and "games_processed" in inputs:
        df_games = inputs["games_processed"].copy()
    else:
        df_games = pd.read_csv(paths["input_games"], dtype={"GAME_ID": "string"}, parse_dates=["date"])

    if inputs and "player_boxscores" in inputs:
        df_players_raw = inputs["player_boxscores"].copy()
    else:
        df_players_raw = pd.read_csv(paths["input_players"])

    df_players, df_games = clean_player_data(df_players_raw, df_games)

    roll_windows = rolling_cfg["windows"]
    min_games = rolling_cfg.get("min_games", 3)

    # lagged per-player form: only these three feed the features below
    for w in roll_windows:
        for stat in ["points", "usage_proxy", "minutes_played"]:
            df_players[f"player_{stat}_r{w}"] = (
                df_players.groupby("PLAYER_ID")[stat]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=min_games).mean())
            ).fillna(0)

    df_players["usage_rank_game"] = df_players.groupby(["GAME_ID", "TEAM_ID"])[
        "player_usage_proxy_r5"
    ].rank(method="first", ascending=False)

    # Only pregame-known absences; player_*_r5 are lagged, usage_rank_game ranks lagged usage.
    df_players["pregame_missing_top1"] = (
        (df_players["usage_rank_game"] == 1) & df_players["pregame_out"]
    ).astype(int)
    df_players["pregame_missing_top2"] = (
        (df_players["usage_rank_game"] == 2) & df_players["pregame_out"]
    ).astype(int)

    df_players["pregame_minutes_out"] = np.where(
        df_players["pregame_out"], df_players["player_minutes_played_r5"], 0
    )
    df_players["pregame_usage_out"] = np.where(
        df_players["pregame_out"], df_players["player_usage_proxy_r5"], 0
    )

    # Games backfilled from the box-score-only CSV have no DNP/inactive rows: absences are UNKNOWN there, not zero.
    if "endpoint_family" in df_players.columns:
        df_players["absence_unknown"] = df_players["endpoint_family"].eq("csv_backfill")
    else:
        df_players["absence_unknown"] = False

    missing_flags = (
        df_players.groupby(["GAME_ID", "TEAM_ID", "team"], as_index=False)
        .agg(
            date=("date", "first"),
            absence_unknown=("absence_unknown", "max"),
            pregame_missing_top1=("pregame_missing_top1", "max"),
            pregame_missing_top2=("pregame_missing_top2", "max"),
            pregame_minutes_out=("pregame_minutes_out", "sum"),
            pregame_usage_out=("pregame_usage_out", "sum"),
        )
        .sort_values(["TEAM_ID", "date", "GAME_ID"])
    )
    # what the team was missing going INTO its previous game (a true "last game" version)
    missing_flags["prev_game_minutes_out"] = (
        missing_flags.groupby("TEAM_ID")["pregame_minutes_out"].shift(1).fillna(0)
    )
    prev_unknown = missing_flags.groupby("TEAM_ID")["absence_unknown"].shift(1).fillna(False).astype(bool)
    missing_flags.loc[prev_unknown, "prev_game_minutes_out"] = np.nan
    missing_flags = missing_flags.drop(columns="date")
    MISSING_COLS = [
        "pregame_missing_top1", "pregame_missing_top2",
        "pregame_minutes_out", "pregame_usage_out", "prev_game_minutes_out",
    ]

    counting_cols = [
        "fgm",
        "fga",
        "three_pm",
        "three_pa",
        "ftm",
        "fta",
        "OREB",
        "DREB",
        "rebounds",
        "assists",
        "turnovers",
        "points",
        "minutes_played",
    ]

    team_game = (
        df_players[df_players["did_play"]]
        .groupby(["GAME_ID", "date", "TEAM_ID", "team"], as_index=False)[counting_cols]
        .sum()
    )

    team_game["three_rate"] = team_game["three_pa"] / (
        team_game["three_pa"] + team_game["fga"] + 1e-6
    )

    team_game["ft_rate"] = team_game["fta"] / (team_game["fga"] + 1e-6)

    team_game["ast_to_to"] = team_game["assists"] / (team_game["turnovers"] + 1e-6)

    # overtime games have ~265 team minutes; flag whether each team's PREVIOUS game went to overtime
    team_game = team_game.sort_values(["TEAM_ID", "date", "GAME_ID"])
    team_game["ot_game"] = (team_game["minutes_played"] > 245).astype(int)
    team_game["after_ot"] = team_game.groupby("TEAM_ID")["ot_game"].shift(1).fillna(0)

    team_rolls = rolling_cfg.get("team_rolls", [5, 10])

    team_game = team_game.sort_values(["TEAM_ID", "date"])

    for r in team_rolls:
        for col in ["points", "rebounds", "assists", "three_rate", "ft_rate", "ast_to_to"]:
            team_game[f"{col}_r{r}"] = (
                team_game.groupby("TEAM_ID")[col].transform(
                    lambda x: x.shift(1).rolling(r, min_periods=3).mean()
                )
            )

    team_game["team_poss"] = (
        team_game["fga"] + 0.44 * team_game["fta"] - team_game["OREB"] + team_game["turnovers"]
    ).clip(lower=1)

    bench_top_n = rolling_cfg.get("bench_top_n", 5)
    df_players["minutes_rank_game"] = df_players.groupby(["GAME_ID", "TEAM_ID"])[
        "minutes_played"
    ].rank(method="first", ascending=False)

    bench_plus_minus = (
        df_players[(df_players["did_play"]) & (df_players["minutes_rank_game"] > bench_top_n)]
        .groupby(["GAME_ID", "TEAM_ID"], as_index=False)["PLUS_MINUS"]
        .sum()
        .rename(columns={"PLUS_MINUS": "bench_plus_minus"})
    )

    team_game = team_game.merge(bench_plus_minus, on=["GAME_ID", "TEAM_ID"], how="left")
    team_game["bench_plus_minus"] = team_game["bench_plus_minus"].fillna(0)
    team_game["bench_net_rating"] = (
        team_game["bench_plus_minus"] / team_game["team_poss"] * 100
    ).fillna(0)

    team_game["bench_net_rating_last_5"] = (
        team_game.groupby("TEAM_ID")["bench_net_rating"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=3).mean())
    )

    top_n = rolling_cfg.get("top_n_players", 8)

    team_features = (
        df_players[~df_players["pregame_out"]]
        .sort_values(["player_minutes_played_r5", "PLAYER_ID"], ascending=[False, True])
        .groupby(["GAME_ID", "team"])
        .head(top_n)
        .groupby(["GAME_ID", "team"], as_index=False)["player_points_r5"]
        .mean()
        .rename(columns={"player_points_r5": f"top{top_n}_points_r5"})
    )

    home_feats = team_features.rename(
        columns={"team": "home", f"top{top_n}_points_r5": f"home_top{top_n}_points_r5"}
    )

    away_feats = team_features.rename(
        columns={"team": "away", f"top{top_n}_points_r5": f"away_top{top_n}_points_r5"}
    )

    df_model = df_games.merge(home_feats, on=["GAME_ID", "home"])
    df_model = df_model.merge(away_feats, on=["GAME_ID", "away"])

    df_model = df_model.dropna(
        subset=[f"home_top{top_n}_points_r5", f"away_top{top_n}_points_r5"]
    )

    home_missing = missing_flags.rename(columns={"team": "home", "absence_unknown": "home_absence_unknown",
                                                 **{c: f"home_{c}" for c in MISSING_COLS}})
    away_missing = missing_flags.rename(columns={"team": "away", "absence_unknown": "away_absence_unknown",
                                                 **{c: f"away_{c}" for c in MISSING_COLS}})

    df_model = df_model.merge(home_missing.drop(columns="TEAM_ID"), on=["GAME_ID", "home"], how="left")
    df_model = df_model.merge(away_missing.drop(columns="TEAM_ID"), on=["GAME_ID", "away"], how="left")

    bench_features = team_game[["GAME_ID", "team", "bench_net_rating_last_5"]].copy()
    home_bench = bench_features.rename(columns={
        "team": "home",
        "bench_net_rating_last_5": "home_bench_net_rating_last_5",
    })
    away_bench = bench_features.rename(columns={
        "team": "away",
        "bench_net_rating_last_5": "away_bench_net_rating_last_5",
    })

    df_model = df_model.merge(home_bench, on=["GAME_ID", "home"], how="left")
    df_model = df_model.merge(away_bench, on=["GAME_ID", "away"], how="left")

    unknown = df_model["home_absence_unknown"].fillna(False).astype(bool) | df_model["away_absence_unknown"].fillna(False).astype(bool)
    for c in MISSING_COLS:
        df_model[c] = df_model[f"home_{c}"].fillna(0) - df_model[f"away_{c}"].fillna(0)
        df_model.loc[unknown, c] = np.nan  # backfilled game: absences unknown, not "none"
    # the game after a backfilled one: its "previous game" absences are unknown too
    prev_nan = df_model["home_prev_game_minutes_out"].isna() | df_model["away_prev_game_minutes_out"].isna()
    df_model.loc[prev_nan, "prev_game_minutes_out"] = np.nan
    ot = team_game[["GAME_ID", "team", "after_ot"]]
    df_model = df_model.merge(ot.rename(columns={"team": "home", "after_ot": "home_after_ot"}), on=["GAME_ID", "home"], how="left")
    df_model = df_model.merge(ot.rename(columns={"team": "away", "after_ot": "away_after_ot"}), on=["GAME_ID", "away"], how="left")
    df_model["after_ot_flag"] = df_model["home_after_ot"].fillna(0) - df_model["away_after_ot"].fillna(0)

    df_model["bench_net_rating_last_5"] = df_model["home_bench_net_rating_last_5"] - df_model["away_bench_net_rating_last_5"]

    df_model["top8_points_diff"] = (
        df_model[f"home_top{top_n}_points_r5"] - df_model[f"away_top{top_n}_points_r5"]
    )

    # Save to cache if enabled
    save_output(team_game, output_team_game, cache_cfg)
    save_output(df_model, output_df_model, cache_cfg)

    return {"team_game": team_game, "df_model": df_model}


if __name__ == "__main__":
    run(get_config())

