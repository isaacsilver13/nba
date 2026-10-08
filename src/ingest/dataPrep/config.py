"""Configuration and defaults for the data preparation pipeline."""

from copy import deepcopy

DEFAULT_CONFIG = {
    "cache": {
        "enabled": True,
        "read_on_hit": True,
        "write_outputs": True,
    },
    "steps": {
        "step1": {
            "paths": {
                "input_file": "data/raw/kaggle_nba_2008-2025.csv",
                "output_file": "data/processed/nba_games_with_game_id.csv",
                "processed_fallback": "data/processed/nba_games_with_game_id_processed.csv",
                "error_log": "data/logs/scoreboard_failures.csv",
                "retry_output": "data/processed/scoreboard_retry_games.csv",
            },
            "filters": {
                "regular_only": True,
                "min_season": 2020,
                "max_season": 2025,
            },
            "api": {
                "request_sleep": 0.25,
                "timeout": 40,
                "max_retries": 5,
                "headers": {
                    "Host": "stats.nba.com",
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "application/json, text/plain, */*",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://www.nba.com/",
                    "Connection": "keep-alive",
                },
            },
        },
        "step2": {
            "paths": {
                "input_file": "data/processed/nba_games_with_game_id.csv",
                "output_file": "data/processed/nba_games_with_game_id_processed.csv",
            },
            "elo": {
                "start": 1500,
                "k": 20,
                "home_advantage": 65,
            },
            "rolling": {
                "window": 5,
                "margin_short": 5,
                "margin_long": 10,
                "margin_short_min_periods": 3,
                "margin_long_min_periods": 5,
                "elo_rolling_window": 164,
            },
        },
        "step3": {
            "paths": {
                "input_games": "data/processed/nba_games_with_game_id_processed.csv",
                "output_players": "data/processed/all_boxscores.csv",
                "output_team_v3": "data/processed/team_boxscores_v3.csv",
                "failed_games": "data/processed/failed_game_ids.csv",
                "cache_dir": "data/cache/boxscores",
            },
            "endpoint_families": {
                "traditional": True,
                "advanced": True,
                "four_factors": True,
                "scoring": True,
                "usage": True,
                "misc": True,
                "player_track": True,
            },
            "api": {
                "sleep_between_calls": 1.2,
                "max_retries": 4,
                "timeout": 60,
                "batch_size": 7000,
                "restart_after_batch": True,
                "headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                    "Accept": "application/json, text/plain, */*",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://www.nba.com/",
                    "Origin": "https://www.nba.com",
                    "Connection": "keep-alive",
                },
            },
        },
        "step3a": {
            "paths": {
                "input_boxscores": "data/processed/all_boxscores.csv",
                "input_team_boxscores_v3": "data/processed/team_boxscores_v3.csv",
                "output_team_opp": "data/processed/team opp stats.csv",
            },
            "stats": {
                "stat_cols": [
                    "points",
                    "rebounds",
                    "assists",
                    "STL",
                    "BLK",
                    "turnovers",
                    "OREB",
                    "fgm",
                    "fga",
                    "three_pm",
                    "three_pa",
                    "ftm",
                    "fta",
                    "minutes_played",
                ],
            },
        },
        "step4": {
            "paths": {
                "input_games": "data/processed/nba_games_with_game_id_processed.csv",
                "input_players": "data/processed/all_boxscores.csv",
                "output_team_game": "data/processed/team_game.csv",
                "output_df_model": "data/processed/df_model.csv",
            },
            "rolling": {
                "windows": [5, 10],
                "min_games": 3,
                "team_rolls": [5, 10],
                "top_n_players": 8,
                "top_usage_n": 2,
                "bench_top_n": 5,
            },
            "player_strength": {
                "key_score_threshold": 0.75,
            },
            "injury": {
                "half_life_days": 7,
            },
        },
        "step5": {
            "paths": {
                "input_model": "data/processed/df_model.csv",
                "arena_locations": "data/raw/arena_locations.csv",
                "output_model": "data/processed/df_model_with_travel.csv",
            },
            "travel": {
                "windows": [1, 3, 7, 10],
            },
            "altitude": {
                "teams": {
                    "DEN": 5280,
                    "UTA": 4226,
                    "PHX": 1086,
                },
                "threshold": 1000,
            },
            "fatigue": {
                "weights": {
                    "d1": 0.5,
                    "d3": 0.3,
                    "d7": 0.2,
                }
            },
        },
        "step6": {
            "paths": {
                "input_team_opp": "data/processed/team opp stats.csv",
                "input_model": "data/processed/df_model_with_travel.csv",
                "output_model": "data/processed/df_model_2.csv",
            },
            "rolling": {
                "windows": [5, 10, 20],
                "min_games": 3,
            },
            "ewm": {
                "decay": 0.8,
                "windows": [5, 10, 20],
            },
            "season_to_date": {
                "enabled": True,
                "min_games": 3,
            },
            "stats": {
                "roll_columns": [
                    "net_points",
                    "net_rebounds",
                    "net_assists",
                    "net_STL",
                    "net_BLK",
                    "net_turnovers",
                    "net_OREB",
                    "net_fgm",
                    "net_fga",
                    "net_three_pm",
                    "net_three_pa",
                    "net_ftm",
                    "net_fta",
                    "off_rating",
                    "def_rating",
                    "net_rating",
                    "pace",
                    "three_pa_rate",
                    "paint_points",
                    "rebound_rate",
                    "net_three_pa",
                    "adv_off_rating",
                    "adv_def_rating",
                    "adv_net_rating",
                    "adv_pace",
                    "adv_true_shooting_pct",
                    "adv_effective_fg_pct",
                    "adv_rebound_pct",
                    "adv_turnover_pct",
                    "ff_effective_fg_pct",
                    "ff_turnover_pct",
                    "ff_off_reb_pct",
                    "ff_free_throw_rate",
                    "sc_points_in_paint",
                    "sc_points_off_turnovers",
                    "sc_second_chance_points",
                    "sc_fast_break_points",
                    "sc_points_from_bench",
                    "usg_top2_share",
                    "usg_top3_share",
                    "usg_est_possessions",
                    "misc_points_off_turnovers",
                    "misc_second_chance_points",
                    "misc_fast_break_points",
                    "misc_points_in_paint",
                    "ptrk_touches",
                    "ptrk_passes",
                    "ptrk_distance",
                    "ptrk_rebound_chances",
                ],
                "ewm_columns": [
                    "off_rating",
                    "def_rating",
                    "net_rating",
                    "pace",
                    "adv_off_rating",
                    "adv_def_rating",
                    "adv_net_rating",
                    "adv_pace",
                    "ff_effective_fg_pct",
                    "ff_turnover_pct",
                    "sc_points_in_paint",
                    "usg_top2_share",
                    "usg_top3_share",
                    "misc_points_off_turnovers",
                    "ptrk_touches",
                    "ptrk_passes",
                ],
            },
        },
        "step6a": {
            "paths": {
                "input_model": "data/processed/df_model_2.csv",
                "output_model": "data/processed/df_model_3.csv",
            },
            "columns": {
                "keep": [
                    "season",
                    "date_x",
                    "GAME_ID",
                    "home_margin",
                    "away_margin",
                    "spread",
                    "spread_signed",
                    "is_home_favorite",
                    "home_margin_last5",
                    "away_margin_last5",
                    "home_b2b",
                    "away_b2b",
                    "pts_diff_last5",
                    "margin_diff_last5",
                    "rest_diff",
                    "elo_diff",
                    "home_elo_roll_2Y",
                    "away_elo_roll_2Y",
                    "rolling_margin_diff_5",
                    "rolling_margin_diff_10",
                    "rest_advantage",
                    "rest_advantage_sq",
                    "top8_points_diff",
                    "travel_diff_3d",
                    "travel_diff_7d",
                    "travel_diff_10d",
                    "travel_1d_z",
                    "travel_3d_z",
                    "travel_7d_z",
                    "travel_10d_z",
                    "fatigue_diff",
                    "home_team_net_points_r5",
                    "home_team_net_rebounds_r5",
                    "home_team_net_assists_r5",
                    "home_team_net_STL_r5",
                    "home_team_net_BLK_r5",
                    "home_team_net_turnovers_r5",
                    "home_team_net_OREB_r5",
                    "home_team_net_fgm_r5",
                    "home_team_net_fga_r5",
                    "home_team_net_three_pm_r5",
                    "home_team_net_three_pa_r5",
                    "home_team_net_ftm_r5",
                    "home_team_net_fta_r5",
                    "home_team_net_points_r10",
                    "home_team_net_rebounds_r10",
                    "home_team_net_assists_r10",
                    "home_team_net_STL_r10",
                    "home_team_net_BLK_r10",
                    "home_team_net_turnovers_r10",
                    "home_team_net_OREB_r10",
                    "home_team_net_fgm_r10",
                    "home_team_net_fga_r10",
                    "home_team_net_three_pm_r10",
                    "home_team_net_three_pa_r10",
                    "home_team_net_ftm_r10",
                    "home_team_net_fta_r10",
                    "away_team_net_points_r5",
                    "away_team_net_rebounds_r5",
                    "away_team_net_assists_r5",
                    "away_team_net_STL_r5",
                    "away_team_net_BLK_r5",
                    "away_team_net_turnovers_r5",
                    "away_team_net_OREB_r5",
                    "away_team_net_fgm_r5",
                    "away_team_net_fga_r5",
                    "away_team_net_three_pm_r5",
                    "away_team_net_three_pa_r5",
                    "away_team_net_ftm_r5",
                    "away_team_net_fta_r5",
                    "away_team_net_points_r10",
                    "away_team_net_rebounds_r10",
                    "away_team_net_assists_r10",
                    "away_team_net_STL_r10",
                    "away_team_net_BLK_r10",
                    "away_team_net_turnovers_r10",
                    "away_team_net_OREB_r10",
                    "away_team_net_fgm_r10",
                    "away_team_net_fga_r10",
                    "away_team_net_three_pm_r10",
                    "away_team_net_three_pa_r10",
                    "away_team_net_ftm_r10",
                    "away_team_net_fta_r10",
                    "abs_elo_diff",
                    "elo_diff_squared",
                    "sigmoid_elo_diff",
                    "after_ot_flag",
                    "win_streak",
                    "loss_streak",
                    "streak_length_squared",
                    "games_last_3_days",
                    "games_last_5_days",
                    "games_last_7_days",
                    "rest_weighted_load",
                    "margin_std_last_10",
                    "home_margin_std_last_10",
                    "away_margin_std_last_10",
                    "blowout_pct_last20",
                    "blowout_pct_season",
                    "ats_roll_last20",
                    "pregame_missing_top1",
                    "pregame_missing_top2",
                    "pregame_minutes_out",
                    "pregame_usage_out",
                    "prev_game_minutes_out",
                    "bench_net_rating_last_5",
                    "off_rating_last_5",
                    "def_rating_last_5",
                    "net_rating_last_5",
                    "net_rating_trend",
                    "pace_diff",
                    "combined_pace",
                    "pace_interaction",
                    "net_rating_diff",
                    "rest_diff_elo_diff",
                    "home_elo_roll_2Y_rest_advantage",
                    "pace_diff_rest_diff",
                    "net_rating_diff_rest_diff",
                    "abs_edge",
                    "edge_rank_today",
                    "edge_percentile_last_30_days",
                    "model_error_roll_last20",
                    "team_style_vs_opponent_style",
                ]
            },
        },
    },
}


def _deep_merge(base, updates):
    """Recursively merge `updates` into `base` in-place.

    Parameters
    ----------
    base : dict
        Base dictionary to mutate.
    updates : dict
        Update values; nested dicts are merged recursively.

    Returns
    -------
    dict
        The mutated base dictionary.
    """
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def get_config(overrides=None):
    """Return a deep-copied config with optional overrides applied.

    Parameters
    ----------
    overrides : dict, optional
        Partial config override tree to merge into defaults.

    Returns
    -------
    dict
        Resolved configuration dictionary.
    """
    config = deepcopy(DEFAULT_CONFIG)
    if overrides:
        _deep_merge(config, overrides)
    return config


def resolve_step_config(config, step_name):
    """Resolve step-specific configuration by merging step overrides.

    Parameters
    ----------
    config : dict
        Base configuration dictionary.
    step_name : str
        Step key to resolve (e.g., "step6a").

    Returns
    -------
    dict
        Combined configuration for the given step.
    """
    base = deepcopy(config)
    step_overrides = base.pop("steps", {}).get(step_name, {})
    return _deep_merge(base, step_overrides)
