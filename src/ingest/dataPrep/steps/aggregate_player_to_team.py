"""Aggregate player box scores to team-level features."""

from pathlib import Path

import numpy as np
import pandas as pd

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


def usage_concentration(df_players, top_ns=(2, 3)):
    """How concentrated a team's offense was in one game: the top-N players' share of team usage.

    Usage proxy per player = FGA + 0.44*FTA + turnovers (the possessions he used); share = proxy / team total.
    Returns one row per (GAME_ID, team) with usg_top{N}_share. This replaces the V3 team-level usage percentage,
    which is 1.0 by definition for a whole team and so carries no information. Built from counting stats only, so it
    exists for every game that has a box score. It describes THIS game; step 6 lags it before the model sees it.
    """
    d = df_players[["GAME_ID", "team", "fga", "fta", "turnovers"]].copy()
    d["used"] = d["fga"] + 0.44 * d["fta"] + d["turnovers"]
    total = d.groupby(["GAME_ID", "team"])["used"].transform("sum").replace(0, np.nan)
    d["share"] = d["used"] / total
    ranked = d.sort_values("share", ascending=False).groupby(["GAME_ID", "team"])["share"]
    out = d[["GAME_ID", "team"]].drop_duplicates().set_index(["GAME_ID", "team"])
    for n in top_ns:
        out[f"usg_top{n}_share"] = ranked.apply(lambda x, n=n: x.head(n).sum())
    return out.reset_index()


def run(config=None, inputs=None):
    """Aggregate player box scores into team and opponent metrics.

    Parameters
    ----------
    config : dict, optional
        Pipeline configuration; defaults to `get_config()` when None.
    inputs : dict, optional
        Prior step outputs; uses `player_boxscores` when provided.

    Returns
    -------
    dict
        Dictionary with `team_opp_stats` dataframe.
    """
    cfg = resolve_step_config(get_config(), "step3a") if config is None else resolve_step_config(config, "step3a")
    paths = cfg["paths"]
    stats_cfg = cfg["stats"]
    cache_cfg = cfg.get("cache", {})

    input_path = Path(paths["input_boxscores"])
    input_team_v3_path = Path(paths.get("input_team_boxscores_v3", "data/processed/team_boxscores_v3.csv"))
    output_path = Path(paths["output_team_opp"])

    # Try to load from cache
    df_cached, from_cache = load_cached_output(str(output_path), cache_cfg)
    if from_cache and not inputs:
        print(f"[step3a] Cache hit: loaded team/opponent stats from {output_path}")
        return {"team_opp_stats": df_cached}

    if inputs and "player_boxscores" in inputs:
        df_boxscores = inputs["player_boxscores"].copy()
    else:
        df_boxscores = pd.read_csv(input_path)

    print(f"[step3a] Player boxscore rows: {len(df_boxscores):,}")

    # Normalize player boxscore schema (supports V2/V3/camelCase variants)
    column_aliases = {
        "points": ["points", "PTS"],
        "rebounds": ["rebounds", "REB", "reboundsTotal"],
        "assists": ["assists", "AST"],
        "STL": ["STL", "steals"],
        "BLK": ["BLK", "blocks"],
        "turnovers": ["turnovers", "TO"],
        "OREB": ["OREB", "reboundsOffensive"],
        "fgm": ["fgm", "FGM", "fieldGoalsMade"],
        "fga": ["fga", "FGA", "fieldGoalsAttempted"],
        "three_pm": ["three_pm", "FG3M", "threePointersMade"],
        "three_pa": ["three_pa", "FG3A", "threePointersAttempted"],
        "ftm": ["ftm", "FTM", "freeThrowsMade"],
        "fta": ["fta", "FTA", "freeThrowsAttempted"],
        "minutes": ["minutes", "MIN"],
        "team": ["team", "TEAM_ABBREVIATION"],
        "GAME_ID": ["GAME_ID", "gameId"],
    }

    for target, aliases in column_aliases.items():
        if target in df_boxscores.columns:
            continue
        src = next((c for c in aliases if c in df_boxscores.columns), None)
        if src is not None:
            df_boxscores[target] = df_boxscores[src]

    if "GAME_ID" in df_boxscores.columns:
        df_boxscores["GAME_ID"] = (
            df_boxscores["GAME_ID"].astype(str).str.replace(".0", "", regex=False).str.zfill(10)
        )

    if "team" in df_boxscores.columns:
        df_boxscores["team"] = df_boxscores["team"].astype(str).str.upper().str.strip()

    df_boxscores = dedupe_player_games(df_boxscores)  # duplicated player-games would be summed into team totals

    numeric_cols = [
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
    ]
    for col in numeric_cols:
        if col in df_boxscores.columns:
            df_boxscores[col] = pd.to_numeric(df_boxscores[col], errors="coerce").fillna(0)

    if inputs and "team_boxscores_v3" in inputs and inputs["team_boxscores_v3"] is not None:
        df_team_v3 = inputs["team_boxscores_v3"].copy()
    elif input_team_v3_path.exists():
        df_team_v3 = pd.read_csv(input_team_v3_path)
    else:
        df_team_v3 = pd.DataFrame()

    print(f"[step3a] Team V3 rows: {len(df_team_v3):,}")

    def _parse_minutes(x):
        """Parse a minutes string (MM:SS) into float minutes.

        Parameters
        ----------
        x : str or float
            Raw minutes value.

        Returns
        -------
        float
            Minutes as a float; 0.0 on missing/unparseable values.
        """
        if pd.isna(x):
            return 0.0
        if ":" in str(x):
            m, s = str(x).split(":")
            return int(m) + int(s) / 60
        return 0.0

    if "minutes_played" not in df_boxscores.columns and "minutes" in df_boxscores.columns:
        df_boxscores["minutes_played"] = df_boxscores["minutes"].apply(_parse_minutes)

    stat_cols = stats_cfg["stat_cols"]
    if "minutes_played" in df_boxscores.columns and "minutes_played" not in stat_cols:
        stat_cols = stat_cols + ["minutes_played"]

    team_stats = (
        df_boxscores.groupby(["GAME_ID", "team"], as_index=False)[stat_cols].sum()
    )
    print(f"[step3a] Aggregated team-game rows: {len(team_stats):,}")

    team_with_opp = team_stats.merge(team_stats, on="GAME_ID", suffixes=("", "_opp"))

    team_with_opp = team_with_opp[team_with_opp["team"] != team_with_opp["team_opp"]]

    team_with_opp = team_with_opp.rename(columns={"team_opp": "opp_team"})

    for stat in stat_cols:
        team_with_opp[f"net_{stat}"] = team_with_opp[stat] - team_with_opp[f"{stat}_opp"]

    team_with_opp["team_poss"] = (
        team_with_opp["fga"] + 0.44 * team_with_opp["fta"] - team_with_opp["OREB"] + team_with_opp["turnovers"]
    )
    team_with_opp["opp_poss"] = (
        team_with_opp["fga_opp"] + 0.44 * team_with_opp["fta_opp"] - team_with_opp["OREB_opp"] + team_with_opp["turnovers_opp"]
    )
    team_with_opp["possessions"] = 0.5 * (team_with_opp["team_poss"] + team_with_opp["opp_poss"])

    minutes_base = (team_with_opp["minutes_played"] / 5).replace(0, np.nan)
    team_with_opp["pace"] = (48 * team_with_opp["possessions"] / minutes_base).fillna(0)

    team_with_opp["off_rating"] = (
        team_with_opp["points"] / team_with_opp["possessions"].replace(0, np.nan) * 100
    ).fillna(0)
    team_with_opp["def_rating"] = (
        team_with_opp["points_opp"] / team_with_opp["possessions"].replace(0, np.nan) * 100
    ).fillna(0)
    team_with_opp["net_rating"] = team_with_opp["off_rating"] - team_with_opp["def_rating"]

    team_with_opp["three_pa_rate"] = (
        team_with_opp["three_pa"] / (team_with_opp["fga"] + 1e-6)
    ).fillna(0)

    team_with_opp["paint_points"] = (
        team_with_opp["points"] - 3 * team_with_opp["three_pm"] - team_with_opp["ftm"]
    ).clip(lower=0)

    team_with_opp["rebound_rate"] = (
        team_with_opp["rebounds"] / (team_with_opp["rebounds"] + team_with_opp["rebounds_opp"] + 1e-6)
    ).fillna(0)

    team_with_opp = team_with_opp.merge(usage_concentration(df_boxscores), on=["GAME_ID", "team"], how="left")

    if not df_team_v3.empty:
        team_v3 = df_team_v3.copy()
        if "GAME_ID" in team_v3.columns:
            team_v3["GAME_ID"] = team_v3["GAME_ID"].astype(str).str.replace(".0", "", regex=False).str.zfill(10)

        if "TEAM_ABBREVIATION" in team_v3.columns:
            team_v3 = team_v3.rename(columns={"TEAM_ABBREVIATION": "team"})
        elif "team_abbreviation" in team_v3.columns:
            team_v3 = team_v3.rename(columns={"team_abbreviation": "team"})

        canonical_map = {
            "adv_off_rating": ["adv_off_rating", "adv_offensive_rating"],
            "adv_def_rating": ["adv_def_rating", "adv_defensive_rating"],
            "adv_net_rating": ["adv_net_rating"],
            "adv_pace": ["adv_pace"],
            "adv_true_shooting_pct": ["adv_true_shooting_pct", "adv_true_shooting_percentage"],
            "adv_effective_fg_pct": [
                "adv_effective_fg_pct",
                "adv_effective_fg_percentage",
                "adv_effective_field_goal_percentage",
            ],
            "adv_rebound_pct": ["adv_rebound_pct", "adv_rebound_percentage"],
            "adv_turnover_pct": ["adv_turnover_pct", "adv_turnover_percentage", "adv_team_turnover_percentage"],
            "ff_effective_fg_pct": [
                "ff_effective_fg_pct",
                "ff_effective_fg_percentage",
                "ff_effective_field_goal_percentage",
            ],
            "ff_turnover_pct": ["ff_turnover_pct", "ff_turnover_percentage", "ff_team_turnover_percentage"],
            "ff_off_reb_pct": ["ff_off_reb_pct", "ff_offensive_rebound_percentage"],
            "ff_free_throw_rate": ["ff_free_throw_rate", "ff_free_throw_attempt_rate"],
            "sc_points_in_paint": ["sc_points_in_paint", "sc_points_paint"],
            "sc_points_off_turnovers": ["sc_points_off_turnovers"],
            "sc_second_chance_points": ["sc_second_chance_points"],
            "sc_fast_break_points": ["sc_fast_break_points"],
            "sc_points_from_bench": ["sc_points_from_bench", "sc_bench_points"],
            "usg_usage_pct": ["usg_usage_pct", "usg_usage_percentage"],
            "usg_est_possessions": ["usg_est_possessions", "usg_possessions"],
            "misc_points_off_turnovers": ["misc_points_off_turnovers"],
            "misc_second_chance_points": ["misc_second_chance_points"],
            "misc_fast_break_points": ["misc_fast_break_points"],
            "misc_points_in_paint": ["misc_points_in_paint", "misc_points_paint"],
            "ptrk_touches": ["ptrk_touches"],
            "ptrk_passes": ["ptrk_passes"],
            "ptrk_distance": ["ptrk_distance"],
            "ptrk_rebound_chances": ["ptrk_rebound_chances"],
        }

        base_cols = [c for c in ["GAME_ID", "team"] if c in team_v3.columns]
        team_v3_norm = team_v3[base_cols].copy()

        for target_col, candidates in canonical_map.items():
            src = next((c for c in candidates if c in team_v3.columns), None)
            if src is not None:
                team_v3_norm[target_col] = pd.to_numeric(team_v3[src], errors="coerce")

        metric_cols = [c for c in team_v3_norm.columns if c not in ["GAME_ID", "team"]]
        if metric_cols and "GAME_ID" in team_v3_norm.columns and "team" in team_v3_norm.columns:
            team_v3_norm = team_v3_norm.groupby(["GAME_ID", "team"], as_index=False)[metric_cols].mean()
            team_v3_norm["_has_v3"] = 1
            team_with_opp = team_with_opp.merge(team_v3_norm, on=["GAME_ID", "team"], how="left")
            # ESPN-sourced games (tools/ingest_daily.py, GAME_ID ...9xxxx) carry only the metrics ESPN can supply
            espn = team_with_opp["GAME_ID"].astype(int) % 100000 > 90000
            has_v3 = team_with_opp.pop("_has_v3").notna() & ~espn
            for col in metric_cols:
                # a metric missing inside an existing NBA V3 row is filled with 0 as before; a game with NO V3 row
                # (backfilled from the box-score-only CSV) or an ESPN game stays NaN so rolling means skip it instead
                # of averaging in zeros
                vals = pd.to_numeric(team_with_opp[col], errors="coerce")
                team_with_opp[col] = vals.where(~has_v3, vals.fillna(0))

    assert team_with_opp.groupby("GAME_ID").size().eq(2).all()
    assert (team_with_opp["team"] != team_with_opp["opp_team"]).all()

    print(f"[step3a] Final team/opponent rows: {len(team_with_opp):,}")

    # Save to cache if enabled
    save_output(team_with_opp, str(output_path), cache_cfg)
    print(f"[step3a] Saved output to {output_path}")

    return {"team_opp_stats": team_with_opp}


if __name__ == "__main__":
    run(get_config())
