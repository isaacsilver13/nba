"""Shared utilities for dataPrep steps."""

import os
from pathlib import Path

import pandas as pd


def load_cached_output(output_path, cache_cfg=None):
    """Load cached CSV output if caching is enabled.

    Parameters
    ----------
    output_path : str or Path
        Target CSV path to read.
    cache_cfg : dict, optional
        Cache settings with keys: enabled, read_on_hit.

    Returns
    -------
    tuple[pd.DataFrame | None, bool]
        (dataframe, True) when loaded; otherwise (None, False).
    """
    cache_cfg = cache_cfg or {}
    if cache_cfg.get("enabled") and cache_cfg.get("read_on_hit") and Path(output_path).exists():
        return pd.read_csv(output_path), True
    return None, False


def save_output(df, output_path, cache_cfg=None):
    """Persist dataframe to CSV when caching allows writes.

    Parameters
    ----------
    df : pd.DataFrame
        Data to write.
    output_path : str or Path
        Destination CSV path.
    cache_cfg : dict, optional
        Cache settings with keys: enabled, write_outputs.

    Returns
    -------
    bool
        True when the file is written; False otherwise.
    """
    cache_cfg = cache_cfg or {}
    if cache_cfg.get("enabled") and cache_cfg.get("write_outputs"):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return True
    return False


def compute_rolling_stats(df, group_col, stat_cols, windows, min_periods=None):
    """Compute shifted rolling means for multiple stats and windows.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    group_col : str
        Column to group by (e.g., team or player id).
    stat_cols : list[str]
        Stat column names to roll.
    windows : list[int]
        Window sizes for rolling mean.
    min_periods : int, optional
        Minimum periods for rolling; defaults to max(1, window//3).

    Returns
    -------
    pd.DataFrame
        DataFrame with rolling feature columns appended.
    """
    df = df.copy()
    
    for w in windows:
        window_min = min_periods if min_periods is not None else max(1, w // 3)
        for stat in stat_cols:
            if stat in df.columns:
                df[f"{stat}_r{w}"] = (
                    df.groupby(group_col)[stat]
                    .transform(lambda x: x.shift(1).rolling(w, min_periods=window_min).mean())
                ).fillna(0)
    
    return df


def normalize_status(s):
    """Normalize a player status string into a canonical form.

    Parameters
    ----------
    s : str or None
        Raw status string.

    Returns
    -------
    str
        Upper-cased, de-delimited status label.
    """
    if pd.isna(s):
        return ""
    return s.upper().replace("_", " ").replace("-", " ").strip()


def parse_absence_reason(row, injury_key="injury_absence", rest_key="rest_absence", 
                         suspension_key="suspension_absence", coach_key="coach_decision_absence",
                         personal_key="personal_absence", trade_key="trade_absence"):
    """Select a primary absence reason from multiple boolean indicators.

    Parameters
    ----------
    row : dict-like
        Row with boolean absence flags.
    injury_key, rest_key, suspension_key, coach_key, personal_key, trade_key : str
        Column names for each absence category.

    Returns
    -------
    str
        Canonical reason label (injury, rest, suspension, coach, personal, other).
    """
    if row.get(injury_key):
        return "injury"
    if row.get(rest_key):
        return "rest"
    if row.get(suspension_key):
        return "suspension"
    if row.get(coach_key):
        return "coach"
    if row.get(personal_key) or row.get(trade_key):
        return "personal"
    return "other"


def dedupe_player_games(df, game_col="GAME_ID", player_col="PLAYER_ID"):
    """Keep one boxscore row per (game, player).

    The ingested all_boxscores.csv repeats some player-games 16 or 32 times (a join fan-out; the box score columns are
    identical, only advanced columns vary). Left in, team sums are inflated (e.g. 379 points) and a player's rolling
    window counts the same game several times, so a duplicate row's "previous games" include its own game.
    Expects game ids already normalised.
    """
    if game_col not in df.columns or player_col not in df.columns:
        return df
    out = df.drop_duplicates([game_col, player_col], keep="first")
    if len(out) < len(df):
        print(f"[dedupe] dropped {len(df) - len(out):,} duplicate player-game rows")
    return out
