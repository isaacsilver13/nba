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
