"""Fetch player box score data per game from the NBA API."""

import os
import time
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from nba_api.stats import endpoints as nba_endpoints
from nba_api.stats.library.http import NBAStatsHTTP

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


ENDPOINT_CLASS_MAP = {
    "traditional": "BoxScoreTraditionalV3",
    "advanced": "BoxScoreAdvancedV3",
    "four_factors": "BoxScoreFourFactorsV3",
    "scoring": "BoxScoreScoringV3",
    "usage": "BoxScoreUsageV3",
    "misc": "BoxScoreMiscV3",
    "player_track": "BoxScorePlayerTrackV3",
}

FAMILY_PREFIX = {
    "traditional": "trad",
    "advanced": "adv",
    "four_factors": "ff",
    "scoring": "sc",
    "usage": "usg",
    "misc": "misc",
    "player_track": "ptrk",
}


def _to_snake(col_name):
    """Convert an API column name to snake_case."""
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", str(col_name))
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.replace("%", "pct").replace(" ", "_").replace("-", "_").lower()


def _canonicalize_keys(df):
    """Normalize key identifiers across endpoint payload variants."""
    if df is None or df.empty:
        return df

    rename = {}
    key_aliases = {
        "GAME_ID": ["GAME_ID", "game_id", "gameId"],
        "TEAM_ID": ["TEAM_ID", "team_id", "teamId"],
        "PLAYER_ID": ["PLAYER_ID", "player_id", "person_id", "playerId", "personId"],
        "TEAM_ABBREVIATION": ["TEAM_ABBREVIATION", "team_abbreviation", "teamTricode", "team_tricode"],
    }

    cols = {c: c for c in df.columns}
    cols_snake = {_to_snake(c): c for c in df.columns}

    for canonical, aliases in key_aliases.items():
        found = None
        for alias in aliases:
            if alias in cols:
                found = alias
                break
            alias_snake = _to_snake(alias)
            if alias_snake in cols_snake:
                found = cols_snake[alias_snake]
                break
        if found and found != canonical:
            rename[found] = canonical

    out = df.rename(columns=rename).copy()
    if "GAME_ID" in out.columns:
        out["GAME_ID"] = out["GAME_ID"].astype(str).str.replace(".0", "", regex=False).str.zfill(10)
    return out


def _prefix_metric_columns(df, family, keys):
    """Prefix non-key metrics from a family frame with family abbreviation."""
    if df is None or df.empty:
        return pd.DataFrame()

    prefix = FAMILY_PREFIX[family]
    out = df.copy()
    rename = {}
    for col in out.columns:
        if col in keys:
            continue
        snake = _to_snake(col)
        rename[col] = f"{prefix}_{snake}"
    out = out.rename(columns=rename)

    # Keep numeric and key columns only for compact merges
    metric_cols = [c for c in out.columns if c in keys or pd.api.types.is_numeric_dtype(out[c])]
    return out.loc[:, metric_cols]


def _coerce_join_keys(df, keys):
    """Normalize merge-key dtypes to stable string formats."""
    out = df.copy()
    for key in keys:
        if key not in out.columns:
            continue
        if key == "GAME_ID":
            out[key] = out[key].astype(str).str.replace(".0", "", regex=False).str.zfill(10)
        elif key in {"TEAM_ID", "PLAYER_ID"}:
            out[key] = out[key].astype(str).str.replace(".0", "", regex=False)
        elif key == "TEAM_ABBREVIATION":
            out[key] = out[key].astype(str).str.upper().str.strip()
    return out


def _require_unique(df, keys, label):
    """Fail loudly if a consolidated table has more than one row per key.

    A family frame with repeated keys used to multiply rows (16x / 32x) in the merges below; such a table must
    never be written, because every downstream sum and rolling window would silently double count.
    """
    keys = [k for k in keys if k in df.columns]
    dup = int(df.duplicated(keys).sum()) if keys and not df.empty else 0
    if dup:
        raise ValueError(f"{label}: {dup:,} rows repeat their {keys} key after merging endpoint families")


def _build_cache_index(cache_dir, families):
    """Scan cache folders once and index game ids by family and kind."""
    index = {family: {"player": set(), "team": set()} for family in families}
    root = Path(cache_dir)

    for family in families:
        fam_dir = root / family
        if not fam_dir.exists():
            continue

        for name in os.listdir(fam_dir):
            if name.endswith("_player.csv"):
                gid = name.replace("_player.csv", "")
                if gid.isdigit():
                    index[family]["player"].add(gid.zfill(10))
            elif name.endswith("_team.csv"):
                gid = name.replace("_team.csv", "")
                if gid.isdigit():
                    index[family]["team"].add(gid.zfill(10))

    return index


def _rebuild_outputs_from_cache(game_ids_z, active_families, cache_dir, cache_index=None):
    """Rebuild consolidated player and team tables from cached family files.

    Fast path: load per-family frames once, then merge once per family.
    """
    if cache_index is None:
        cache_index = _build_cache_index(cache_dir, active_families)

    game_id_set = set(game_ids_z)

    player_base = pd.DataFrame()
    player_metric_frames = []
    team_metric_frames = []

    for family in active_families:
        player_gids = sorted(game_id_set & cache_index[family]["player"])
        team_gids = sorted(game_id_set & cache_index[family]["team"])

        family_player_df = pd.DataFrame()
        family_team_df = pd.DataFrame()

        if player_gids:
            player_parts = []
            for gid in player_gids:
                player_path = os.path.join(cache_dir, family, f"{gid}_player.csv")
                part = pd.read_csv(player_path)
                part = _canonicalize_keys(part)
                player_parts.append(part)
            family_player_df = pd.concat(player_parts, ignore_index=True) if player_parts else pd.DataFrame()

        if team_gids:
            team_parts = []
            for gid in team_gids:
                team_path = os.path.join(cache_dir, family, f"{gid}_team.csv")
                part = pd.read_csv(team_path)
                part = _canonicalize_keys(part)
                team_parts.append(part)
            family_team_df = pd.concat(team_parts, ignore_index=True) if team_parts else pd.DataFrame()

        if family == "traditional":
            if not family_player_df.empty:
                player_base = family_player_df.copy()
                player_base = _coerce_join_keys(player_base, ["GAME_ID", "TEAM_ID", "PLAYER_ID"])
                player_base = player_base.drop_duplicates(["GAME_ID", "TEAM_ID", "PLAYER_ID"], keep="first")
        else:
            if not family_player_df.empty:
                keys = [k for k in ["GAME_ID", "TEAM_ID", "PLAYER_ID"] if k in family_player_df.columns]
                if all(k in keys for k in ["GAME_ID", "TEAM_ID", "PLAYER_ID"]):
                    player_metrics = _prefix_metric_columns(family_player_df, family=family, keys=keys)
                    player_metrics = _coerce_join_keys(player_metrics, keys)
                    player_metrics = player_metrics.groupby(keys, as_index=False).mean(numeric_only=True)
                    player_metric_frames.append(player_metrics)

        if not family_team_df.empty:
            team_keys = [k for k in ["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION"] if k in family_team_df.columns]
            if "GAME_ID" in team_keys and "TEAM_ID" in team_keys:
                team_metrics = _prefix_metric_columns(family_team_df, family=family, keys=team_keys)
                team_metrics = _coerce_join_keys(team_metrics, team_keys)
                team_metrics = team_metrics.groupby(team_keys, as_index=False).mean(numeric_only=True)
                team_metric_frames.append(team_metrics)

    df_player = player_base.copy()
    if not df_player.empty:
        for metric_df in player_metric_frames:
            merge_keys = [k for k in ["GAME_ID", "TEAM_ID", "PLAYER_ID"] if k in df_player.columns and k in metric_df.columns]
            if merge_keys:
                df_player = df_player.merge(metric_df, on=merge_keys, how="left")

    df_team = pd.DataFrame()
    for i, team_df in enumerate(team_metric_frames):
        if i == 0:
            df_team = team_df.copy()
        else:
            merge_keys = [k for k in ["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION"] if k in df_team.columns and k in team_df.columns]
            if merge_keys:
                df_team = df_team.merge(team_df, on=merge_keys, how="outer")

    _require_unique(df_player, ["GAME_ID", "TEAM_ID", "PLAYER_ID"], "player boxscores")
    _require_unique(df_team, ["GAME_ID", "TEAM_ID"], "team boxscores")
    return df_player, df_team


def fetch_endpoint_cached(
    game_id,
    family,
    cache_dir,
    headers,
    timeout,
    max_retries,
    read_on_hit,
    write_outputs,
):
    """Fetch a single game endpoint family with disk caching and retries.

    Parameters
    ----------
    game_id : int or str
        NBA game id.
    cache_dir : str
        Directory for cached CSV boxscores.
    headers : dict
        HTTP headers for NBA Stats API.
    timeout : int or float
        Request timeout in seconds.
    max_retries : int
        Maximum retries on API failure.
    read_on_hit : bool
        Whether to read cached files if present.
    write_outputs : bool
        Whether to write fetched results to cache.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, bool]
        Player frame, team frame, and cache-hit flag.
    """
    gid = str(int(game_id)).zfill(10)
    family_dir = os.path.join(cache_dir, family)
    os.makedirs(family_dir, exist_ok=True)
    player_cache_path = os.path.join(family_dir, f"{gid}_player.csv")
    team_cache_path = os.path.join(family_dir, f"{gid}_team.csv")

    if read_on_hit and os.path.exists(player_cache_path) and os.path.exists(team_cache_path):
        return pd.read_csv(player_cache_path), pd.read_csv(team_cache_path), True

    endpoint_name = ENDPOINT_CLASS_MAP.get(family)
    endpoint_cls = getattr(nba_endpoints, endpoint_name, None)
    if endpoint_cls is None:
        print(f"[SKIP] Endpoint not available: {endpoint_name}")
        return pd.DataFrame(), pd.DataFrame(), True

    for attempt in range(max_retries):
        try:
            endpoint = endpoint_cls(game_id=gid, timeout=timeout, headers=headers)
            frames = endpoint.get_data_frames()
            player_df = frames[0] if len(frames) > 0 else pd.DataFrame()
            team_df = frames[1] if len(frames) > 1 else pd.DataFrame()

            player_df = _canonicalize_keys(player_df)
            team_df = _canonicalize_keys(team_df)

            if player_df.empty and team_df.empty:
                raise ValueError("Empty endpoint response")

            player_df["endpoint_family"] = family
            team_df["endpoint_family"] = family

            if write_outputs:
                player_df.to_csv(player_cache_path, index=False)
                team_df.to_csv(team_cache_path, index=False)
            return player_df, team_df, False

        except Exception as e:
            wait = 2 ** attempt
            NBAStatsHTTP._session = None

            print(f"[Retry {attempt + 1}] {family} {gid} failed — waiting {wait}s")
            time.sleep(wait)
    print(f"[FAILED PERMANENTLY] {family} {gid}")
    return pd.DataFrame(), pd.DataFrame(), False


def run(config=None, inputs=None):
    """Run step 3 to fetch player box scores across games.

    Parameters
    ----------
    config : dict, optional
        Pipeline configuration; defaults to `get_config()` when None.
    inputs : dict, optional
        Prior step outputs; uses `games_processed` when provided.

    Returns
    -------
    dict
        Dictionary with `player_boxscores` dataframe.
    """
    cfg = resolve_step_config(get_config(), "step3") if config is None else resolve_step_config(config, "step3")
    paths = cfg["paths"]
    api_cfg = cfg["api"]
    cache_cfg = cfg.get("cache", {})

    input_path = Path(paths["input_games"])
    output_path = Path(paths["output_players"])
    output_team_path = Path(paths.get("output_team_v3", "data/processed/team_boxscores_v3.csv"))
    failed_path = Path(paths["failed_games"])
    cache_dir = Path(paths["cache_dir"])

    cache_dir.mkdir(parents=True, exist_ok=True)

    if inputs and "games_processed" in inputs:
        df_games = inputs["games_processed"].copy()
    else:
        df_games = pd.read_csv(input_path, dtype={"GAME_ID": "Int64"}, parse_dates=["date"])

    df_games = df_games.dropna(subset=["GAME_ID"]).copy()

    print(f"Games to process: {len(df_games):,}")

    failed_game_ids = []

    family_cfg = cfg.get("endpoint_families", {})
    active_families = [
        name
        for name in ["traditional", "advanced", "four_factors", "scoring", "usage", "misc", "player_track"]
        if family_cfg.get(name, False)
    ]
    if "traditional" not in active_families:
        active_families = ["traditional"] + active_families
    print(f"Active endpoint families: {', '.join(active_families)}")

    batch_size = api_cfg.get("batch_size", 1000)
    restart_after_batch = api_cfg.get("restart_after_batch", True)
    endpoint_workers = max(1, int(api_cfg.get("endpoint_workers", min(4, len(active_families)))))

    game_ids = df_games["GAME_ID"].astype("int64").tolist()
    game_ids_z = [str(int(gid)).zfill(10) for gid in game_ids]

    cache_index = _build_cache_index(str(cache_dir), active_families)
    missing_by_family = {}
    for family in active_families:
        missing_player = set(game_ids_z) - cache_index[family]["player"]
        missing_team = set(game_ids_z) - cache_index[family]["team"]
        missing_by_family[family] = {
            "player": missing_player,
            "team": missing_team,
        }
        print(
            f"[CACHE INDEX] {family}: player {len(cache_index[family]['player']):,}/{len(game_ids_z):,}, "
            f"team {len(cache_index[family]['team']):,}/{len(game_ids_z):,}"
        )

    games_to_process = sorted(
        {
            gid
            for family in active_families
            for gid in (missing_by_family[family]["player"] | missing_by_family[family]["team"])
        }
    )

    if not games_to_process:
        df_cached, from_cache = load_cached_output(str(output_path), cache_cfg)
        team_cached, team_from_cache = load_cached_output(str(output_team_path), cache_cfg)
        if from_cache and team_from_cache:
            print("[OK] Cache already complete for all active families; skipping API pulls")
            return {"player_boxscores": df_cached, "team_boxscores_v3": team_cached}

    print(f"[MISSING] Games needing API pulls: {len(games_to_process):,}")

    for i, gid in enumerate(games_to_process, 1):

        families_to_fetch = [
            family
            for family in active_families
            if gid in missing_by_family[family]["player"] or gid in missing_by_family[family]["team"]
        ]

        if not families_to_fetch:
            continue

        all_cached = True

        endpoint_results = {}
        if len(families_to_fetch) == 1:
            fam = families_to_fetch[0]
            endpoint_results[fam] = fetch_endpoint_cached(
                int(gid),
                family=fam,
                cache_dir=str(cache_dir),
                headers=api_cfg["headers"],
                timeout=api_cfg.get("timeout", 60),
                max_retries=api_cfg.get("max_retries", 4),
                read_on_hit=cache_cfg.get("read_on_hit", True),
                write_outputs=cache_cfg.get("write_outputs", True),
            )
        else:
            with ThreadPoolExecutor(max_workers=min(endpoint_workers, len(families_to_fetch))) as executor:
                futures = {
                    executor.submit(
                        fetch_endpoint_cached,
                        int(gid),
                        family,
                        str(cache_dir),
                        api_cfg["headers"],
                        api_cfg.get("timeout", 60),
                        api_cfg.get("max_retries", 4),
                        cache_cfg.get("read_on_hit", True),
                        cache_cfg.get("write_outputs", True),
                    ): family
                    for family in families_to_fetch
                }
                for future in as_completed(futures):
                    family = futures[future]
                    try:
                        endpoint_results[family] = future.result()
                    except Exception:
                        endpoint_results[family] = (pd.DataFrame(), pd.DataFrame(), False)

        for family in families_to_fetch:
            player_df, team_df, from_cache = endpoint_results.get(
                family,
                (pd.DataFrame(), pd.DataFrame(), False),
            )
            all_cached = all_cached and bool(from_cache)

            if family == "traditional" and gid in missing_by_family[family]["player"] and player_df.empty:
                failed_game_ids.append(int(gid))

        if not all_cached:
            time.sleep(api_cfg.get("sleep_between_calls", 1.2))

        if restart_after_batch and i % batch_size == 0:
            print("[BATCH LIMIT] Batch limit reached — stopping early per config")
            break

        if i % 10 == 0 or i == len(games_to_process):
            print(f"[PROGRESS] Processed {i:,}/{len(games_to_process):,} missing-game pulls")

    if failed_game_ids and cache_cfg.get("enabled") and cache_cfg.get("write_outputs"):
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        df_failed = pd.DataFrame({"GAME_ID": failed_game_ids})
        df_failed.to_csv(failed_path, index=False)
        print(f"[WARNING] Failed games saved: {len(df_failed):,}")
    elif failed_game_ids:
        print(f"[WARNING] Failed games (not saved): {len(failed_game_ids):,}")
    else:
        print("[OK] No failed games")

    df_player, df_team_v3 = _rebuild_outputs_from_cache(
        game_ids_z=game_ids_z,
        active_families=active_families,
        cache_dir=str(cache_dir),
        # the index was built before the downloads above; re-scan if anything was fetched, or this run's new games
        # would be left out of the consolidated tables
        cache_index=None if games_to_process else cache_index,
    )

    if df_player.empty:
        raise RuntimeError("No boxscores available after cache rebuild — API likely blocked")

    print(f"[OK] Rebuilt outputs from cache | player_rows={len(df_player):,}, team_rows={len(df_team_v3):,}")

    df_player_clean = df_player.rename(
        columns={
            "PLAYER_NAME": "player",
            "TEAM_ABBREVIATION": "team",
            "MIN": "minutes",
            "PTS": "points",
            "REB": "rebounds",
            "AST": "assists",
            "TO": "turnovers",
            "FGM": "fgm",
            "FGA": "fga",
            "FG3M": "three_pm",
            "FG3A": "three_pa",
            "FTM": "ftm",
            "FTA": "fta",
        }
    )

    if not df_team_v3.empty and "GAME_ID" in df_team_v3.columns:
        df_team_v3["GAME_ID"] = df_team_v3["GAME_ID"].astype(str).str.zfill(10)

    if save_output(df_team_v3, str(output_team_path), cache_cfg):
        print(f"[OK] Saved {len(df_team_v3):,} team V3 rows to {output_team_path}")

    if save_output(df_player_clean, str(output_path), cache_cfg):
        print(f"[OK] Saved {len(df_player_clean):,} player rows to {output_path}")

    return {"player_boxscores": df_player_clean, "team_boxscores_v3": df_team_v3}


if __name__ == "__main__":
    run(get_config())
