"""Fetch a held-out season's games and box scores into data/holdout/<season>/ (never into data/processed).

    python tools/fetch_holdout.py --season 2025-26 [--games-only]

Writes games_<tag>.csv (GAME_ID, date, season, home, away, score_home, score_away; season = ending year, as in
the processed games table) from LeagueGameFinder, then runs step 3's box-score fetcher with its outputs redirected
here. Per-game downloads share data/cache/boxscores, so a rerun resumes where it stopped. Lines/odds are NOT
fetched: the holdout needs its own closing-line source before it can be evaluated.
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from ingest.dataPrep.config import get_config  # noqa: E402


def fetch_games(season, headers):
    from nba_api.stats.endpoints import LeagueGameFinder
    g = LeagueGameFinder(season_nullable=season, season_type_nullable="Regular Season", league_id_nullable="00",
                         timeout=60, headers=headers).get_data_frames()[0]
    # designated home team from the matchup text: "MIA vs. ATL" and "ATL @ MIA" both mean MIA. (Neutral-site games
    # such as international games and NBA Cup semifinals show "X @ Y" on BOTH teams' rows.)
    m = g["MATCHUP"].str.split(r" vs\. | @ ", regex=True)
    g["home_team"] = m.str[0].where(g["MATCHUP"].str.contains(" vs. "), m.str[1])
    is_home = g["TEAM_ABBREVIATION"] == g["home_team"]
    home, away = g[is_home].set_index("GAME_ID"), g[~is_home].set_index("GAME_ID")
    games = pd.DataFrame({
        "date": pd.to_datetime(home["GAME_DATE"]),
        "season": int(season[:4]) + 1,
        "home": home["TEAM_ABBREVIATION"],
        "away": away["TEAM_ABBREVIATION"].reindex(home.index),
        "score_home": home["PTS"],
        "score_away": away["PTS"].reindex(home.index),
    }).rename_axis("GAME_ID").reset_index().sort_values(["date", "GAME_ID"])
    assert games["GAME_ID"].is_unique and games.notna().all().all(), "every game needs one home and one away row"
    return games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True, help="e.g. 2025-26")
    ap.add_argument("--games-only", action="store_true")
    a = ap.parse_args()
    os.chdir(REPO)  # config paths are relative to the repo root

    tag = a.season.replace("-", "_")
    out = Path("data") / "holdout" / tag
    out.mkdir(parents=True, exist_ok=True)
    cfg = get_config({"steps": {"step3": {"paths": {
        "input_games": str(out / f"games_{tag}.csv"),
        "output_players": str(out / f"all_boxscores_{tag}.csv"),
        "output_team_v3": str(out / f"team_boxscores_v3_{tag}.csv"),
        "failed_games": str(out / f"failed_game_ids_{tag}.csv"),
    }}}})

    games = fetch_games(a.season, cfg["steps"]["step3"]["api"]["headers"])
    games.to_csv(out / f"games_{tag}.csv", index=False)
    print(f"{len(games)} games {games['date'].min().date()} .. {games['date'].max().date()} -> {out}")
    if a.games_only:
        return

    from ingest.dataPrep.steps import fetch_player_boxscores
    res = fetch_player_boxscores.run(cfg)
    p = res["player_boxscores"]
    print(f"box scores: {p['GAME_ID'].nunique()} / {len(games)} games, {len(p):,} player rows")


if __name__ == "__main__":
    main()
