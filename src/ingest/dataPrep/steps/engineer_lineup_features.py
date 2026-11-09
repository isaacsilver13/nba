"""Pre-game lineup features from five-man stints (see build_lineup_stints).

For each team and game, only that team's previous `window` regular-season games of the SAME season feed the row, so a
game's own stints (and anything later) can never reach its own features. A season opener therefore has no lineup
features at all (NaN, `lu_games == 0`) rather than last year's lineups presented as current; `lu_games` counts how many
window games actually had stints, so a thin window is visible to the model.

Columns per team (prefixed home_/away_ in the output):
  lu_games                 window games with stints
  lu_days_gap              days since the team's previous game (any season)
  lu_top_unit_share        share of window seconds played by the most-used five
  lu_top_unit_net          that five's net points per 100 possessions, shrunk toward 0 by possessions played
  lu_starter_unit_share    same share for the starting five of the most recent window game
  lu_starter_unit_net      that five's shrunk net rating over the window
  lu_unit_concentration    sum of squared five-man shares (1 = one lineup all the time, ~0 = constant churn)
Plus home-minus-away differences for the net and share columns.
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ..config import get_config, resolve_step_config
    from ..helpers import save_output
except ImportError:
    from ingest.dataPrep.config import get_config, resolve_step_config
    from ingest.dataPrep.helpers import save_output

LU_COLS = ["lu_games", "lu_days_gap", "lu_top_unit_share", "lu_top_unit_net",
           "lu_starter_unit_share", "lu_starter_unit_net", "lu_unit_concentration"]
DIFF_COLS = ["lu_top_unit_net", "lu_top_unit_share", "lu_starter_unit_net", "lu_starter_unit_share"]


def _per_game_units(stints):
    """(GAME_ID, team) -> per-five totals plus that game's starting five."""
    s = stints.copy()
    poss_for = s.fga_for + 0.44 * s.fta_for + s.tov_for - s.oreb_for
    poss_against = s.fga_against + 0.44 * s.fta_against + s.tov_against - s.oreb_against
    s["poss"] = (poss_for + poss_against) / 2
    units, starters = {}, {}
    for (gid, team), g in s.groupby(["GAME_ID", "team"], sort=False):
        units[(gid, team)] = g.groupby("lineup_key")[["seconds", "pts_for", "pts_against", "poss"]].sum()
        starters[(gid, team)] = g.loc[g.start_sec.idxmin(), "lineup_key"]
    return units, starters


def _window_row(window, units, starters, team, shrink_poss):
    """Features for one team from its prior same-season games that have stints (most recent last)."""
    have = [gid for gid in window if (gid, team) in units]
    row = dict.fromkeys(LU_COLS[2:], np.nan)
    row["lu_games"] = len(have)
    if not have:
        return row
    agg = pd.concat([units[(gid, team)] for gid in have]).groupby(level=0).sum()
    total = agg.seconds.sum()
    if total <= 0:
        return row
    share = agg.seconds / total
    net = (agg.pts_for - agg.pts_against) / agg.poss.replace(0, np.nan) * 100
    net = (net * agg.poss / (agg.poss + shrink_poss)).fillna(0.0)
    top = share.idxmax()
    start = starters[(have[-1], team)]
    row.update(lu_top_unit_share=share[top], lu_top_unit_net=net[top],
               lu_starter_unit_share=share.get(start, 0.0), lu_starter_unit_net=net.get(start, 0.0),
               lu_unit_concentration=float((share ** 2).sum()))
    return row


def team_lineup_features(stints, schedule, window=10, shrink_poss=300):
    """One row per (GAME_ID, team). `stints` needs a `team` abbreviation column; `schedule` has GAME_ID, date, season,
    home, away (regular-season games only)."""
    units, starters = _per_game_units(stints)
    long = pd.concat([
        schedule[["GAME_ID", "date", "season", "home"]].rename(columns={"home": "team"}),
        schedule[["GAME_ID", "date", "season", "away"]].rename(columns={"away": "team"}),
    ]).assign(date=lambda d: pd.to_datetime(d.date)).sort_values(["team", "date", "GAME_ID"])
    rows = []
    for team, g in long.groupby("team", sort=False):
        ids, seasons, dates = g.GAME_ID.tolist(), g.season.tolist(), g.date.tolist()
        for i, gid in enumerate(ids):
            prior = [ids[j] for j in range(max(0, i - window), i) if seasons[j] == seasons[i]]
            row = _window_row(prior, units, starters, team, shrink_poss)
            row.update(GAME_ID=gid, team=team,
                       lu_days_gap=(dates[i] - dates[i - 1]).days if i else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)[["GAME_ID", "team"] + LU_COLS]


def game_lineup_features(team_feats, schedule):
    """Wide per-game table: home_/away_ columns and home-minus-away differences."""
    out = schedule[["GAME_ID", "home", "away"]]
    for side in ("home", "away"):
        f = team_feats.rename(columns={"team": side, **{c: f"{side}_{c}" for c in LU_COLS}})
        out = out.merge(f, on=["GAME_ID", side], how="left")
    for c in DIFF_COLS:
        out[f"{c}_diff"] = out[f"home_{c}"] - out[f"away_{c}"]
    return out.drop(columns=["home", "away"])


def run(config=None, inputs=None):
    """Read stints + schedule, write per-game lineup features. Returns {"lineup_features": df}."""
    cfg = resolve_step_config(get_config() if config is None else config, "lineup_features")
    paths, opts = cfg["paths"], cfg["lineup_features"]
    stints = pd.read_csv(paths["input_stints"])
    teams = pd.read_csv(paths["input_boxscores"], usecols=["TEAM_ID", "team"]).drop_duplicates("TEAM_ID")
    stints = stints.merge(teams.rename(columns={"team": "team_abbr"}), on="TEAM_ID").rename(columns={"team_abbr": "team"})
    games = pd.read_csv(paths["input_games"])
    # every regular-season game from the first stint season on, including games with no stints, so a window of
    # "the last 10 games" means 10 games played, not 10 games we happened to reconstruct
    schedule = games[games.regular.astype(bool) & (games.GAME_ID >= stints.GAME_ID.min() // 100000 * 100000)]
    team_feats = team_lineup_features(stints, schedule, opts["window"], opts["shrink_poss"])
    out = game_lineup_features(team_feats, schedule)
    save_output(out, paths["output_features"], cfg.get("cache"))
    print(f"[lineup_features] {len(out)} games -> {paths['output_features']}")
    return {"lineup_features": out}
