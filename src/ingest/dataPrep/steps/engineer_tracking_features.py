"""Pre-game player-tracking features, weighted to a team's expected rotation.

Same leakage rule as engineer_lineup_features: a team's row uses only its previous `window` regular-season games of
the SAME season, so the target game and later games never reach it, and a season opener is blank (`trk_games == 0`).
Within the window the rotation is the team's `top_n` players by window minutes, and every rate is
sum(stat) / sum(minutes) over those players, so a player who barely played cannot move the number. Games whose
tracking columns are empty (older seasons, and the ESPN-ingested 2025-26 season) contribute nothing and are not
counted in `trk_games`.

Columns per team (prefixed home_/away_ in the output), plus home-minus-away differences:
  trk_games                 window games with tracking data
  trk_dist_per_min          miles run per rotation minute
  trk_speed                 minutes-weighted average speed (mph)
  trk_touches_per_min       ball touches per rotation minute
  trk_passes_per_min        passes per rotation minute
  trk_uncontested_fga_share uncontested share of the rotation's field-goal attempts (shot quality)
  trk_rim_fga_per_min       shots defended at the rim per rotation minute (rim presence)
  trk_rim_fg_allowed        field-goal % allowed at the rim, shrunk toward the league mean by attempts faced
  trk_reb_chances_per_min   rebound chances per rotation minute
"""

import numpy as np
import pandas as pd

try:
    from ..config import get_config, resolve_step_config
    from ..helpers import save_output
except ImportError:
    from ingest.dataPrep.config import get_config, resolve_step_config
    from ingest.dataPrep.helpers import save_output

SRC = {
    "dist": "ptrk_distance", "touches": "ptrk_touches", "passes": "ptrk_passes",
    "ufga": "ptrk_uncontested_field_goals_attempted", "cfga": "ptrk_contested_field_goals_attempted",
    "dfgm": "ptrk_defended_at_rim_field_goals_made", "dfga": "ptrk_defended_at_rim_field_goals_attempted",
    "reb": "ptrk_rebound_chances_total",
}
TRK_COLS = ["trk_games", "trk_dist_per_min", "trk_speed", "trk_touches_per_min", "trk_passes_per_min",
            "trk_uncontested_fga_share", "trk_rim_fga_per_min", "trk_rim_fg_allowed", "trk_reb_chances_per_min"]
DIFF_COLS = TRK_COLS[1:]
SUMS = ["mins", "speed_mins", *SRC]


def _minutes(val):
    if pd.isna(val):
        return np.nan
    m, s = str(val).split(":")
    return int(m) * 60 + int(s)


def _per_game_players(box):
    """(GAME_ID, team) -> per-player totals for games that have tracking data (minutes in seconds)."""
    b = box[box.ptrk_distance.notna()].copy()
    b["mins"] = b.minutes.map(_minutes) / 60
    b = b[b.mins > 0]
    b["speed_mins"] = b.ptrk_speed * b.mins
    b = b.rename(columns={v: k for k, v in SRC.items()})
    return {key: g.set_index("PLAYER_ID")[SUMS] for key, g in b.groupby(["GAME_ID", "team"], sort=False)}


def _window_row(window, players, team, top_n, league_rim_fg, shrink_att):
    have = [gid for gid in window if (gid, team) in players]
    row = dict.fromkeys(TRK_COLS[1:], np.nan)
    row["trk_games"] = len(have)
    if not have:
        return row
    agg = pd.concat([players[(gid, team)] for gid in have]).groupby(level=0).sum()
    rot = agg.nlargest(top_n, "mins")
    tot = rot.sum()
    if tot.mins <= 0:
        return row
    fga = tot.ufga + tot.cfga
    row.update(
        trk_dist_per_min=tot.dist / tot.mins, trk_speed=tot.speed_mins / tot.mins,
        trk_touches_per_min=tot.touches / tot.mins, trk_passes_per_min=tot.passes / tot.mins,
        trk_uncontested_fga_share=tot.ufga / fga if fga > 0 else np.nan,
        trk_rim_fga_per_min=tot.dfga / tot.mins,
        trk_rim_fg_allowed=(tot.dfgm + league_rim_fg * shrink_att) / (tot.dfga + shrink_att),
        trk_reb_chances_per_min=tot.reb / tot.mins,
    )
    return row


def team_tracking_features(box, schedule, window=10, top_n=8, shrink_att=50, league_rim_fg=0.65):
    """One row per (GAME_ID, team). `box` is player box scores (GAME_ID, team, PLAYER_ID, minutes, ptrk_*);
    `schedule` has GAME_ID, date, season, home, away (regular-season games only)."""
    players = _per_game_players(box)
    long = pd.concat([
        schedule[["GAME_ID", "date", "season", "home"]].rename(columns={"home": "team"}),
        schedule[["GAME_ID", "date", "season", "away"]].rename(columns={"away": "team"}),
    ]).assign(date=lambda d: pd.to_datetime(d.date)).sort_values(["team", "date", "GAME_ID"])
    rows = []
    for team, g in long.groupby("team", sort=False):
        ids, seasons = g.GAME_ID.tolist(), g.season.tolist()
        for i, gid in enumerate(ids):
            prior = [ids[j] for j in range(max(0, i - window), i) if seasons[j] == seasons[i]]
            row = _window_row(prior, players, team, top_n, league_rim_fg, shrink_att)
            row.update(GAME_ID=gid, team=team)
            rows.append(row)
    return pd.DataFrame(rows)[["GAME_ID", "team"] + TRK_COLS]


def game_tracking_features(team_feats, schedule):
    out = schedule[["GAME_ID", "home", "away"]]
    for side in ("home", "away"):
        f = team_feats.rename(columns={"team": side, **{c: f"{side}_{c}" for c in TRK_COLS}})
        out = out.merge(f, on=["GAME_ID", side], how="left")
    for c in DIFF_COLS:
        out[f"{c}_diff"] = out[f"home_{c}"] - out[f"away_{c}"]
    return out.drop(columns=["home", "away"])


def run(config=None, inputs=None):
    """Read player box scores + schedule, write per-game tracking features. Returns {"tracking_features": df}."""
    cfg = resolve_step_config(get_config() if config is None else config, "tracking_features")
    paths, opts = cfg["paths"], cfg["tracking_features"]
    box = pd.read_csv(paths["input_boxscores"],
                      usecols=["GAME_ID", "team", "PLAYER_ID", "minutes", "ptrk_speed", *SRC.values()],
                      low_memory=False)
    games = pd.read_csv(paths["input_games"])
    first = box.loc[box.ptrk_distance.notna(), "GAME_ID"].min() // 100000 * 100000
    schedule = games[games.regular.astype(bool) & (games.GAME_ID >= first)]
    team_feats = team_tracking_features(box, schedule, opts["window"], opts["top_n"], opts["shrink_att"],
                                      opts["league_rim_fg"])
    out = game_tracking_features(team_feats, schedule)
    save_output(out, paths["output_features"], cfg.get("cache"))
    print(f"[tracking_features] {len(out)} games -> {paths['output_features']}")
    return {"tracking_features": out}
