"""Append completed regular-season games (scores, closing spread, player box scores) from ESPN to data/processed,
then rebuild the feature table, so tools/predict_today.py sees every game through yesterday.

    python tools/ingest_daily.py                  # from the last ingested date through yesterday (US/Eastern)
    python tools/ingest_daily.py --no-features    # append only; run tools/run_features.py yourself

Why ESPN: stats.nba.com and cdn.nba.com refuse this PC (Akamai 403 / timeouts, 2026-10-21). ESPN's public JSON has the
schedule, final scores, a closing spread (ESPN BET / DraftKings, core-API odds) and traditional player box scores.
team_boxscores_v3.csv gets the NBA team metrics the model reads (four factors, offensive/defensive rating), computed
from ESPN's team totals (v3_rows); the other V3 metrics stay NaN for these games (step 3a).

- GAME_ID: ESPN has no NBA game id, so new games get 002<yy>9<nnnn> (season's ending year, sequence from 1). Real NBA
  ids end at <yy>01230, so these never collide; a game is "already ingested" by (date, home, away).
- PLAYER_ID: the NBA person id when the ESPN name matches exactly one player in earlier NBA box scores, else
  ESPN_ID_OFFSET + ESPN athlete id, so a player keeps one id and their rolling form carries across the switch.
- minutes: ESPN gives whole minutes, written "M:00" (NBA gives M:SS); a team's sum can be off 240 by a minute or two.
- Box scores are appended before games, so a run that dies in between re-appends the same rows next time and the
  feature steps' dedupe_player_games drops the copies. V3 rows come last and are written for any ESPN game that
  lacks them, so a run that dies before them fills them in next time.
Raw ESPN responses are cached under data/cache/espn/ once a game is final.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NBA_DATA_DIR", ROOT / "data"))
SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
ESPN_TO_NBA = {"GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}
NBA_TEAMS = {"ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM",
             "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"}
ESPN_ID_OFFSET = 10**8  # NBA person ids are < 2e6
RESCAN_DAYS = 7  # re-read the last week too, for a game that was postponed or not final at the last run
# NBA V3 value ~ a + b * box-score estimate, least squares on 2022-24 NBA games (tested on 2024-25, see v3_rows):
POSS_FIT = (5.6798, 0.9330)  # possessions vs mean of the two teams' FGA + 0.44 FTA - OREB + TOV
OREB_FIT = (0.0628, 0.9068)  # OREB% vs OREB / (OREB + opponent DREB)
# ESPN box score key -> all_boxscores.csv column ("a-b" keys hold made-attempted)
STAT_COLS = {"points": "points", "rebounds": "reboundsTotal", "assists": "assists", "turnovers": "turnovers",
             "steals": "steals", "blocks": "blocks", "offensiveRebounds": "reboundsOffensive",
             "defensiveRebounds": "reboundsDefensive", "fouls": "foulsPersonal", "plusMinus": "plusMinusPoints"}
PAIR_COLS = {"fieldGoalsMade-fieldGoalsAttempted": ("fieldGoalsMade", "fieldGoalsAttempted"),
             "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("threePointersMade", "threePointersAttempted"),
             "freeThrowsMade-freeThrowsAttempted": ("freeThrowsMade", "freeThrowsAttempted")}


def get_json(url, cache=None, params=None, retries=4):
    if cache is not None and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    j = r.json()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(j), encoding="utf-8")
    return j


def tricode(abbr):
    code = ESPN_TO_NBA.get(abbr, abbr)
    if code not in NBA_TEAMS:
        raise ValueError(f"unknown ESPN team abbreviation {abbr!r}: add it to ESPN_TO_NBA")
    return code


def scoreboard(day, final_only=True):
    """Regular-season games on `day` (the NBA's US/Eastern date): DataFrame espn_id, date, season, home, away,
    score_home, score_away, ARENA_NAME, final. Excludes the NBA Cup final (not a regular-season game), the All-Star
    games, play-in and playoffs, as LeagueGameFinder's "Regular Season" does."""
    cache = DATA / "cache" / "espn" / "scoreboard" / f"{day:%Y%m%d}.json"
    today = datetime.now(ZoneInfo("America/New_York")).date()
    j = get_json(f"{SITE}/scoreboard", cache if cache.exists() else None, {"dates": f"{day:%Y%m%d}", "limit": 50})
    rows = []
    for e in j.get("events", []):
        c = e["competitions"][0]
        if e["season"]["type"] != 2 or c.get("type", {}).get("abbreviation", "STD") != "STD":
            continue
        side = {t["homeAway"]: t for t in c["competitors"]}
        rows.append({"espn_id": e["id"], "date": pd.Timestamp(day), "season": e["season"]["year"],
                     "home": tricode(side["home"]["team"]["abbreviation"]),
                     "away": tricode(side["away"]["team"]["abbreviation"]),
                     "score_home": pd.to_numeric(side["home"].get("score"), errors="coerce"),
                     "score_away": pd.to_numeric(side["away"].get("score"), errors="coerce"),
                     "ARENA_NAME": c.get("venue", {}).get("fullName"),
                     "final": c["status"]["type"]["name"] == "STATUS_FINAL"})
    done = {"STATUS_FINAL", "STATUS_POSTPONED", "STATUS_CANCELED"}
    if not cache.exists() and day < today and all(e["competitions"][0]["status"]["type"]["name"] in done
                                                   for e in j.get("events", [])):
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(j), encoding="utf-8")
    g = pd.DataFrame(rows, columns=["espn_id", "date", "season", "home", "away", "score_home", "score_away",
                                    "ARENA_NAME", "final"])
    return g[g["final"]] if final_only else g


def game_id_base(season):
    """GAME_IDs for ESPN-sourced games of `season` (ending year) are base+1, base+2, ...: 2026 -> 22590000."""
    return 20000000 + (season - 2001) * 100000 + 90000


def closing_spread(espn_id):
    """The home team's closing handicap (-6.5 = home favoured by 6.5), or NaN when ESPN has no line."""
    j = get_json(f"{CORE}/events/{espn_id}/competitions/{espn_id}/odds",
                 DATA / "cache" / "espn" / "odds" / f"{espn_id}.json")
    lines = [i["spread"] for i in sorted(j.get("items", []), key=lambda i: i.get("provider", {}).get("priority", 99))
             if i.get("spread") is not None]
    return float(lines[0]) if lines else np.nan


def _norm(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return " ".join("".join(ch for ch in s if ch.isalnum() or ch == " ").split())


def summary(espn_id):
    return get_json(f"{SITE}/summary", DATA / "cache" / "espn" / "summary" / f"{espn_id}.json", {"event": espn_id})


def v3_rows(espn_id, game_id, team_ids):
    """team_boxscores_v3.csv rows (the columns the model reads) from ESPN's team totals, in the NBA's units/rounding.

    eFG% and FT rate are exact. TOV% = all turnovers (team ones too) / possessions, as the NBA's. The NBA counts
    possessions and OREB% from play-by-play (team rebounds included), which ESPN does not report, so those use
    POSS_FIT / OREB_FIT; ratings use the fitted possessions."""
    sides = {}
    for team in summary(espn_id)["boxscore"]["teams"]:
        s = {x["name"]: x["displayValue"] for x in team["statistics"]}
        t = {k: float(v) for k, v in s.items() if k in ("offensiveRebounds", "defensiveRebounds", "turnovers",
                                                         "totalTurnovers")}
        for k, (made, att) in PAIR_COLS.items():
            t[made], t[att] = map(float, s[k].split("-"))
        t["pts"] = 2 * t["fieldGoalsMade"] + t["threePointersMade"] + t["freeThrowsMade"]
        t["poss_est"] = t["fieldGoalsAttempted"] + 0.44 * t["freeThrowsAttempted"] - t["offensiveRebounds"] + t["turnovers"]
        sides[tricode(team["team"]["abbreviation"])] = t
    (h, th), (a, ta) = sides.items()
    poss = POSS_FIT[0] + POSS_FIT[1] * (th["poss_est"] + ta["poss_est"]) / 2
    rows = []
    for code, t, o in ((h, th, ta), (a, ta, th)):
        ortg, drtg = 100 * t["pts"] / poss, 100 * o["pts"] / poss
        rows.append({
            "GAME_ID": game_id, "TEAM_ID": team_ids[code], "TEAM_ABBREVIATION": code,
            "ff_effective_field_goal_percentage": round((t["fieldGoalsMade"] + 0.5 * t["threePointersMade"])
                                                        / t["fieldGoalsAttempted"], 3),
            "ff_free_throw_attempt_rate": round(t["freeThrowsAttempted"] / t["fieldGoalsAttempted"], 3),
            "ff_team_turnover_percentage": round(t["totalTurnovers"] / poss, 3),
            "ff_offensive_rebound_percentage": round(OREB_FIT[0] + OREB_FIT[1] * t["offensiveRebounds"]
                                                     / (t["offensiveRebounds"] + o["defensiveRebounds"]), 3),
            "adv_offensive_rating": round(ortg, 1), "adv_defensive_rating": round(drtg, 1),
            "adv_net_rating": round(ortg - drtg, 1), "adv_possessions": round(poss)})
    return rows


def write_v3(games, team_ids, proc):
    """Append V3 rows for every ESPN-sourced game in `games` that has none (backfills, and heals a run that died
    after the games table was written)."""
    gid = pd.to_numeric(games["GAME_ID"])
    have = set(pd.read_csv(proc / "team_boxscores_v3.csv", usecols=["GAME_ID"])["GAME_ID"])
    todo = games[(gid % 100000 > 90000) & ~gid.isin(have)]
    if todo.empty:
        return 0
    sb = pd.concat([scoreboard(d.date()) for d in pd.to_datetime(todo["date"].unique())], ignore_index=True)
    todo = todo.merge(sb[["date", "home", "away", "espn_id"]], on=["date", "home", "away"], validate="1:1")
    append_csv(pd.DataFrame([r for e, g in zip(todo["espn_id"], todo["GAME_ID"]) for r in v3_rows(e, g, team_ids)]),
               proc / "team_boxscores_v3.csv")
    print(f"appended V3 team rows for {len(todo)} games")
    return len(todo)


def box_rows(espn_id, game_id, team_ids):
    """all_boxscores.csv rows for one game (every listed player, DNPs included, as the NBA box score lists them)."""
    j = summary(espn_id)
    rows = []
    for team in j["boxscore"]["players"]:
        code = tricode(team["team"]["abbreviation"])
        st = team["statistics"][0]
        for a in st["athletes"]:
            if "id" not in a["athlete"]:
                continue  # an ESPN stub with no id or name (seen once, a coach's-decision DNP)
            name = a["athlete"]["displayName"]
            first, _, last = name.partition(" ")
            r = {"GAME_ID": game_id, "TEAM_ID": team_ids[code], "team": code, "espn_player_id": int(a["athlete"]["id"]),
                 "firstName": first, "familyName": last, "nameI": a["athlete"].get("shortName"),
                 "position": a["athlete"].get("position", {}).get("abbreviation") if a.get("starter") else None,
                 "jerseyNum": pd.to_numeric(a["athlete"].get("jersey"), errors="coerce"), "endpoint_family": "espn"}
            stats = dict(zip(st["keys"], a.get("stats") or []))
            if a.get("didNotPlay") or not stats:
                reason = (a.get("reason") or "").strip()
                # step 4 reads any non-coach reason as a pregame-known absence, as with the NBA's "DND - Injury/Illness"
                r["comment"] = "DNP - Coach's Decision" if not reason or "COACH" in reason.upper() else f"DNP - {reason.title()}"
                stats = {}
            elif not str(stats.get("minutes", "")).isdigit():
                continue  # "--" minutes and no DNP reason: not in the NBA's box score either
            else:
                r["minutes"] = f"{int(stats['minutes'])}:00"
            for k, col in STAT_COLS.items():
                r[col] = pd.to_numeric(str(stats.get(k, "0")).lstrip("+"), errors="coerce") if stats else 0.0
            for k, (made, att) in PAIR_COLS.items():
                m, _, n = str(stats.get(k, "0-0")).partition("-")
                r[made], r[att] = float(m or 0), float(n or 0)
            rows.append(r)
    return rows


def player_ids(box_new, box_old):
    """ESPN athlete id -> PLAYER_ID: the NBA id when the normalised name is unique among NBA-sourced rows."""
    nba = box_old[box_old["endpoint_family"] != "espn"].drop_duplicates("PLAYER_ID", keep="last")
    nba = nba.assign(key=(nba["firstName"].fillna("") + " " + nba["familyName"].fillna("")).map(_norm))
    nba = nba[~nba["key"].duplicated(keep=False)].set_index("key")["PLAYER_ID"]
    keys = (box_new["firstName"] + " " + box_new["familyName"]).map(_norm)
    return keys.map(nba).fillna(box_new["espn_player_id"] + ESPN_ID_OFFSET).astype("int64")


def append_csv(df, path):
    """Append rows in the file's own column order (a missing column is written empty)."""
    cols = pd.read_csv(path, nrows=0).columns
    df.reindex(columns=cols).to_csv(path, mode="a", header=False, index=False)


def ingest(through, data_dir=DATA):
    proc = data_dir / "processed"
    games = pd.read_csv(proc / "nba_games_with_game_id.csv", parse_dates=["date"], date_format="%m/%d/%Y")
    box_old = pd.read_csv(proc / "all_boxscores.csv", usecols=["TEAM_ID", "team", "PLAYER_ID", "firstName",
                                                                "familyName", "endpoint_family"], low_memory=False)
    team_ids = box_old.drop_duplicates("team", keep="last").set_index("team")["TEAM_ID"].to_dict()
    seen = set(zip(games["date"], games["home"], games["away"]))
    days = pd.date_range(games["date"].max() - pd.Timedelta(days=RESCAN_DAYS), through)
    new = pd.concat([scoreboard(d.date()) for d in days], ignore_index=True)
    new = new[[k not in seen for k in zip(new["date"], new["home"], new["away"])]]
    if new.empty:
        print(f"no new completed games through {through}")
        return write_v3(games, team_ids, proc)
    new = new.sort_values(["date", "espn_id"]).reset_index(drop=True)
    gid = pd.to_numeric(games["GAME_ID"])
    for season, idx in new.groupby("season").groups.items():
        base = game_id_base(season)
        last = gid[(gid > base) & (gid < base + 10000)].max()  # this season's earlier ESPN games
        new.loc[idx, "GAME_ID"] = (base if pd.isna(last) else last) + 1 + np.arange(len(idx))
    new["GAME_ID"] = new["GAME_ID"].astype("int64")

    home_line = new["espn_id"].map(closing_spread)
    new["spread"] = home_line.abs()
    new["whos_favored"] = np.where(home_line.isna(), None, np.where(home_line <= 0, "home", "away"))
    new["regular"], new["id_spread"] = True, 1  # id_spread only gates step 2's filter (0/1 kept)

    box = pd.DataFrame([r for e, g in zip(new["espn_id"], new["GAME_ID"]) for r in box_rows(e, g, team_ids)])
    box["PLAYER_ID"] = player_ids(box, box_old)
    lacking = set(new["GAME_ID"]) - set(box.loc[box["minutes"].notna(), "GAME_ID"])
    if lacking:
        raise RuntimeError(f"ESPN has no box score yet for GAME_IDs {sorted(lacking)}; rerun later")

    append_csv(box, proc / "all_boxscores.csv")  # box scores first: see the module docstring
    append_csv(new.assign(date=new["date"].map(lambda d: f"{d.month}/{d.day}/{d.year}")),  # the table's format
               proc / "nba_games_with_game_id.csv")
    n_v3 = write_v3(pd.concat([games, new], ignore_index=True), team_ids, proc)
    print(f"appended {len(new)} games ({new['date'].iloc[0]:%Y-%m-%d} .. {new['date'].iloc[-1]:%Y-%m-%d}), "
          f"{len(box):,} player rows; "
          f"{home_line.isna().sum()} without a closing line, "
          f"{(box['PLAYER_ID'] >= ESPN_ID_OFFSET).sum()} player rows with an ESPN-only id")
    return len(new) + n_v3


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--through", help="last date to ingest, YYYY-MM-DD (default: yesterday, US/Eastern)")
    ap.add_argument("--no-features", action="store_true", help="skip the tools/run_features.py rebuild")
    a = ap.parse_args()
    through = pd.Timestamp(a.through) if a.through else \
        pd.Timestamp(datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1))
    if ingest(through) and not a.no_features:
        # the step configs' paths are relative ("data/processed/..."), so run from the data dir's parent
        subprocess.run([sys.executable, str(ROOT / "tools" / "run_features.py")], cwd=DATA.parent, check=True)


if __name__ == "__main__":
    main()
