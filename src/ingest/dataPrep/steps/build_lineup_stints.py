"""Build five-man lineup stints from V3 play-by-play (nba_data archive layout).

A stint is a maximal stretch of one game where both teams' on-court five stay unchanged. Starters come from the
player box score (`position` is filled for starters); substitutions come from `Substitution` events, which carry only
the player going OUT as `personId` and the player coming IN as a family name in `description` ("SUB: IN FOR OUT").
The incoming name is resolved against that team's box-score roster, preferring players not already on court.

Stints describe THIS game. They are post-game facts: step 6-style lagging must happen before any model sees them.
Games that cannot be reconstructed cleanly (not five starters, unresolved sub, minutes that do not reconcile with the
box score) are returned in the audit table and excluded from the stint table, never patched.
"""

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ..config import get_config, resolve_step_config
    from ..helpers import save_output
except ImportError:
    from ingest.dataPrep.config import get_config, resolve_step_config
    from ingest.dataPrep.helpers import save_output

COUNT_COLS = ["fga", "fta", "tov", "oreb"]
_SUB_RE = re.compile(r"^SUB:\s*(.+?)\s+FOR\s+(.+?)\s*$")
_OFF_RE = re.compile(r"Off:(\d+)")


_SUFFIX = {"jr", "sr", "ii", "iii", "iv"}


def _norm(name, keep_suffix=False):
    """Fold a player name so the feed's variants meet: accents, Jr./III suffixes, ö vs oe."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    words = [w for w in re.sub(r"[^a-z ]", " ", s).split()]
    while words and words[-1] in _SUFFIX and not keep_suffix:
        words.pop()
    return "".join(words).replace("oe", "o").replace("ae", "a").replace("ue", "u")


def _name_keys(r):
    """Every spelling the feed uses for a roster player: last name, 'F. Last', and 'Fi. Last' (same-last-name teammates)."""
    fam, short, fi = r.familyName, r.nameI, f"{str(r.firstName)[:2]} {r.familyName}"
    return {f"{'x' if k else 's'}:{_norm(x, k)}" for x in (fam, short, fi) for k in (False, True)}


def _lookup(names_t, nm, exclude=()):
    """Roster ids for a feed name: the suffix-exact spelling first, then the suffix-free one, so 'Jones Jr.'
    does not collide with a teammate named Jones."""
    for key in (f"x:{_norm(nm, True)}", f"s:{_norm(nm)}"):
        found = [pid for pid in names_t.get(key, []) if pid not in exclude]
        if found:
            return found
    return []


def _period_bounds(period):
    """(elapsed seconds at period start, period length)."""
    if period <= 4:
        return 720 * (period - 1), 720
    return 2880 + 300 * (period - 5), 300


def _remaining(clock):
    m = re.match(r"PT(\d+)M([\d.]+)S", str(clock))
    return int(m.group(1)) * 60 + float(m.group(2))


def _parse_minutes(val):
    if pd.isna(val):
        return 0.0
    m, s = str(val).split(":")
    return int(m) * 60 + int(s)


_ACTIVE = {"Made Shot", "Missed Shot", "Rebound", "Turnover", "Free Throw", "Foul"}


def _period_start(prows, t, roster_ids, names, prev_on):
    """Who started team `t`'s period. The feed often omits substitutions made at a period break, so infer: players
    whose first appearance in the period is a play (or a sub-out) were on court; players whose first appearance is
    a sub-in were not; any gap is filled from the previous period's closing five."""
    first = {}
    for r in prows.itertuples():
        if r.actionType == "Substitution":
            if int(r.teamId) != t:
                continue
            first.setdefault(int(r.personId), "on")
            m = _SUB_RE.match(str(r.description))
            ids = _lookup(names, m.group(1)) if m else []
            for pid in ids:  # a name clash marks every namesake not already seen playing
                first.setdefault(pid, "in")
        elif r.actionType in _ACTIVE and pd.notna(r.personId) and int(r.personId) in roster_ids:
            if "echnical" in f"{r.subType} {r.description}" or "T.FOUL" in str(r.description):
                continue
            first.setdefault(int(r.personId), "on")
    start = {pid for pid, v in first.items() if v == "on"}
    if len(start) > 5:
        return None
    benched = {pid for pid, v in first.items() if v == "in"}
    fill = sorted(prev_on - benched - start)[: 5 - len(start)]
    start |= set(fill)
    return start if len(start) == 5 else None


def _align_ids(pbp, roster, teams):
    """Rename PBP person ids to the box score's id for the same player. Box scores ingested from ESPN give some
    players (2025 rookies) an ESPN id instead of the NBA one; the box id is the repo's key, so it wins. Matched on
    (team, name), and only when exactly one unmatched box player fits."""
    pbp_ids = set(pbp.personId.dropna().astype(int))
    box_ids = set(roster.PLAYER_ID.astype(int))
    swap = {}
    for t in teams:
        loose = roster[(roster.TEAM_ID == t) & ~roster.PLAYER_ID.isin(pbp_ids)]
        seen = pbp[(pbp.teamId == t) & (pbp.personId > 0) & pbp.playerNameI.notna()]
        seen = seen[~seen.personId.isin(box_ids)].drop_duplicates("personId")
        for q in seen.itertuples():
            want = {f"s:{_norm(q.playerName)}", f"s:{_norm(q.playerNameI)}"}
            hit = [int(r.PLAYER_ID) for r in loose.itertuples() if want & _name_keys(r)]
            if len(set(hit)) == 1:
                swap[int(q.personId)] = hit[0]
    return pbp.assign(personId=pbp.personId.replace(swap)) if swap else pbp


def _with_pbp_players(roster, pbp, teams):
    """Add players who appear in the play-by-play but are missing from the box score (ESPN-sourced box scores
    drop some). They carry no minutes to reconcile; they only need an id and a name to be resolved."""
    seen = pbp[pbp.teamId.isin(teams) & (pbp.personId > 0) & pbp.playerNameI.notna()]
    seen = seen[["teamId", "personId", "playerName", "playerNameI"]].drop_duplicates(["teamId", "personId"])
    seen = seen[~seen.personId.isin(roster.PLAYER_ID)]
    if seen.empty:
        return roster
    extra = pd.DataFrame({"GAME_ID": roster.GAME_ID.iloc[0], "TEAM_ID": seen.teamId.values,
                          "PLAYER_ID": seen.personId.values, "firstName": "", "familyName": seen.playerName.values,
                          "nameI": seen.playerNameI.values, "position": None, "minutes": "1:00"})
    return pd.concat([roster, extra], ignore_index=True)


def build_game_stints(pbp, roster):
    """Stints for one game.

    pbp: that game's V3 play-by-play rows in file order. roster: that game's box-score rows
    (TEAM_ID, PLAYER_ID, familyName, position). Returns (stints_df, error_or_None).
    """
    teams = list(roster["TEAM_ID"].unique())
    if len(teams) != 2:
        return None, "teams!=2"
    pbp = _align_ids(pbp, roster, teams)
    roster = _with_pbp_players(roster, pbp, teams)
    names = {t: {} for t in teams}
    roster_ids = {t: set(roster[roster.TEAM_ID == t].PLAYER_ID.astype(int)) for t in teams}
    for r in roster.itertuples():
        if _parse_minutes(r.minutes) <= 0:  # a DNP never enters, so he cannot be the "IN" of a name clash
            continue
        for key in _name_keys(r):
            names[r.TEAM_ID].setdefault(key, []).append(r.PLAYER_ID)
    on = {}
    for t in teams:
        starters = roster[(roster.TEAM_ID == t) & roster.position.notna() & (roster.position != "")]
        if len(starters) == 5:
            on[t] = set(starters.PLAYER_ID.astype(int))
        else:  # older box scores lack the starter flag: infer the opening five from period 1 like any other period
            on[t] = _period_start(pbp[pbp.period == 1], t, roster_ids[t], names[t], set())
            if on[t] is None:
                return None, f"starters={len(starters)}"

    side = pbp.loc[pbp.location.isin(["h", "v"]) & (pbp.teamId != 0), ["location", "teamId"]].drop_duplicates()
    home = side[side.location == "h"].teamId
    if len(home) != 1 or home.iloc[0] not in teams:
        return None, "home team unresolved"
    home = int(home.iloc[0])
    away = [t for t in teams if t != home][0]

    stints, cur = [], None
    off_seen = {}  # personId -> last "Off:n" total seen, to tell offensive rebounds
    score = {home: 0, away: 0}

    def open_stint(period, t_el):
        return {"period": period, "start": t_el, "pts": {home: 0, away: 0},
                "lineup": {t: tuple(sorted(on[t])) for t in teams},
                "cnt": {t: dict.fromkeys(COUNT_COLS, 0) for t in teams}}

    def close(stint, t_el):
        stint["end"] = t_el
        # a zero-length stint can still hold free throws taken between two substitution batches at the same clock
        busy = any(stint["pts"].values()) or any(v for c in stint["cnt"].values() for v in c.values())
        if stint["end"] > stint["start"] or busy:
            stints.append(stint)

    batch = []  # substitution rows sharing one (period, clock); applied together because file order within a
    #             batch is not reliable and chains (A for B, B for C) can appear in either order

    def flush():
        nonlocal cur, batch
        if not batch:
            return None
        p, t_el = batch[0][0], batch[0][1]
        outs = {t: set() for t in teams}
        ins = {t: [] for t in teams}
        for _, _, t, pid, desc in batch:
            m = _SUB_RE.match(str(desc))
            if not m or t not in on:
                return f"bad sub '{desc}'"
            outs[t].add(pid)
            ins[t].append(m.group(1))
        batch = []
        close(cur, t_el)  # the lineup before this substitution batch ends here
        cur = open_stint(p, t_el)
        for t in teams:
            if not outs[t]:
                continue
            if not outs[t] <= on[t]:
                return f"sub-out not on court {sorted(outs[t] - on[t])}"
            staying = on[t] - outs[t]
            new = set(staying)
            for nm in ins[t]:
                cands = _lookup(names[t], nm, staying)
                if len(cands) != 1:
                    return f"sub-in unresolved '{nm}' ({len(cands)})"
                new.add(cands[0])
            if len(new) != 5:
                return f"lineup size {len(new)} after subs"
            on[t] = new
        cur["lineup"] = {x: tuple(sorted(on[x])) for x in teams}
        return None

    for r in pbp.itertuples():
        if pd.isna(r.period):  # period start/end rows stay: the closing one carries the final score
            continue
        p = int(r.period)
        p0, plen = _period_bounds(p)
        t_el = p0 + plen - _remaining(r.clock)
        at = r.actionType
        if batch and (at != "Substitution" or batch[0][:2] != (p, t_el)):
            err = flush()
            if err:
                return None, err
        if cur is None:
            cur = open_stint(p, p0)
        elif p != cur["period"]:
            close(cur, _period_bounds(cur["period"])[0] + _period_bounds(cur["period"])[1])
            prows = pbp[pbp.period == p]
            for t in teams:
                start = _period_start(prows, t, roster_ids[t], names[t], on[t])
                if start is None:
                    return None, f"period {p} start lineup unresolved"
                on[t] = start
            cur = open_stint(p, p0)
        if at == "Substitution":
            batch.append((p, t_el, int(r.teamId), int(r.personId), r.description))
            continue
        # score and box-style counts attributed to the stint in force
        if pd.notna(r.scoreHome) and pd.notna(r.scoreAway) and (r.scoreHome + r.scoreAway) >= sum(score.values()):
            cur["pts"][home] += int(r.scoreHome) - score[home]
            cur["pts"][away] += int(r.scoreAway) - score[away]
            score = {home: int(r.scoreHome), away: int(r.scoreAway)}
        t = int(r.teamId) if pd.notna(r.teamId) else 0
        if t in cur["cnt"]:
            c = cur["cnt"][t]
            if at in ("Made Shot", "Missed Shot"):
                c["fga"] += 1
            elif at == "Free Throw":
                c["fta"] += 1
            elif at == "Turnover":
                c["tov"] += 1
            elif at == "Rebound" and r.personId:
                m = _OFF_RE.search(str(r.description))
                if m:
                    n = int(m.group(1))
                    if n > off_seen.get(r.personId, 0):
                        c["oreb"] += 1
                    off_seen[r.personId] = n
    err = flush()
    if err:
        return None, err
    if cur is None:
        return None, "no events"
    p = cur["period"]
    close(cur, _period_bounds(p)[0] + _period_bounds(p)[1])
    if not stints or (score[home] + score[away]) == 0:
        return None, "no scoring"

    rows = []
    for i, s in enumerate(stints):
        for t, opp in ((home, away), (away, home)):
            rows.append({
                "stint": i, "TEAM_ID": t, "OPP_TEAM_ID": opp, "is_home": int(t == home), "period": s["period"],
                "start_sec": s["start"], "end_sec": s["end"], "seconds": s["end"] - s["start"],
                "lineup_key": "-".join(map(str, s["lineup"][t])),
                "opp_lineup_key": "-".join(map(str, s["lineup"][opp])),
                "pts_for": s["pts"][t], "pts_against": s["pts"][opp],
                **{f"{k}_for": s["cnt"][t][k] for k in COUNT_COLS},
                **{f"{k}_against": s["cnt"][opp][k] for k in COUNT_COLS},
            })
    return pd.DataFrame(rows), None


def _reconcile(stints, roster, tol_sec):
    """Max |stint seconds - box-score seconds| over rostered players with minutes, plus final-score check."""
    secs = {}
    for r in stints.itertuples():
        for pid in r.lineup_key.split("-"):
            secs[int(pid)] = secs.get(int(pid), 0.0) + r.seconds
    worst = 0.0
    for r in roster.itertuples():
        box = _parse_minutes(r.minutes)
        worst = max(worst, abs(secs.get(int(r.PLAYER_ID), 0.0) - box))
    return worst, worst <= tol_sec


def remap_game_ids(pbp, games, known_ids):
    """Give PBP games the repo's GAME_ID when it differs from the NBA's.

    Current-season games ingested from ESPN carry synthetic ids (2025-26 -> 22590001...), so the NBA's own ids in the
    PBP archive never meet the box scores. Match on (home, away, final score), which is unique within a season; a
    game with zero or several matches keeps its id and shows up in the audit as "no box score".
    """
    pbp = pbp.copy()
    pbp["gameId"] = pbp["gameId"].astype(int)
    missing = sorted(set(pbp.gameId) - set(known_ids))
    if not missing:
        return pbp
    sub = pbp[pbp.gameId.isin(missing)]
    home = (sub[sub.location == "h"].dropna(subset=["teamTricode"]).groupby("gameId").teamTricode.first())
    away = (sub[sub.location == "v"].dropna(subset=["teamTricode"]).groupby("gameId").teamTricode.first())
    fin = sub.dropna(subset=["scoreHome", "scoreAway"]).groupby("gameId")[["scoreHome", "scoreAway"]].last()
    keys = pd.DataFrame({"home": home, "away": away}).join(fin).dropna().reset_index()
    games = games[~games.GAME_ID.isin(set(pbp.gameId) - set(missing))].rename(columns={"GAME_ID": "box_id"})
    games = games.assign(scoreHome=games.score_home, scoreAway=games.score_away)
    m = keys.merge(games[["box_id", "home", "away", "scoreHome", "scoreAway"]], on=["home", "away", "scoreHome", "scoreAway"])
    m = m[~m.gameId.duplicated(keep=False) & ~m.box_id.duplicated(keep=False)]
    pbp["gameId"] = pbp.gameId.map(dict(zip(m.gameId, m.box_id))).fillna(pbp.gameId).astype(int)
    return pbp


def build_stints(pbp, boxscores, tol_sec=60):
    """All games in `pbp`. Returns (stints, audit) with one audit row per game."""
    pbp = pbp.copy()
    pbp["gameId"] = pbp["gameId"].astype(int)
    box = boxscores[boxscores.GAME_ID.isin(pbp.gameId.unique())]
    out, audit = [], []
    for gid, g in pbp.groupby("gameId", sort=False):
        roster = box[box.GAME_ID == gid]
        if roster.empty:
            audit.append({"GAME_ID": gid, "ok": False, "reason": "no box score", "max_min_gap_sec": np.nan})
            continue
        st, err = build_game_stints(g, roster)
        if err:
            audit.append({"GAME_ID": gid, "ok": False, "reason": err, "max_min_gap_sec": np.nan})
            continue
        gap, ok = _reconcile(st[st.is_home == 1], roster[roster.TEAM_ID == st[st.is_home == 1].TEAM_ID.iloc[0]], tol_sec)
        gap_a, ok_a = _reconcile(st[st.is_home == 0], roster[roster.TEAM_ID == st[st.is_home == 0].TEAM_ID.iloc[0]], tol_sec)
        gap = max(gap, gap_a)
        if ok and ok_a:
            st.insert(0, "GAME_ID", gid)
            out.append(st)
        audit.append({"GAME_ID": gid, "ok": ok and ok_a, "reason": "" if ok and ok_a else "minutes mismatch",
                      "max_min_gap_sec": gap})
    stints = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return stints, pd.DataFrame(audit)


def run(config=None, inputs=None):
    """Read PBP archives + player box scores, write stints and an audit table.

    Returns {"lineup_stints": df, "lineup_stint_audit": df}.
    """
    cfg = resolve_step_config(get_config() if config is None else config, "stints")
    paths, opts = cfg["paths"], cfg["stints"]
    seasons = opts["seasons"]
    pbp = pd.concat(
        [pd.read_csv(Path(paths["pbp_dir"]) / f"nbastatsv3_{y}.csv", low_memory=False) for y in seasons],
        ignore_index=True,
    )
    cols = ["GAME_ID", "TEAM_ID", "PLAYER_ID", "firstName", "familyName", "nameI", "position", "minutes"]
    box = (inputs or {}).get("player_boxscores")
    box = pd.read_csv(paths["input_boxscores"], usecols=cols, low_memory=False) if box is None else box[cols]
    games = pd.read_csv(paths["input_games"], usecols=["GAME_ID", "home", "away", "score_home", "score_away"])
    pbp = remap_game_ids(pbp, games, box.GAME_ID.unique())
    stints, audit = build_stints(pbp, box, tol_sec=opts["minutes_tolerance_sec"])
    save_output(stints, paths["output_stints"], cfg.get("cache"))
    save_output(audit, paths["output_audit"], cfg.get("cache"))
    ok = int(audit.ok.sum()) if len(audit) else 0
    print(f"[stints] {ok}/{len(audit)} games reconstructed; audit -> {paths['output_audit']}")
    return {"lineup_stints": stints, "lineup_stint_audit": audit}
