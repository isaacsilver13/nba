"""Compare the current daily model with a history-only rotation proxy.

Run from a local data checkout, for example::

    python tools/season_rotation_poc.py --date 2026-10-22 --game CLE:PHI --game DEN:OKC

`--game` is AWAY:HOME. This is an opt-in research run: it reads the same
completed-game tables as predict_today, trains both variants, and does not
write or publish a daily pick. No current-game participants or injury feed are
used. A prior-ten-game player pool is a historical proxy, not a known lineup.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import experiment as ex  # noqa: E402
import predict_today as pt  # noqa: E402
import run_features as rf  # noqa: E402
from encompass_check import fit_probit_scale  # noqa: E402

DATA = Path(os.environ.get("NBA_DATA_DIR", ROOT / "data"))
FEATURE_VERSION = "history-rotation-v2"
ROTATION_GAMES = 10


def _gid(values):
    return pd.to_numeric(values.astype(str).str.replace(r"\.0$", "", regex=True), errors="raise").astype("int64")


def rotation_history(games, boxscores):
    """One row per team-game from *earlier* team games, including at a new season.

    Within a team/date, calculate before appending any of that date's boxscores.
    That also protects the rare case of two games recorded on one date.
    """
    games = games[["GAME_ID", "date", "season", "home", "away"]].copy()
    games["GAME_ID"] = _gid(games["GAME_ID"])
    games["date"] = pd.to_datetime(games["date"]).dt.normalize()
    team_games = pd.concat([
        games[["GAME_ID", "date", "season", "home"]].rename(columns={"home": "team"}),
        games[["GAME_ID", "date", "season", "away"]].rename(columns={"away": "team"}),
    ], ignore_index=True).sort_values(["team", "date", "GAME_ID"], kind="stable")

    b = boxscores[["GAME_ID", "team", "PLAYER_ID", "minutes", "points"]].copy()
    b["GAME_ID"] = _gid(b["GAME_ID"])
    b = b.drop_duplicates(["GAME_ID", "team", "PLAYER_ID"], keep="first")
    b["minutes"] = pd.to_numeric(b["minutes"].fillna("").astype(str).str.split(":").str[0], errors="coerce").fillna(0)
    b["points"] = pd.to_numeric(b["points"], errors="coerce").fillna(0)
    by_game = {
        (int(gid), team): list(zip(part["PLAYER_ID"], part["minutes"], part["points"]))
        for (gid, team), part in b.groupby(["GAME_ID", "team"], sort=False)
    }

    rows = []
    for team, dates in team_games.groupby("team", sort=False):
        recent = deque(maxlen=ROTATION_GAMES)
        for day, matches in dates.groupby("date", sort=False):
            season = int(matches["season"].iloc[0])
            minutes, points = defaultdict(float), defaultdict(float)
            observed = [(d, s, roster) for d, s, roster in recent if roster is not None]
            coverage = len(observed) / len(recent) if recent else np.nan
            for _, _, roster in observed:
                for pid, mins, pts in roster:
                    minutes[pid] += float(mins)
                    points[pid] += float(pts)
            leaders = sorted(minutes, key=lambda pid: (-minutes[pid], str(pid)))[:8]
            value = (sum(points[pid] for pid in leaders) / (len(observed) * 8)
                     if len(observed) >= 3 and coverage >= 0.8 and len(leaders) >= 8 else np.nan)
            prior_season = recent[-1][1] if recent else np.nan
            gap = (day - observed[-1][0]).days if observed else np.nan
            same_season_games = sum(s == season for _, s, _ in observed)
            for gid in matches["GAME_ID"]:
                rows.append((int(gid), team, value, prior_season, gap, same_season_games, coverage))
            for gid in matches["GAME_ID"]:
                recent.append((day, season, by_game.get((int(gid), team))))
    return pd.DataFrame(rows, columns=["GAME_ID", "team", "rotation_points", "prior_season",
                                       "gap_days", "same_season_games", "rotation_coverage"])


def add_poc_features(frame, summary):
    """Attach team-history signals without inspecting this game's box score."""
    cols = ["rotation_points", "prior_season", "gap_days", "same_season_games", "rotation_coverage"]
    left = summary.rename(columns={"team": "home", **{c: "home_" + c for c in cols}})
    right = summary.rename(columns={"team": "away", **{c: "away_" + c for c in cols}})
    out = frame.merge(left, on=["GAME_ID", "home"], how="left", validate="many_to_one")
    out = out.merge(right, on=["GAME_ID", "away"], how="left", validate="many_to_one")
    both_current = out["home_prior_season"].eq(out["season"]) & out["away_prior_season"].eq(out["season"])
    out["poc_prev_minutes_out_same_season"] = out["prev_game_minutes_out"].where(both_current)
    out["poc_rotation_points_diff"] = out["home_rotation_points"] - out["away_rotation_points"]
    out["poc_rotation_gap_days_max"] = out[["home_gap_days", "away_gap_days"]].max(axis=1)
    out["poc_rotation_same_season_games_min"] = out[
        ["home_same_season_games", "away_same_season_games"]].min(axis=1)
    out["poc_rotation_coverage_min"] = out[["home_rotation_coverage", "away_rotation_coverage"]].min(axis=1)
    return out


def _predict_variant(done, slate, features):
    done = done.copy()
    slate = slate.copy()
    for table in (done, slate):
        table[features] = table[features].apply(pd.to_numeric, errors="coerce")
    oos, margins = pt.walk_forward(done, features, slate)
    known = oos.notna()
    actual = done.loc[known, "home_margin"].to_numpy()
    pred_oos = oos.loc[known].to_numpy()
    sigma = float(np.std(actual - pred_oos, ddof=1))
    probabilities = fit_probit_scale(pred_oos, (actual > 0).astype(float), margins)
    valid = done.loc[known, ["GAME_DATE", "season"]]
    squared_error = (actual - pred_oos) ** 2
    groups = {
        "all": np.ones(len(valid), dtype=bool),
        "oct_nov": valid["GAME_DATE"].dt.month.isin([10, 11]).to_numpy(),
        "season_2026": valid["season"].eq(2026).to_numpy(),
    }
    metrics = {name: {"games": int(mask.sum()), "rmse": round(float(np.sqrt(squared_error[mask].mean())), 3)}
               for name, mask in groups.items() if mask.any()}
    return margins, probabilities, sigma, metrics


def compare(date, schedule):
    D = pd.Timestamp(date).normalize()
    games_path = DATA / "processed" / "nba_games_with_game_id.csv"
    box_path = DATA / "processed" / "all_boxscores.csv"
    fingerprints = {}
    for path in [Path(__file__), ROOT / "experiment.py", ROOT / "tools" / "predict_today.py",
                 ROOT / "tools" / "run_features.py", games_path, box_path,
                 DATA / "processed" / "df_model_3.csv"]:
        fingerprints[str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)] = hashlib.sha256(
            path.read_bytes()).hexdigest()[:12]
    history = pt.load_history()
    done = history[history["GAME_DATE"] < D].copy()
    if done.empty:
        raise ValueError("no completed-game history before slate")
    slate = rf.slate_rows(D, schedule).rename(columns={"date_x": "GAME_DATE"})
    if len(slate) != len(schedule):
        raise ValueError("slate rows do not match supplied games")
    slate = ex.add_oliver_features(slate)

    games = pd.read_csv(games_path, parse_dates=["date"])
    games = games[games["date"] < D]
    future = schedule.assign(date=D, season=int(slate["season"].iloc[0]))
    games = pd.concat([games, future], ignore_index=True)
    boxscores = pd.read_csv(box_path, usecols=["GAME_ID", "team", "PLAYER_ID", "minutes", "points"], low_memory=False)
    summary = rotation_history(games, boxscores)
    done = add_poc_features(done, summary)
    slate = add_poc_features(slate, summary)

    baseline = [c for c in ex.build_features(done) if c not in pt.EXCLUDE]
    season_guard = sorted((set(baseline) - {"prev_game_minutes_out"}) | {
        "poc_prev_minutes_out_same_season", "poc_rotation_gap_days_max",
        "poc_rotation_same_season_games_min",
    })
    candidate = sorted((set(season_guard) - {"top8_points_diff"}) | {"poc_rotation_points_diff"})
    results = {}
    for label, features in (("baseline", baseline), ("season_guard", season_guard),
                            ("history_rotation_poc", candidate)):
        margins, probabilities, sigma, metrics = _predict_variant(done, slate, features)
        results[label] = {"features": len(features), "walk_forward": metrics, "sigma": round(sigma, 3),
                          "games": [{"away": away, "home": home,
                                     "home_margin": round(float(margin), 2),
                                     "p_home_win": round(float(prob), 4)}
                                    for away, home, margin, prob in
                                    zip(slate["away"], slate["home"], margins, probabilities)]}
    return {"slate_date": f"{D:%Y-%m-%d}", "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_through": f"{done['GAME_DATE'].max():%Y-%m-%d}",
            "feature_version": FEATURE_VERSION, "input_sha256_12": fingerprints,
            "input_games": len(games) - len(schedule), "input_boxscore_rows": len(boxscores),
            "slate_features": slate[["away", "home", "prev_game_minutes_out", "poc_prev_minutes_out_same_season",
                                      "top8_points_diff", "poc_rotation_points_diff", "poc_rotation_gap_days_max",
                                      "poc_rotation_same_season_games_min", "poc_rotation_coverage_min"]].replace(
                                          {np.nan: None}).to_dict("records"),
            "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", required=True, help="NBA Eastern schedule date, YYYY-MM-DD")
    parser.add_argument("--game", action="append", required=True, help="AWAY:HOME team codes; repeat for each game")
    parser.add_argument("--out", type=Path, help="optional local JSON path")
    args = parser.parse_args()
    parsed = []
    for spec in args.game:
        away, sep, home = spec.upper().partition(":")
        if not sep or not away or not home:
            parser.error(f"invalid --game {spec!r}; expected AWAY:HOME")
        parsed.append((away, home))
    if len(set(parsed)) != len(parsed):
        parser.error("duplicate --game")
    schedule = pd.DataFrame(parsed, columns=["away", "home"])
    D = pd.Timestamp(args.date)
    season = D.year + (D.month >= 7)
    game_id_base = 20_000_000 + (season - 2001) * 100_000 + 99_000
    schedule["GAME_ID"] = [game_id_base + i for i in range(1, len(schedule) + 1)]
    result = compare(args.date, schedule)
    payload = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
