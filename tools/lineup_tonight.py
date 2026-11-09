"""Compare the daily baseline with the +lineup variant on a live slate (opt-in research run; publishes nothing).

    NBA_DATA_DIR=<data dir> python tools/lineup_tonight.py --date 2026-10-22

Lineup features for a slate game come from the same window rule as history: only that team's earlier SAME-season games
that have stints. Stints exist only for games in lineup_stints.csv, so a team whose season starts tonight (or whose
earlier games have no stints) gets no lineup features, and the +lineup model then differs from the baseline only
through how it treats those missing values. The output says which case each game is in.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import experiment as ex  # noqa: E402
import lineup_tracking_ablation as ab  # noqa: E402
import predict_today as pt  # noqa: E402
import run_features as rf  # noqa: E402
from encompass_check import fit_probit_scale  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    D = pd.Timestamp(args.date).normalize()

    schedule = pt.live_schedule(D)
    table = ab.load_table()
    done = table[table["GAME_DATE"] < D].copy()
    slate = rf.slate_rows(D, schedule).rename(columns={"date_x": "GAME_DATE"})
    slate = ex.add_oliver_features(slate)

    # lineup features for the slate: none of tonight's teams has stints this season unless lineup_stints.csv says so
    stints = pd.read_csv(ab.DATA / "processed" / "lineup_stints.csv", usecols=["GAME_ID", "TEAM_ID"])
    played = pd.read_csv(ab.DATA / "processed" / "nba_games_with_game_id.csv", usecols=["GAME_ID", "date", "home", "away"])
    played["date"] = pd.to_datetime(played["date"])
    season_start = played[played.GAME_ID >= int(done.GAME_ID.max() // 100000 * 100000)]
    with_stints = set(stints.GAME_ID)
    notes = []
    for g in slate.itertuples():
        n = {}
        for side in ("home", "away"):
            team = getattr(g, side)
            games = season_start[((season_start.home == team) | (season_start.away == team)) & (season_start.date < D)]
            n[team] = {"season_games_before": int(len(games)),
                       "with_stints": int(games.GAME_ID.isin(with_stints).sum())}
        notes.append({"home": g.home, "away": g.away, **{"teams": n}})
    for c in ab.LINEUP + ab.TRACKING:
        slate[c] = np.nan
    slate["lu_games_min"] = [min(v["with_stints"] for v in n["teams"].values()) for n in notes]
    slate["trk_games_min"] = 0

    base = [c for c in ex.build_features(done) if c not in pt.EXCLUDE]
    variants = {"baseline": base, "+lineup": base + ab.LINEUP}
    out = {}
    for name, cols in variants.items():
        for t in (done, slate):
            t[cols] = t[cols].apply(pd.to_numeric, errors="coerce")
        oos, pred = pt.walk_forward(done, cols, slate)
        ok = oos.notna()
        y, x = done.loc[ok, "home_margin"].to_numpy(), oos[ok].to_numpy()
        out[name] = {"margin": np.asarray(pred, float),
                     "p_home": fit_probit_scale(x, (y > 0).astype(float), np.asarray(pred, float))}
    games = []
    for i, g in enumerate(slate.itertuples()):
        games.append({"away": g.away, "home": g.home, "lineup_coverage": notes[i]["teams"],
                      "baseline": {"home_margin": round(float(out["baseline"]["margin"][i]), 2),
                                   "p_home": round(float(out["baseline"]["p_home"][i]), 4)},
                      "+lineup": {"home_margin": round(float(out["+lineup"]["margin"][i]), 2),
                                  "p_home": round(float(out["+lineup"]["p_home"][i]), 4)},
                      "change_in_home_margin": round(float(out["+lineup"]["margin"][i] - out["baseline"]["margin"][i]), 2)})
    payload = json.dumps({"slate_date": f"{D:%Y-%m-%d}", "data_through": f"{done.GAME_DATE.max():%Y-%m-%d}",
                          "games": games}, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
