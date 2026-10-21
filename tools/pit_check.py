"""Point-in-time leakage check for the feature pipeline (steps 2, 4, 5, 6, 6a).

For a date D it rebuilds the feature table from inputs that contain
  * no game after D                       (catches look-ahead into the future), and
  * scrambled OUTCOMES for games on D     (catches use of the game's own result),
then compares the rows for date D against the full build. Pre-game facts are kept intact: schedule, spread, and the
players whose absence was known before tip-off (DND/NWT/injury/rest/...). Any column that differs depends on
information that did not exist when the bet would be placed.

    python tools/pit_check.py --date 2023-01-15                  # check the current code
    python tools/pit_check.py --date 2023-01-15 --code-root ../x  # run another checkout's code (positive control)

Slow (~2 min per date): it runs the feature steps on a temporary copy of the data.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("NBA_DATA_DIR", REPO / "data"))

# columns that legitimately change when the game's own result changes
OUTCOME_COLS = {"home_margin", "away_margin"}

RUNNER = r"""
import sys, os, importlib.util
root = sys.argv[1]
sys.path.insert(0, os.path.join(root, "src"))
from ingest.dataPrep.config import get_config
from ingest.dataPrep.steps import calculate_elo_and_rest as s2, engineer_advanced_features as s4
def load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(root, "src", "ingest", "dataPrep", "steps", fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
cfg = get_config(); cfg["cache"]["read_on_hit"] = False
out = {}
for mod in [s2, s4, load("s5", "calculate_travel_fatigue.py"), load("s6", "engineer_rolling_features.py"),
            load("s6a", "finalize_feature_selection.py")]:
    out.update(mod.run(cfg, out))
"""


def _gid(s):
    return s.astype(str).str.replace(".0", "", regex=False).str.zfill(10)


def _pregame_out(box):
    status = box["comment"].fillna("").astype(str).str.upper()
    mins = box["minutes"].fillna("").astype(str).str.split(":", n=1).str[0]
    played = pd.to_numeric(mins, errors="coerce").fillna(0) > 0
    return (~played) & (status != "") & (~status.str.contains("COACH"))


def build_inputs(date, workdir, seed=0):
    rng = np.random.default_rng(seed)
    proc = Path(workdir) / "data" / "processed"
    raw = Path(workdir) / "data" / "raw"
    proc.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    D = pd.Timestamp(date)

    games = pd.read_csv(DATA / "processed" / "nba_games_with_game_id.csv", parse_dates=["date"])
    games["_gid"] = _gid(games["GAME_ID"])
    gdate = games.set_index("_gid")["date"]
    games = games[games["date"] <= D].copy()
    on_d = games["date"] == D
    new_home = games.loc[on_d, "score_away"] + rng.integers(1, 15, on_d.sum())
    new_away = games.loc[on_d, "score_home"].copy()
    games.loc[on_d, "score_home"], games.loc[on_d, "score_away"] = new_home.values, new_away.values
    games.drop(columns="_gid").to_csv(proc / "nba_games_with_game_id.csv", index=False)

    box = pd.read_csv(DATA / "processed" / "all_boxscores.csv")
    box["_date"] = _gid(box["GAME_ID"]).map(gdate)
    box = box[box["_date"] <= D].copy()
    ids = ["GAME_ID", "TEAM_ID", "teamCity", "teamName", "team", "teamSlug", "PLAYER_ID", "firstName",
           "familyName", "nameI", "playerSlug", "position", "jerseyNum", "_date"]
    vals = [c for c in box.columns if c not in ids]
    box["_pre"] = _pregame_out(box)
    for _, idx in box[(box["_date"] == D) & (~box["_pre"])].groupby(["GAME_ID", "TEAM_ID"]).groups.items():
        idx = np.asarray(idx)
        box.loc[idx, vals] = box.loc[rng.permutation(idx), vals].values  # who played / minutes / stats reshuffled
    box.drop(columns=["_date", "_pre"]).to_csv(proc / "all_boxscores.csv", index=False)

    opp = pd.read_csv(DATA / "processed" / "team opp stats.csv", dtype={"GAME_ID": str})
    opp["_date"] = _gid(opp["GAME_ID"]).map(gdate)
    opp = opp[opp["_date"] <= D].copy()
    num = [c for c in opp.select_dtypes("number").columns]
    idx = opp.index[opp["_date"] == D].to_numpy()
    opp.loc[idx, num] = opp.loc[rng.permutation(idx), num].values
    opp.drop(columns="_date").to_csv(proc / "team opp stats.csv", index=False)

    for f in ("arena_locations.csv",):
        (raw / f).write_bytes((DATA / "raw" / f).read_bytes())


def compare(workdir, date, reference):
    D = pd.Timestamp(date)
    new = pd.read_csv(Path(workdir) / "data" / "processed" / "df_model_3.csv", dtype={"GAME_ID": str})
    ref = pd.read_csv(reference, dtype={"GAME_ID": str})
    for t in (new, ref):
        t["GAME_ID"] = _gid(t["GAME_ID"])
    new = new[pd.to_datetime(new["date_x"]) == D].set_index("GAME_ID").sort_index()
    ref = ref[pd.to_datetime(ref["date_x"]) == D].set_index("GAME_ID").sort_index()
    common = new.index.intersection(ref.index)
    missing = ref.index.difference(new.index)
    bad = {}
    for c in ref.select_dtypes("number").columns:
        if c in OUTCOME_COLS or c not in new.columns:
            continue
        a, b = ref.loc[common, c].to_numpy(float), new.loc[common, c].to_numpy(float)
        ok = np.isclose(a, b, rtol=1e-7, atol=1e-7, equal_nan=True)
        if not ok.all():
            bad[c] = float(np.nanmax(np.abs(a - b)[~ok])) if np.isfinite(np.abs(a - b)[~ok]).any() else float("nan")
    return len(ref), len(common), list(missing), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--code-root", default=str(REPO))
    ap.add_argument("--reference", default=str(DATA / "processed" / "df_model_3.csv"),
                    help="df_model_3.csv built by the SAME code on the full data")
    ap.add_argument("--workdir", help="build here and keep the files (for debugging) instead of a temp dir")
    a = ap.parse_args()
    if a.workdir:
        Path(a.workdir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = a.workdir or tmp
        build_inputs(a.date, work)
        r = subprocess.run([sys.executable, "-c", RUNNER, a.code_root], cwd=work, capture_output=True, text=True)
        if r.returncode:
            print(r.stderr[-3000:])
            sys.exit(2)
        n_ref, n_common, missing, bad = compare(work, a.date, a.reference)
    print(f"date {a.date}: {n_ref} games in full build, {n_common} compared, {len(missing)} missing from rebuild")
    if bad:
        print(f"LEAK: {len(bad)} columns depend on information after tip-off / after the date:")
        for c, d in sorted(bad.items(), key=lambda x: -(x[1] if x[1] == x[1] else 0)):
            print(f"  {c:45s} max|diff| {d:.4g}")
        sys.exit(1)
    print("PASS: every feature for that date is identical without later games and with scrambled outcomes")


if __name__ == "__main__":
    main()
