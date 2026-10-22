"""Pack or restore the minimum data tools/ingest_daily.py + tools/predict_today.py need, as one .tar.gz.

    python tools/state.py pack [--out state.tar.gz]   # $NBA_DATA_DIR files -> archive
    python tools/state.py unpack state.tar.gz         # archive -> the same paths under $NBA_DATA_DIR (overwrites)

The daily-predictions workflow keeps this archive as a release asset (state-YYYY-MM-DD) in the private data repo,
since all_boxscores.csv is over GitHub's 100 MB file limit. Everything else is rebuilt: ingest_daily refetches the
ESPN cache and runs tools/run_features.py, which writes the derived tables (df_model_3.csv, team opp stats.csv, ...).
"""
import argparse
import os
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NBA_DATA_DIR", ROOT / "data"))
FILES = ["processed/nba_games_with_game_id.csv", "processed/all_boxscores.csv", "processed/team_boxscores_v3.csv",
         "raw/arena_locations.csv"]  # relative to DATA


def pack(out, data=DATA):
    with tarfile.open(out, "w:gz") as tar:
        for f in FILES:
            tar.add(data / f, arcname=f)


def unpack(archive, data=DATA):
    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
        if names != set(FILES):
            raise SystemExit(f"{archive}: expected {sorted(FILES)}, found {sorted(names)}")
        tar.extractall(data, filter="data")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pack").add_argument("--out", default="state.tar.gz")
    sub.add_parser("unpack").add_argument("archive")
    a = ap.parse_args()
    if a.cmd == "pack":
        pack(a.out)
        print(f"packed {DATA} -> {a.out} ({os.path.getsize(a.out) / 1e6:.0f} MB)")
    else:
        unpack(a.archive)
        print(f"unpacked {a.archive} -> {DATA}")
