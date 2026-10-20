"""Wrap explicit, already-produced artifacts into outputs/research_runs/<run_id>/ with a manifest.

Usage (run from repo root; paths are required, nothing is auto-picked):
    python -m src.research_dashboard.capture_run --run-id r1 --metrics m.json --fold-metrics f.csv \
        --predictions p.csv --bets b.csv --odds-audit o.csv [--outputs-dir DIR]
Omitted artifacts or bad columns make the run 'partial'. Duplicate bet keys or an existing run id abort.
"""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .catalog import outputs_dir
from .manifest import ARTIFACTS, ODDS_TIERS, SCHEMA_VERSION, check_artifact, check_folds, sha256


def capture(run_id: str, sources: dict, outputs: Path, created_at: str | None = None) -> Path:
    """sources: artifact name -> source path (may omit names). Returns the run dir."""
    run_dir = Path(outputs) / "research_runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"run {run_id} already exists; completed runs are never overwritten")
    unknown = set(sources) - set(ARTIFACTS)
    if unknown:
        raise ValueError(f"unknown artifacts: {sorted(unknown)}")

    if "bets" in sources:
        bets = pd.read_csv(sources["bets"])
        if {"game_id", "bet_side"} <= set(bets.columns) and bets.duplicated(["game_id", "bet_side"]).any():
            raise ValueError("bets: duplicate (game_id, bet_side) rows; refusing to capture")

    run_dir.mkdir(parents=True)
    arts, problems = {}, []
    for name, src in sources.items():
        dest = run_dir / ARTIFACTS[name][0]
        shutil.copyfile(src, dest)
        entry = {"path": dest.name, "sha256": sha256(dest)}
        if ARTIFACTS[name][1] is not None:
            entry["rows"] = len(pd.read_csv(dest))
        arts[name] = entry
        problems += check_artifact(run_dir, name, entry)
    problems += [f"{n}: not provided" for n in ARTIFACTS if n not in sources]

    counts = {t: 0 for t in ODDS_TIERS}
    folds = []
    if "odds_audit" in arts and not any(p.startswith("odds_audit") for p in problems):
        vc = pd.read_csv(run_dir / "odds_audit.csv")["odds_tier"].value_counts()
        unexpected = set(vc.index) - set(ODDS_TIERS)
        if unexpected:
            problems.append(f"odds_audit: unknown tiers {sorted(unexpected)}")
        counts = {t: int(vc.get(t, 0)) for t in ODDS_TIERS}
    if "fold_metrics" in arts and not any(p.startswith("fold_metrics") for p in problems):
        fm = pd.read_csv(run_dir / "fold_metrics.csv")
        problems += check_folds(fm)
        folds = fm[["fold", "test_start", "test_end"]].astype(str).to_dict("records")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "complete" if not problems else "partial",
        "artifacts": arts,
        "odds_counts": counts,
        "folds": folds,
        "warnings": problems,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), newline="\n")  # written last
    return run_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--outputs-dir", type=Path)
    for name in ARTIFACTS:
        ap.add_argument("--" + name.replace("_", "-"), type=Path, dest=name)
    a = ap.parse_args()
    sources = {n: getattr(a, n) for n in ARTIFACTS if getattr(a, n)}
    out = a.outputs_dir or outputs_dir(Path(__file__).resolve().parents[2])
    print(capture(a.run_id, sources, out))


if __name__ == "__main__":
    main()
