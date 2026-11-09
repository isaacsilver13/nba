"""Does the +lineup (or +tracking, +both) increment predict the closing line's error? Per the frozen pre-registration.

    NBA_DATA_DIR=<data dir> python tools/lineup_market_encompass.py [--preds outputs/ablation/lineup_tracking.csv]

Reads the saved out-of-sample predictions from tools/lineup_tracking_ablation.py; refits nothing. Rules and the
regression are in docs/lineup-market-encompassing-preregistration.md and are not repeated here.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import experiment as ex  # noqa: E402
import predict_today as pt  # noqa: E402
from encompass_check import ols  # noqa: E402

PREREG = ROOT / "docs" / "lineup-market-encompassing-preregistration.md"
POOLED, REPLICATION = (2022, 2023, 2024, 2025), 2026


def gamma(d, variant):
    """Coefficient on the variant's increment, plus se, one-sided p (g > 0), n and the increment's sd."""
    X = np.column_stack([d["pred_baseline"] - d["market"], d[f"pred_{variant}"] - d["pred_baseline"]])
    b, se, _ = ols(d["home_margin"] - d["market"], X, d["GAME_DATE"].values)
    from scipy.stats import norm
    return {"g": round(float(b[2]), 3), "se": round(float(se[2]), 3), "p_one_sided": round(float(norm.sf(b[2] / se[2])), 4),
            "baseline_b": round(float(b[1]), 3), "increment_sd": round(float((X[:, 1]).std()), 3), "n": int(len(d))}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preds", type=Path, default=ex.NBA_OUTPUTS_DIR / "ablation" / "lineup_tracking.csv")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if "Frozen: yes" not in PREREG.read_text(encoding="utf-8"):
        raise SystemExit("pre-registration is not frozen; refusing to run")

    preds = pd.read_csv(args.preds, parse_dates=["GAME_DATE"])
    hist = pt.load_history()[["GAME_ID", "spread_signed"]]
    d = preds.merge(hist, on="GAME_ID", how="left", validate="one_to_one")
    d["market"] = -pd.to_numeric(d["spread_signed"], errors="coerce")
    d = d.dropna(subset=["market"])

    out = {"prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
           "preds_file": str(args.preds), "games_with_line": int(len(d)), "results": {}}
    for variant in ("+lineup", "+tracking", "+both"):
        key = variant
        dd = d.rename(columns={f"pred_{variant}": f"pred_{key}"})
        pooled = dd[dd.season.isin(POOLED)]
        rep = dd[dd.season == REPLICATION]
        res = {"pooled_2022_2025": gamma(pooled, key), "replication_2026": gamma(rep, key),
               "by_season": {int(s): gamma(dd[dd.season == s], key) for s in POOLED}}
        out["results"][variant] = res
    r = out["results"]["+lineup"]
    supported = r["pooled_2022_2025"]["g"] > 0 and r["pooled_2022_2025"]["p_one_sided"] < 0.05 and r["replication_2026"]["g"] > 0
    out["primary_lineup_supported"] = bool(supported)
    text = json.dumps(out, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
