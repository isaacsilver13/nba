"""Walk-forward ablation: baseline vs +lineup vs +tracking vs +both, per the frozen pre-registration.

    NBA_DATA_DIR=<data dir> python tools/lineup_tracking_ablation.py [--out results.json]

Rules, blocks, metrics and thresholds live in docs/lineup-tracking-ablation-preregistration.md (not repeated here).
The script hashes that file into the result, and refuses to run unless it carries the `Frozen: yes` marker.
Opt-in research run; it writes nothing but the result JSON and a per-game prediction CSV next to it.

Implementation notes for the registered rules: the Brier score of rule 3 fits each variant's probit scale
(`fit_probit_scale`) on its own DEVELOPMENT-block out-of-sample margins and applies it to the holdout margins, so the
holdout never sets its own calibration.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import experiment as ex  # noqa: E402
import predict_today as pt  # noqa: E402
from encompass_check import fit_probit_scale  # noqa: E402

DATA = Path(os.environ.get("NBA_DATA_DIR", ROOT / "data"))
PREREG = ROOT / "docs" / "lineup-tracking-ablation-preregistration.md"
DEV, HOLDOUT, SECOND = (2022, 2023, 2024), (2025,), (2026,)
LINEUP = ["lu_top_unit_net_diff", "lu_top_unit_share_diff", "lu_starter_unit_net_diff", "lu_starter_unit_share_diff",
          "lu_unit_concentration_diff", "lu_games_min"]
TRACKING = ["trk_dist_per_min_diff", "trk_speed_diff", "trk_touches_per_min_diff", "trk_passes_per_min_diff",
            "trk_uncontested_fga_share_diff", "trk_rim_fga_per_min_diff", "trk_rim_fg_allowed_diff",
            "trk_reb_chances_per_min_diff", "trk_games_min"]
DRAWS, SEED = 2000, 20261022


def load_table():
    done = pt.load_history().sort_values(["GAME_DATE", "GAME_ID"], kind="stable").reset_index(drop=True)
    lu = pd.read_csv(DATA / "processed" / "lineup_features.csv")
    tr = pd.read_csv(DATA / "processed" / "tracking_features.csv")
    lu["lu_unit_concentration_diff"] = lu["home_lu_unit_concentration"] - lu["away_lu_unit_concentration"]
    lu["lu_games_min"] = lu[["home_lu_games", "away_lu_games"]].min(axis=1)
    tr["trk_games_min"] = tr[["home_trk_games", "away_trk_games"]].min(axis=1)
    out = done.merge(lu[["GAME_ID", *LINEUP]], on="GAME_ID", how="left", validate="one_to_one")
    out = out.merge(tr[["GAME_ID", *TRACKING]], on="GAME_ID", how="left", validate="one_to_one")
    assert len(out) == len(done)
    return out


def paired_boot(se_base, se_var, dates):
    """RMSE improvement (baseline - variant) with a date-block bootstrap; returns point, 90% interval."""
    d = pd.DataFrame({"b": se_base, "v": se_var, "n": 1.0, "date": dates}).groupby("date").sum()
    b, v, n = d.b.to_numpy(), d.v.to_numpy(), d.n.to_numpy()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(d), size=(DRAWS, len(d)))
    imp = np.sqrt(b[idx].sum(1) / n[idx].sum(1)) - np.sqrt(v[idx].sum(1) / n[idx].sum(1))
    point = float(np.sqrt(b.sum() / n.sum()) - np.sqrt(v.sum() / n.sum()))
    return point, [float(np.percentile(imp, 5)), float(np.percentile(imp, 95))]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ex.NBA_OUTPUTS_DIR / "ablation" / "lineup_tracking.json")
    args = parser.parse_args()
    text = PREREG.read_text(encoding="utf-8")
    if "Frozen: yes" not in text:
        raise SystemExit("pre-registration is not frozen; refusing to run")

    table = load_table()
    base = [c for c in ex.build_features(table) if c not in pt.EXCLUDE]
    assert not set(base) & set(LINEUP + TRACKING), "added columns already in baseline"
    variants = {"baseline": base, "+lineup": base + LINEUP, "+tracking": base + TRACKING,
                "+both": base + LINEUP + TRACKING}
    for cols in variants.values():
        table[cols] = table[cols].apply(pd.to_numeric, errors="coerce")

    oos = {}
    for name, cols in variants.items():
        print(f"[ablation] {name}: {len(cols)} features", flush=True)
        oos[name] = pt.walk_forward(table, cols)[0]

    known = pd.concat(oos, axis=1).notna().all(axis=1)  # identical games for every variant
    t = table[known].copy()
    for name in variants:
        t[f"pred_{name}"] = oos[name][known]
    t["se_base"] = (t.home_margin - t["pred_baseline"]) ** 2

    def block(seasons):
        return t.season.isin(seasons)

    def rmse(name, mask):
        return float(np.sqrt(((t.home_margin - t[f"pred_{name}"])[mask] ** 2).mean()))

    def mae(name, mask):
        return float((t.home_margin - t[f"pred_{name}"])[mask].abs().mean())

    dev, hold, second = block(DEV), block(HOLDOUT), block(SECOND)
    opening = (block(DEV + HOLDOUT)) & t.GAME_DATE.dt.month.isin([10, 11])
    win = (t.home_margin > 0).astype(float)

    def brier(name):
        scale_fit = lambda m: fit_probit_scale(t.loc[dev, f"pred_{name}"].to_numpy(),  # noqa: E731
                                               win[dev].to_numpy(), m)
        p = scale_fit(t.loc[hold, f"pred_{name}"].to_numpy())
        return float(np.mean((p - win[hold].to_numpy()) ** 2))

    report = {}
    for name in variants:
        r = {"features": len(variants[name]),
             "rmse": {k: round(rmse(name, m), 3) for k, m in
                      (("dev", dev), ("holdout", hold), ("second_2026", second), ("opening_dev_holdout", opening))},
             "mae": {k: round(mae(name, m), 3) for k, m in (("dev", dev), ("holdout", hold))},
             "rmse_by_season": {int(s): round(rmse(name, t.season.eq(s)), 3) for s in sorted(t.season.unique())},
             "brier_holdout": round(brier(name), 5)}
        added = [c for c in variants[name] if c in LINEUP + TRACKING]
        r["coverage_holdout"] = {c: round(float(t.loc[hold, c].notna().mean()), 3) for c in added}
        report[name] = r

    verdicts = {}
    for name in ("+lineup", "+tracking", "+both"):
        imp, ci = paired_boot(t.se_base[hold].to_numpy(), ((t.home_margin - t[f"pred_{name}"]) ** 2)[hold].to_numpy(),
                              t.GAME_DATE[hold].to_numpy())
        rules = {
            "1_holdout_effect": imp >= 0.05 and ci[0] > 0,
            "2_no_dev_regression": report[name]["rmse"]["dev"] <= report["baseline"]["rmse"]["dev"] + 0.02,
            "3_calibration": report[name]["brier_holdout"] <= report["baseline"]["brier_holdout"] + 0.001,
            "4_opening_months": (report[name]["rmse"]["opening_dev_holdout"]
                                 <= report["baseline"]["rmse"]["opening_dev_holdout"] + 0.05),
        }
        verdicts[name] = {"holdout_rmse_improvement": round(imp, 4), "interval_90": [round(x, 4) for x in ci],
                          "rules": rules, "supported_for_further_work": all(rules.values())}

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "prereg_mtime": datetime.fromtimestamp(PREREG.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
        "git_head": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                   text=True).stdout.strip(),
        "games_compared": {"all": int(known.sum()), "dev": int(dev.sum()), "holdout": int(hold.sum()),
                           "second_2026": int(second.sum())},
        "primary_candidate": "+both", "report": report, "verdicts": verdicts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    t[["GAME_ID", "GAME_DATE", "season", "home_margin", *[f"pred_{n}" for n in variants]]].to_csv(
        args.out.with_suffix(".csv"), index=False)
    print(json.dumps({"games_compared": result["games_compared"], "report": report, "verdicts": verdicts}, indent=2))


if __name__ == "__main__":
    main()
