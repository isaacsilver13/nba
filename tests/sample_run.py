"""Synthetic run-bundle inputs (no real data). `python -m tests.sample_run` writes data/sample/research_runs/sample_complete."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.research_dashboard.capture_run import capture


def write_sources(d: Path, seed: int = 0) -> dict:
    """Write synthetic artifact files into d; return {artifact: path}."""
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True, exist_ok=True)
    n = 240
    dates = pd.date_range("2024-01-01", periods=n // 4).repeat(4)
    pred = pd.DataFrame({
        "game_id": [f"G{i:04d}" for i in range(n)],
        "game_date": dates.strftime("%Y-%m-%d"),
        "fold": np.repeat([1, 2, 3], n // 3),
        "p_home": rng.uniform(0.2, 0.8, n).round(4),
    })
    pred["home_win"] = (rng.uniform(size=n) < pred["p_home"]).astype(int)

    fold_rows = []
    for f, g in pred.groupby("fold"):
        fold_rows.append({"fold": f, "test_start": g["game_date"].min(), "test_end": g["game_date"].max(),
                          "rows": len(g), "rmse": round(15 + rng.uniform(), 3), "mae": round(12 + rng.uniform(), 3)})
    folds = pd.DataFrame(fold_rows)

    tiers = rng.choice(["matched", "mirrored", "default", "unmatched"], n, p=[0.7, 0.1, 0.15, 0.05])
    audit = pd.DataFrame({"game_id": pred["game_id"], "game_date": pred["game_date"], "odds_tier": tiers})

    sel = pred.sample(60, random_state=seed).sort_values("game_date")
    bets = pd.DataFrame({
        "game_id": sel["game_id"], "game_date": sel["game_date"],
        "bet_side": np.where(sel["p_home"] >= 0.5, "HOME", "AWAY"),
        "odds_tier": audit.set_index("game_id").loc[sel["game_id"], "odds_tier"].values,
        "prob": np.maximum(sel["p_home"], 1 - sel["p_home"]).values,
        "edge": rng.uniform(0.02, 0.1, len(sel)).round(4),
        "kelly_fraction": rng.uniform(0.01, 0.1, len(sel)).round(4),
    })
    bets["win"] = (rng.uniform(size=len(bets)) < bets["prob"]).astype(int)
    stake = bets["kelly_fraction"]
    bets["pnl"] = np.where(bets["win"] == 1, stake * 0.91, -stake).round(5)  # fraction of current bankroll
    bets["bankroll"] = (1 + bets["pnl"]).cumprod().round(5)

    paths = {"fold_metrics": folds, "predictions": pred, "bets": bets, "odds_audit": audit}
    out = {}
    for name, df in paths.items():
        out[name] = d / f"{name}.csv"
        df.to_csv(out[name], index=False, lineterminator="\n")  # same bytes on every OS, so committed checksums hold
    out["metrics"] = d / "metrics.json"
    out["metrics"].write_text(json.dumps({"rmse": float(folds["rmse"].mean()), "mae": float(folds["mae"].mean()),
                                          "n_folds": len(folds), "starting_bankroll": 1.0,
                                          "units": "kelly_fraction and pnl are fractions of the CURRENT bankroll"}))
    return out


SAMPLE_CREATED_AT = "2026-10-06T07:21:44+00:00"


def make_run(outputs: Path, run_id: str, scratch: Path, seed: int = 0, created_at: str | None = None) -> Path:
    return capture(run_id, write_sources(scratch, seed), outputs, created_at=created_at)


def regenerate_committed_sample(root: Path) -> Path:
    """Safely replace only the known synthetic fixture files; write the manifest last."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "research_runs" / "sample_complete"
    target.mkdir(parents=True, exist_ok=True)
    filenames = ["fold_metrics.csv", "predictions.csv", "bets.csv", "odds_audit.csv", "metrics.json", "manifest.json"]
    with TemporaryDirectory(dir=root) as temp:
        temp_root = Path(temp)
        generated = make_run(
            temp_root / "generated",
            "sample_complete",
            temp_root / "sources",
            created_at=SAMPLE_CREATED_AT,
        )
        for filename in filenames:
            (generated / filename).replace(target / filename)
    return target


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "data" / "sample"
    print(regenerate_committed_sample(root))
