"""Run-bundle manifest: load + validate. A run is one directory with manifest.json and its artifacts."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SCHEMA_VERSION = 1
ODDS_TIERS = ("matched", "mirrored", "default", "unmatched")

# artifact name -> (filename, required columns). metrics.json has no column check.
ARTIFACTS = {
    "metrics": ("metrics.json", None),
    "fold_metrics": ("fold_metrics.csv", ["fold", "test_start", "test_end", "rows", "rmse", "mae"]),
    "predictions": ("predictions.csv", ["game_id", "game_date", "fold", "p_home", "home_win"]),
    "bets": ("bets.csv", ["game_id", "game_date", "bet_side", "odds_tier", "prob", "edge",
                          "kelly_fraction", "win", "pnl", "bankroll"]),
    "odds_audit": ("odds_audit.csv", ["game_id", "game_date", "odds_tier"]),
}


@dataclass
class Run:
    run_dir: Path
    data: dict
    status: str  # complete | partial | failed
    problems: list = field(default_factory=list)

    @property
    def run_id(self) -> str:
        return self.data.get("run_id", self.run_dir.name)

    def read(self, name: str) -> pd.DataFrame:
        if self.status != "complete":
            raise ValueError(f"run {self.run_id} is {self.status}; refusing to load {name}")
        return pd.read_csv(self.run_dir / self.data["artifacts"][name]["path"])

    def read_metrics(self) -> dict:
        if self.status != "complete":
            raise ValueError(f"run {self.run_id} is {self.status}; refusing to load metrics")
        path = self.run_dir / self.data["artifacts"]["metrics"]["path"]
        return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_path(run_dir: Path, rel: str) -> Path:
    """Resolve rel inside run_dir; reject absolute paths and traversal."""
    p = (run_dir / rel).resolve()
    if Path(rel).is_absolute() or not p.is_relative_to(run_dir.resolve()):
        raise ValueError(f"artifact path escapes run directory: {rel}")
    return p


def check_artifact(run_dir: Path, name: str, entry: dict) -> list:
    """Return problems for one artifact entry (empty list = valid)."""
    fname, cols = ARTIFACTS[name]
    try:
        p = safe_path(run_dir, entry["path"])
    except (ValueError, KeyError) as e:
        return [f"{name}: {e}"]
    if not p.is_file():
        return [f"{name}: file missing ({entry['path']})"]
    if sha256(p) != entry.get("sha256"):
        return [f"{name}: checksum mismatch (file changed since capture)"]
    if cols is None:
        try:
            payload = json.loads(p.read_text())
        except (OSError, ValueError) as e:
            return [f"{name}: invalid JSON ({e})"]
        if not isinstance(payload, dict):
            return [f"{name}: JSON root must be an object"]
        return []
    df = pd.read_csv(p)
    out = []
    if len(df) != entry.get("rows"):
        out.append(f"{name}: row count {len(df)} != manifest {entry.get('rows')}")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        out.append(f"{name}: missing columns {missing}")
    return out


def check_folds(df: pd.DataFrame) -> list:
    """Test windows must be ordered and non-overlapping (a boundary day may be shared: folds split by row)."""
    d = df.sort_values("fold")
    s, e = pd.to_datetime(d["test_start"]), pd.to_datetime(d["test_end"])
    if (s > e).any() or (s.iloc[1:].values < e.iloc[:-1].values).any():
        return ["fold_metrics: test windows overlap or are out of order"]
    return []


def load_manifest(run_dir) -> Run:
    """Never raises on a bad run; an invalid manifest yields status 'failed', bad artifacts 'partial'."""
    run_dir = Path(run_dir)
    try:
        data = json.loads((run_dir / "manifest.json").read_text())
    except (OSError, ValueError) as e:
        return Run(run_dir, {}, "failed", [f"manifest unreadable: {e}"])
    if data.get("schema_version") != SCHEMA_VERSION:
        return Run(run_dir, data, "failed", [f"unsupported schema_version {data.get('schema_version')}"])

    problems = list(data.get("warnings", []))
    arts = data.get("artifacts", {})
    for name in ARTIFACTS:
        if name not in arts:
            problems.append(f"{name}: not captured")
        else:
            problems += check_artifact(run_dir, name, arts[name])
    if not problems and data.get("status") == "complete":
        problems += check_folds(pd.read_csv(run_dir / arts["fold_metrics"]["path"]))
    status = "complete" if not problems and data.get("status") == "complete" else "partial"
    return Run(run_dir, data, status, problems)
