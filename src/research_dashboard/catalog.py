"""Discover run bundles under <outputs>/research_runs."""

import os
from pathlib import Path

from .manifest import load_manifest


def outputs_dir(base: Path) -> Path:
    return Path(os.environ.get("NBA_OUTPUTS_DIR", base / "outputs"))


def list_runs(outputs: Path) -> list:
    """Newest first by created_at (failed manifests sort last)."""
    runs = [load_manifest(m.parent) for m in sorted(Path(outputs).glob("research_runs/*/manifest.json"))]
    return sorted(runs, key=lambda r: r.data.get("created_at", ""), reverse=True)

