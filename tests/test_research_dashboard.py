import json
from pathlib import Path

import pandas as pd
import pytest

from src.research_dashboard.capture_run import capture
from src.research_dashboard.catalog import legacy_entry, list_runs
from src.research_dashboard.manifest import load_manifest, safe_path
from tests.sample_run import make_run, write_sources


def test_complete_run_loads(tmp_path):
    run = load_manifest(make_run(tmp_path / "out", "r1", tmp_path / "src"))
    assert run.status == "complete", run.problems
    assert len(run.read("bets")) == 60


def test_path_traversal_rejected(tmp_path):
    with pytest.raises(ValueError):
        safe_path(tmp_path, "../x.csv")
    with pytest.raises(ValueError):
        safe_path(tmp_path, str(tmp_path.parent / "x.csv"))


def test_tampered_manifest_path_is_partial(tmp_path):
    d = make_run(tmp_path / "out", "r1", tmp_path / "src")
    m = json.loads((d / "manifest.json").read_text())
    m["artifacts"]["bets"]["path"] = "../../outside.csv"
    (d / "manifest.json").write_text(json.dumps(m))
    run = load_manifest(d)
    assert run.status == "partial"
    with pytest.raises(ValueError):
        run.read("bets")  # partial runs cannot power charts


def test_changed_artifact_is_partial(tmp_path):
    d = make_run(tmp_path / "out", "r1", tmp_path / "src")
    with open(d / "bets.csv", "a") as f:
        f.write("GX,2024-01-01,HOME,matched,0.6,0.05,0.05,1,0.04,1.0\n")
    assert load_manifest(d).status == "partial"


def test_missing_artifact_is_partial(tmp_path):
    src = write_sources(tmp_path / "src")
    del src["bets"]
    run = load_manifest(capture("r1", src, tmp_path / "out"))
    assert run.status == "partial" and any("bets" in p for p in run.problems)


def test_missing_column_is_partial(tmp_path):
    src = write_sources(tmp_path / "src")
    pd.read_csv(src["bets"]).drop(columns=["odds_tier"]).to_csv(src["bets"], index=False)
    assert load_manifest(capture("r1", src, tmp_path / "out")).status == "partial"


def test_duplicate_bet_key_rejected(tmp_path):
    src = write_sources(tmp_path / "src")
    b = pd.read_csv(src["bets"])
    pd.concat([b, b.iloc[:1]]).to_csv(src["bets"], index=False)
    with pytest.raises(ValueError, match="duplicate"):
        capture("r1", src, tmp_path / "out")
    assert not (tmp_path / "out" / "research_runs" / "r1").exists()


def test_no_overwrite(tmp_path):
    src = write_sources(tmp_path / "src")
    capture("r1", src, tmp_path / "out")
    with pytest.raises(FileExistsError):
        capture("r1", src, tmp_path / "out")


def test_overlapping_folds_rejected(tmp_path):
    src = write_sources(tmp_path / "src")
    f = pd.read_csv(src["fold_metrics"])
    f.loc[1, "test_start"] = f.loc[0, "test_start"]
    f.to_csv(src["fold_metrics"], index=False)
    assert load_manifest(capture("r1", src, tmp_path / "out")).status == "partial"


def test_catalog_orders_and_legacy_label(tmp_path):
    out = tmp_path / "out"
    make_run(out, "a", tmp_path / "s1")
    make_run(out, "b", tmp_path / "s2", seed=1)
    (out / "research_runs" / "bad").mkdir()
    (out / "research_runs" / "bad" / "manifest.json").write_text("{not json")
    runs = list_runs(out)
    assert [r.status for r in runs][-1] == "failed" and len(runs) == 3
    assert legacy_entry(tmp_path)["label"] == "Legacy / unbundled"


def test_tier_filter_and_summary(tmp_path):
    from src.research_dashboard.views import bet_summary, filter_tiers, reliability
    run = load_manifest(make_run(tmp_path / "out", "r1", tmp_path / "src"))
    bets = run.read("bets")
    only = filter_tiers(bets, ["matched"])
    assert set(only["odds_tier"]) == {"matched"} and len(only) < len(bets)
    assert 0 <= bet_summary(only)["max_drawdown"] < 1
    assert reliability(run.read("predictions"))["n"].sum() == 240


def test_fold_sharing_a_boundary_day_is_ok(tmp_path):
    src = write_sources(tmp_path / "src")
    f = pd.read_csv(src["fold_metrics"])
    f.loc[1, "test_start"] = f.loc[0, "test_end"]
    f.to_csv(src["fold_metrics"], index=False)
    assert load_manifest(capture("r1", src, tmp_path / "out")).status == "complete"


def test_committed_sample_run_is_complete():
    """The checked-in fixture must verify on every OS (guards against CRLF/LF hash drift)."""
    run = load_manifest(Path(__file__).resolve().parents[1] / "data" / "sample" / "research_runs" / "sample_complete")
    assert run.status == "complete", run.problems
