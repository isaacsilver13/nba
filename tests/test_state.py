"""tools/state.py pack/unpack round trip on temp files: no network, no data."""
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import state  # noqa: E402


def test_round_trip(tmp_path):
    src, dst = tmp_path / "src" / "data", tmp_path / "dst" / "data"
    for i, f in enumerate(state.FILES):
        (src / f).parent.mkdir(parents=True, exist_ok=True)
        (src / f).write_text(f"GAME_ID,x\n{i},{f}\n", encoding="utf-8")
    (src / "processed" / "df_model_3.csv").write_text("derived, not packed", encoding="utf-8")
    archive = tmp_path / "state.tar.gz"
    state.pack(archive, src)
    state.unpack(archive, dst)
    got = sorted(p.relative_to(dst).as_posix() for p in dst.rglob("*") if p.is_file())
    assert got == sorted(state.FILES)
    assert all((dst / f).read_bytes() == (src / f).read_bytes() for f in state.FILES)


def test_unpack_refuses_unexpected_members(tmp_path):
    (tmp_path / "evil.csv").write_text("x", encoding="utf-8")
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(tmp_path / "evil.csv", arcname="../evil.csv")
    with pytest.raises(SystemExit):
        state.unpack(archive, tmp_path / "data")
