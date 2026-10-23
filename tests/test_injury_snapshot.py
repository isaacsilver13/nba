"""Slot naming for the injury-report snapshot (no network)."""
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import snapshot_injury_report as s  # noqa: E402

ET = ZoneInfo("America/New_York")


def test_slots_match_the_nba_file_names_and_walk_back_from_the_last_boundary():
    now = datetime(2026, 10, 22, 17, 41, tzinfo=ET)
    slots = s.slots_back(now, hours=1)
    assert slots == ["05_30PM", "05_15PM", "05_00PM", "04_45PM", "04_30PM"]


def test_walk_back_stops_at_midnight_and_labels_noon_correctly():
    assert s.slots_back(datetime(2026, 10, 22, 0, 20, tzinfo=ET), hours=6) == ["12_15AM", "12_00AM"]
    assert s.slot_name(datetime(2026, 10, 22, 12, 0, tzinfo=ET)) == "12_00PM"
