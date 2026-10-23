"""Save the latest official NBA injury-report PDF as a dated, hashed snapshot. Collection only: nothing is parsed or
fed to a model, and the NBA.com terms question for using it in picks is still open (see
docs/season-transition-and-lineup-features-plan.md).

    python tools/snapshot_injury_report.py                 # newest published 15-minute slot, walking back from now
    python tools/snapshot_injury_report.py --slot 09_30AM  # one slot of today's (US/Eastern) report date

Why: the report is a point-in-time record that cannot be rebuilt afterwards at the state it had at the morning cutoff,
so every day not saved is lost for backtests. Files go to <data>/raw/injury_reports/YYYY-MM-DD/<slot>.pdf with a
manifest.jsonl (url, fetched_at_utc, sha256, bytes). Re-running is harmless: an existing slot is kept as is.
Slots are US/Eastern, named like Injury-Report_2026-10-22_09_30AM.pdf.
"""
import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NBA_DATA_DIR", ROOT / "data"))
URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{day}_{slot}.pdf"
ET = ZoneInfo("America/New_York")


def slot_name(t):
    """15-minute slot label as the NBA names it: 09_30AM, 12_15PM."""
    return t.strftime("%I_%M%p")


def slots_back(now, hours=6):
    """Slot labels from the newest 15-minute boundary at or before `now`, going back within the same date."""
    t = now.replace(minute=now.minute // 15 * 15, second=0, microsecond=0)
    floor = max(t - timedelta(hours=hours), t.replace(hour=0, minute=0))
    out = []
    while t >= floor:
        out.append(slot_name(t))
        t -= timedelta(minutes=15)
    return out


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research snapshot)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):  # NBA's CDN answers 403 for an unpublished slot
            return None
        raise
    return body if body[:5] == b"%PDF-" else None


def save(day, slot, body, url):
    folder = DATA / "raw" / "injury_reports" / day
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{slot}.pdf"
    if path.exists():
        return path, False
    path.write_bytes(body)
    with (folder / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"slot": slot, "url": url, "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}) + "\n")
    return path, True


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slot", help="one slot, e.g. 09_30AM (today's US/Eastern report date)")
    parser.add_argument("--hours", type=int, default=6, help="how far back to look for a published slot")
    args = parser.parse_args()
    now = datetime.now(ET)
    day = f"{now:%Y-%m-%d}"
    for slot in ([args.slot] if args.slot else slots_back(now, args.hours)):
        url = URL.format(day=day, slot=slot)
        body = fetch(url)
        if body:
            path, new = save(day, slot, body, url)
            print(f"{'saved' if new else 'already have'} {path} ({len(body)} bytes)")
            return
        time.sleep(0.3)
    print(f"no published slot found for {day} in the last {args.hours}h")


if __name__ == "__main__":
    main()
