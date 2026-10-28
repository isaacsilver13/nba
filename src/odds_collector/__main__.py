"""Odds collector CLI.

python -m src.odds_collector --dry-run          report missed/due slots from cached files (no network, no writes)
python -m src.odds_collector --run-due          capture every due slot (the scheduled entry point)
python -m src.odds_collector --capture sgo      one manual smoke capture: sgo | sgo-settle | oddsapi
python -m src.odds_collector --rebuild-parquet  regenerate every .parquet from its raw .json.gz
"""
import argparse
import json
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from . import flatten, providers, schedule, store

UTC = timezone.utc
SGO_OBJECT_STOP = 2300
ODDSAPI_CREDIT_STOP = 450
SGO_PAGE_LIMIT = 50
ODDSAPI_ODDS_COST = 3   # h2h,spreads,totals x region us
SCHEDULE_MAX_AGE = timedelta(hours=2)
EVENTS_ENDPOINT = "/v4/sports/basketball_nba/events"


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def utcnow() -> datetime:
    return datetime.now(UTC)


def _schedule_path(root: Path, d: date) -> Path:
    return root / "schedule" / f"{d.isoformat()}.json"


def load_schedule(root: Path, d: date) -> dict:
    try:
        sched = json.loads(_schedule_path(root, d).read_text(encoding="utf-8"))
        if isinstance(sched["events"], dict):
            return sched
    except (OSError, ValueError, KeyError, TypeError):   # missing or corrupt: refetch /events and rebuild
        pass
    return {"fetched_at_utc": None, "events": {}}


def _quota(r: providers.Response) -> dict:
    return {"quota_used": r.headers.get("x-requests-used", ""),
            "quota_remaining": r.headers.get("x-requests-remaining", "")}


def _rc(r: providers.Response) -> int:
    """Auth failures fail the workflow run so GitHub emails the owner; everything else retries next tick.

    (capture() also returns 1 for saved-but-flagged captures: partial pages or a flatten error.)
    """
    return 1 if r.status in (401, 403) else 0


def _row(slot: schedule.Slot, now: datetime, **kw) -> dict:
    return {"provider": slot.provider, "slate_date": slot.slate_date.isoformat(), "slot": slot.name,
            "attempted_at_utc": iso(now), **kw}


def _events_by_id(body: list) -> dict:
    return {e["id"]: e for e in body if isinstance(e, dict) and e.get("id")}


def refresh_schedule(root: Path, d: date, now: datetime) -> tuple[dict, int]:
    """Fetch the quota-free /events list, merge it by event id into the day's cache, and log the call."""
    sched = load_schedule(root, d)
    if sched["fetched_at_utc"] is None and _schedule_path(root, d).exists():   # unreadable cache: say so in the audit log
        store.append_log(root, {"request_id": uuid.uuid4().hex, "provider": "oddsapi", "slate_date": d.isoformat(),
                                "slot": "schedule", "attempted_at_utc": iso(now), "outcome": "error",
                                "error": f"corrupt schedule cache {_schedule_path(root, d).name}, rebuilding"})
    if not sched["events"]:   # first fetch of the day: games that already tipped are gone from /events,
        sched["events"] = load_schedule(root, d - timedelta(days=1))["events"]   # yesterday's list had them
    r = providers.oddsapi_events()
    ok = r.status == 200 and isinstance(r.body, list)
    if ok:
        # Merge, don't replace: started games drop out of /events but their slots must stay evaluable.
        sched = {"fetched_at_utc": iso(now), "events": {**sched["events"], **_events_by_id(r.body)}}
        store.atomic_write(_schedule_path(root, d), json.dumps(sched, indent=1, sort_keys=True).encode())
    store.append_log(root, {"request_id": uuid.uuid4().hex, "provider": "oddsapi", "slate_date": d.isoformat(),
                            "slot": "schedule", "attempted_at_utc": iso(now), "endpoint": EVENTS_ENDPOINT,
                            "http_status": r.status, "outcome": "ok" if ok else "error", "error": r.error,
                            "objects": len(r.body) if ok else 0, **_quota(r)})
    return sched, _rc(r)


def capture(root: Path, slot: schedule.Slot, now: datetime, log: list[dict],
            paginate: bool = True, limit: int = SGO_PAGE_LIMIT) -> int:
    rid = uuid.uuid4().hex
    row = _row(slot, now, request_id=rid)
    after = slot.starts_after if slot.finalized else max(slot.starts_after, now)   # never in-play games
    if slot.provider == "sgo":
        used = providers.sgo_objects_used()   # account-wide: the betting app buys from the same 2,500
        if used is None:
            used = store.sgo_objects_this_month(log, now)   # unreadable: count only what this log knows
        if used >= SGO_OBJECT_STOP:
            store.append_log(root, {**row, "outcome": "skipped_quota"})
            return 0
        endpoint = "/v2/events"
        params = {"leagueID": "NBA", "includeOpenCloseOdds": "true", "limit": str(limit),
                  "startsAfter": iso(after), "startsBefore": iso(slot.starts_before),
                  **({"finalized": "true"} if slot.finalized else {"started": "false"})}
        r = providers.sgo_events(params, paginate)
        # a 200 with an unusable body is still billed as one object
        objects = sum(len(page.get("data") or []) for page in r.body) if r.body else int(r.status == 200)
    else:
        if store.oddsapi_used(log) + ODDSAPI_ODDS_COST > ODDSAPI_CREDIT_STOP:
            store.append_log(root, {**row, "outcome": "skipped_quota"})
            return 0
        endpoint = "/v4/sports/basketball_nba/odds"
        params = {"regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "american",
                  "commenceTimeFrom": iso(after), "commenceTimeTo": iso(slot.starts_before)}
        r = providers.oddsapi_odds(params)
        objects = len(r.body) if r.body else 0
    got = utcnow()   # when the odds were received; `now` is when the run started
    row.update(endpoint=endpoint, http_status=r.status, error=r.error, objects=objects, **_quota(r))
    if r.status != 200 or r.body is None:
        store.append_log(root, {**row, "outcome": "error"})
        return _rc(r)
    if objects == 0:   # SGO bills every response as at least one object
        store.append_log(root, {**row, "outcome": "empty", "objects": 1 if slot.provider == "sgo" else 0})
        return 0
    stem = (f"{slot.provider}/date={slot.slate_date.isoformat()}/"
            f"{got:%Y-%m-%dT%H-%M-%SZ}_{slot.name}_{rid}")
    env = {"request_id": rid, "provider": slot.provider, "slot": slot.name,
           "slate_date": slot.slate_date.isoformat(), "captured_at_utc": iso(got),
           "request": {"endpoint": endpoint, "params": params}, "http_status": r.status,
           "headers": r.headers, "payload": r.body}
    store.write_raw(root / f"{stem}.json.gz", env)
    try:
        rows, schema = flatten.flatten(env)
        store.write_parquet(root / f"{stem}.parquet", rows, schema)
    except Exception as e:   # raw is already safe; --rebuild-parquet can redo this after a fix
        row["error"] = f"flatten: {type(e).__name__}: {e}"
    store.append_log(root, {**row, "outcome": "ok", "payload_path": f"{stem}.json.gz"})
    return 1 if row["error"] else 0   # data is saved, but a partial page set or flatten bug must reach the owner


def run_due(root: Path, now: datetime, dry_run: bool = False) -> int:
    today = now.astimezone(schedule.CT).date()
    yday = today - timedelta(days=1)
    sched, rc = load_schedule(root, today), 0
    fetched = sched["fetched_at_utc"]
    if fetched is None or now - datetime.fromisoformat(fetched) > SCHEDULE_MAX_AGE:
        if dry_run:
            print(f"schedule stale (fetched {fetched}); a real run would refresh /events")
        else:
            sched, rc = refresh_schedule(root, today, now)
    slots = (schedule.slots_for(yday, schedule.slate_tips(load_schedule(root, yday)["events"].values(), yday))
             + schedule.slots_for(today, schedule.slate_tips(sched["events"].values(), today)))
    log = store.read_log(root)
    for s in schedule.missed(slots, log, now):
        if dry_run:
            print("missed", *s.key)
        else:
            store.append_log(root, _row(s, now, request_id=uuid.uuid4().hex, outcome="missed"))
    for s in schedule.due(slots, log, now):
        if dry_run:
            print("due", *s.key)
        else:
            rc |= capture(root, s, now, store.read_log(root))
    return rc


def manual_slot(kind: str, now: datetime) -> schedule.Slot:
    today = now.astimezone(schedule.CT).date()
    if kind == "sgo-settle":
        yday = today - timedelta(days=1)
        return schedule.Slot("sgo", yday, "manual-settle", now, now, schedule.ct(yday, 0), schedule.ct(today, 0),
                             finalized=True)
    return schedule.Slot(kind, today, "manual", now, now, now, now + timedelta(days=1 if kind == "sgo" else 7))


def rebuild_parquet(root: Path) -> int:
    raws = sorted(root.glob("*/date=*/*.json.gz"))
    for raw in raws:
        env = store.read_raw(raw)
        rows, schema = flatten.flatten(env)
        store.write_parquet(raw.with_name(raw.name.removesuffix(".json.gz") + ".parquet"), rows, schema)
    print(f"rebuilt {len(raws)} parquet files")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.odds_collector", description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="report missed/due slots; no network, no writes")
    g.add_argument("--run-due", action="store_true", help="capture every due slot (scheduled entry point)")
    g.add_argument("--capture", choices=["sgo", "sgo-settle", "oddsapi"],
                   help="one manual smoke capture; ignores windows, keeps quota guards; SGO: games in the next 24h, same request shape as scheduled runs")
    g.add_argument("--rebuild-parquet", action="store_true", help="regenerate every .parquet from raw")
    a = p.parse_args(argv)
    root, now = store.data_root(), utcnow()
    if a.rebuild_parquet:
        return rebuild_parquet(root)
    if a.capture:
        return capture(root, manual_slot(a.capture, now), now, store.read_log(root))
    return run_due(root, now, dry_run=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
