"""Pure snapshot-schedule policy: which capture slots a slate has, which are due now, which were missed.

No I/O. Datetimes are timezone-aware; slate dates and windows are America/Chicago wall clock.
"""
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

CT = ZoneInfo("America/Chicago")
MAX_FAILURES = 3
DONE = {"ok", "missed"}
FAILED = {"error", "empty", "skipped_quota"}


@dataclass(frozen=True)
class Slot:
    provider: str            # "sgo" (scheduled) | "oddsapi" (manual smoke capture only)
    slate_date: date         # Central date of the games
    name: str                # open | settle | manual | manual-settle
    opens: datetime          # capture window is [opens, closes)
    closes: datetime
    starts_after: datetime   # request filter on game start time
    starts_before: datetime
    finalized: bool = False  # SGO settle: ask for finished games instead of unstarted ones

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.provider, self.slate_date.isoformat(), self.name)


def ct(d: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(d, time(hour, minute), CT)


def slate_tips(events, d: date) -> list[datetime]:
    """One tip per game, sorted, for Central date d.

    Events without a parseable timezone-aware commence_time are skipped: one bad cached event must not stop every run.
    """
    tips = []
    for e in events:
        try:
            t = datetime.fromisoformat(e["commence_time"])
        except (KeyError, TypeError, ValueError):
            continue
        if t.tzinfo is not None and t.astimezone(CT).date() == d:
            tips.append(t)
    return sorted(tips)


def slots_for(d: date, tips: list[datetime]) -> list[Slot]:
    if not tips:
        return []
    nxt = d + timedelta(days=1)
    lo, hi = ct(d, 0), ct(nxt, 0)
    # SGO is billed per event, so it gets two buys a night. The settle response carries every market's open and
    # close odds (player props included), and its closeOdds is SGO's own close, so no 3pm or pre-tip SGO snapshot
    # is needed (checked on the 2026-10-25/26 captures). The window stays open all day: a missed settle has no
    # other source of the close. The Odds API is not scheduled: the betting app buys NBA from it, and its free
    # /events list is only used above to know a slate exists.
    return [Slot("sgo", d, "open", ct(d, 10), ct(d, 12), lo, hi),
            Slot("sgo", d, "settle", ct(nxt, 10), ct(nxt, 20), lo, hi, finalized=True)]


def _status(log: list[dict]) -> tuple[set, Counter]:
    done, fails = set(), Counter()
    for r in log:
        k = (r.get("provider"), r.get("slate_date"), r.get("slot"))
        if r.get("outcome") in DONE:
            done.add(k)
        elif r.get("outcome") in FAILED:
            fails[k] += 1
    return done, fails


def due(slots: list[Slot], log: list[dict], now: datetime) -> list[Slot]:
    done, fails = _status(log)
    return [s for s in slots if s.opens <= now < s.closes
            and s.key not in done and fails[s.key] < MAX_FAILURES]


def missed(slots: list[Slot], log: list[dict], now: datetime) -> list[Slot]:
    done, _ = _status(log)
    return [s for s in slots if now >= s.closes and s.key not in done]
