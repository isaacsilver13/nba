"""Pure snapshot-schedule policy: which capture slots a slate has, which are due now, which were missed.

No I/O. Datetimes are timezone-aware; slate dates and windows are America/Chicago wall clock.
"""
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

CT = ZoneInfo("America/Chicago")
CLOSE_OPENS = timedelta(minutes=20)   # close window opens this long before tip (first tick captures)...
CLOSE_ENDS = timedelta(minutes=3)     # ...and ends this long before tip
# An SGO close requests games starting within tip +/- this. Tips come from The Odds API, whose pre-game
# commence_time runs ~10 min after the official tip that SGO uses (20:10Z vs 20:00Z), so 5 min missed every game.
# 15 min still separates tip groups 30 min apart.
SGO_TIP_PAD = timedelta(minutes=15)
GROUP_SPAN = timedelta(minutes=30)    # Odds API: tips this close to a group's first tip share one call
MAX_ODDSAPI_CLOSES = 3
MAX_FAILURES = 3
DONE = {"ok", "missed"}
FAILED = {"error", "empty", "skipped_quota"}


@dataclass(frozen=True)
class Slot:
    provider: str            # "sgo" | "oddsapi"
    slate_date: date         # Central date of the games
    name: str                # open | props | settle | close@HHMM | manual | manual-settle
    opens: datetime          # capture window is [opens, closes)
    closes: datetime
    starts_after: datetime   # request filter on game start time
    starts_before: datetime
    finalized: bool = False  # SGO settle: ask for finished games instead of unstarted ones
    capped: bool = False     # over the Odds API daily close cap: never due, reported missed

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.provider, self.slate_date.isoformat(), self.name)


def ct(d: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(d, time(hour, minute), CT)


def slate_tips(events, d: date) -> list[datetime]:
    """One tip per game (duplicates kept so group sizes count games), sorted, for Central date d.

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


def group_tips(tips: list[datetime]) -> list[list[datetime]]:
    groups: list[list[datetime]] = []
    for t in tips:
        if groups and t - groups[-1][0] <= GROUP_SPAN:
            groups[-1].append(t)
        else:
            groups.append([t])
    return groups


def _close_name(tip: datetime) -> str:
    return f"close@{tip.astimezone(CT):%H%M}"


def slots_for(d: date, tips: list[datetime]) -> list[Slot]:
    if not tips:
        return []
    nxt = d + timedelta(days=1)
    lo, hi = ct(d, 0), ct(nxt, 0)
    out = [Slot("sgo", d, "open", ct(d, 10), ct(d, 12), lo, hi),
           Slot("oddsapi", d, "open", ct(d, 10), ct(d, 12), lo, hi),
           Slot("sgo", d, "settle", ct(nxt, 10), ct(nxt, 12), lo, hi, finalized=True)]
    if tips[-1] > ct(d, 15):
        out.append(Slot("sgo", d, "props", ct(d, 15), ct(d, 17), lo, hi))
    for t in sorted(set(tips)):
        out.append(Slot("sgo", d, _close_name(t), t - CLOSE_OPENS, t - CLOSE_ENDS, t - SGO_TIP_PAD, t + SGO_TIP_PAD))
    groups = group_tips(tips)
    keep = sorted(groups, key=len, reverse=True)[:MAX_ODDSAPI_CLOSES]   # stable: ties keep the earlier group
    for g in groups:
        out.append(Slot("oddsapi", d, _close_name(g[0]), g[0] - CLOSE_OPENS, g[0] - CLOSE_ENDS, lo, hi,
                        capped=g not in keep))
    return out


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
    return [s for s in slots if not s.capped and s.opens <= now < s.closes
            and s.key not in done and fails[s.key] < MAX_FAILURES]


def missed(slots: list[Slot], log: list[dict], now: datetime) -> list[Slot]:
    done, _ = _status(log)
    return [s for s in slots if now >= s.closes and s.key not in done]
