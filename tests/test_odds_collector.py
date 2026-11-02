"""Tests for src.odds_collector — synthetic fixtures only; no test may touch the network."""
import http.client
import io
import json
import urllib.error
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.odds_collector import __main__ as cli, flatten, providers, schedule, store
from src.odds_collector.schedule import CT

FIXTURES = Path(__file__).parent / "fixtures" / "odds_collector"
UTC = timezone.utc
REAL_SGO_OBJECTS_USED = providers.sgo_objects_used   # the autouse fixture below replaces the module attribute


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("test attempted a network call")
    monkeypatch.setattr("urllib.request.urlopen", refuse)


@pytest.fixture(autouse=True)
def sgo_usage_unreadable(monkeypatch):
    monkeypatch.setattr(providers, "sgo_objects_used", lambda: None)   # tests count from their own log


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    monkeypatch.setattr(cli, "utcnow", lambda: NOW)   # NOW is defined in the CLI section; looked up at call time


# --- store -----------------------------------------------------------------

def test_data_root_honours_env(tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path))
    assert store.data_root() == tmp_path.resolve() / "raw" / "odds"


def test_raw_roundtrip_is_deterministic_and_leaves_no_tmp(tmp_path):
    env = {"request_id": "r1", "payload": [{"a": 1}]}
    p = tmp_path / "sgo" / "date=2026-10-21" / "x.json.gz"
    store.write_raw(p, env)
    first = p.read_bytes()
    store.write_raw(p, env)
    assert p.read_bytes() == first
    assert store.read_raw(p) == env
    assert [f.name for f in p.parent.iterdir()] == ["x.json.gz"]


def test_log_append_and_read(tmp_path):
    store.append_log(tmp_path, {"request_id": "a", "provider": "sgo", "outcome": "ok", "objects": 3})
    store.append_log(tmp_path, {"request_id": "b", "provider": "oddsapi", "outcome": "error", "error": None})
    rows = store.read_log(tmp_path)
    assert [r["request_id"] for r in rows] == ["a", "b"]
    assert rows[0]["objects"] == "3" and rows[1]["error"] == ""
    assert list(rows[0]) == store.LOG_FIELDS


def test_read_log_missing_file_is_empty(tmp_path):
    assert store.read_log(tmp_path / "nope") == []


def test_sgo_objects_counts_only_sgo_this_utc_month():
    rows = [
        {"provider": "sgo", "attempted_at_utc": "2026-10-31T23:59:00Z", "objects": "100"},
        {"provider": "sgo", "attempted_at_utc": "2026-11-01T00:05:00Z", "objects": "7"},
        {"provider": "sgo", "attempted_at_utc": "2026-11-02T15:00:00Z", "objects": ""},
        {"provider": "oddsapi", "attempted_at_utc": "2026-11-02T15:00:00Z", "objects": "9"},
    ]
    assert store.sgo_objects_this_month(rows, datetime(2026, 11, 2, 16, tzinfo=UTC)) == 7


def test_oddsapi_used_takes_latest_logged_header():
    rows = [{"provider": "oddsapi", "quota_used": "40"}, {"provider": "sgo", "quota_used": ""},
            {"provider": "oddsapi", "quota_used": "43"}, {"provider": "oddsapi", "quota_used": ""}]
    assert store.oddsapi_used(rows) == 43
    assert store.oddsapi_used([]) == 0


def test_parquet_with_no_rows_keeps_schema(tmp_path):
    schema = pa.schema([("a", pa.string()), ("b", pa.float64())])
    store.write_parquet(tmp_path / "x.parquet", [], schema)
    assert pq.read_table(tmp_path / "x.parquet").schema.equals(schema)


# --- schedule --------------------------------------------------------------

def ct(y, m, d, h, mi=0):
    return datetime(y, m, d, h, mi, tzinfo=CT)


D = date(2026, 10, 21)
TIPS = [ct(2026, 10, 21, 18), ct(2026, 10, 21, 18), ct(2026, 10, 21, 18, 30), ct(2026, 10, 21, 21)]



def names(slots, provider=None):
    return sorted(s.name for s in slots if provider in (None, s.provider))


def test_slate_tips_filters_central_date_and_keeps_one_per_game():
    events = [{"commence_time": "2026-10-21T23:00:00Z"}, {"commence_time": "2026-10-21T23:00:00Z"},
              {"commence_time": "2026-10-22T02:00:00Z"},   # 21:00 CT, still the 21st's slate
              {"commence_time": "2026-10-22T23:00:00Z"}]   # next slate
    assert schedule.slate_tips(events, D) == [ct(2026, 10, 21, 18), ct(2026, 10, 21, 18), ct(2026, 10, 21, 21)]


def test_slots_for_a_normal_night():
    slots = schedule.slots_for(D, TIPS)
    assert names(slots, "sgo") == ["open", "settle"]   # settle carries open+close for every market
    assert names(slots, "oddsapi") == []   # the betting app buys NBA from The Odds API; the collector does not
    settle = next(s for s in slots if s.name == "settle")
    assert settle.finalized and settle.opens == ct(2026, 10, 22, 10)
    assert settle.closes == ct(2026, 10, 22, 20)   # a missed settle has no other source of the close
    assert (settle.starts_after, settle.starts_before) == (ct(2026, 10, 21, 0), ct(2026, 10, 22, 0))


def test_slate_tips_skips_malformed_events():
    good = ct(2026, 10, 21, 19)
    events = [{"id": "no-time"}, {"commence_time": "garbage"}, {"commence_time": None},
              {"commence_time": "2026-10-21T23:00:00"},   # naive: cannot be placed on the Central calendar
              {"commence_time": good.isoformat()}]
    assert schedule.slate_tips(events, D) == [good]


def test_no_games_no_slots():
    assert schedule.slots_for(D, []) == []


def test_due_respects_window_done_and_failure_cap():
    slots = schedule.slots_for(D, TIPS)
    now = ct(2026, 10, 21, 17, 45)
    now = ct(2026, 10, 21, 10, 30)
    assert [s.key for s in schedule.due(slots, [], now)] == [("sgo", "2026-10-21", "open")]
    log = [{"provider": "sgo", "slate_date": "2026-10-21", "slot": "open", "outcome": "ok"}]
    assert schedule.due(slots, log, now) == []
    log = [{"provider": "sgo", "slate_date": "2026-10-21", "slot": "open", "outcome": o}
           for o in ("error", "empty", "skipped_quota")]
    assert schedule.due(slots, log, now) == []   # three failures retire the slot
    assert [s.name for s in schedule.due(slots, [], ct(2026, 10, 22, 19, 59))] == ["settle"]
    assert schedule.due(slots, [], ct(2026, 10, 22, 20)) == []


def test_missed_reports_closed_windows_once():
    slots = schedule.slots_for(D, TIPS)
    now = ct(2026, 10, 21, 13)
    assert [s.key for s in schedule.missed(slots, [], now)] == [("sgo", "2026-10-21", "open")]
    log = [{"provider": "sgo", "slate_date": "2026-10-21", "slot": "open", "outcome": "missed"}]
    assert schedule.missed(slots, log, now) == []


def test_dst_end_day_uses_local_wall_clock():
    d = date(2026, 11, 1)   # US DST ends 02:00 this morning
    tips = schedule.slate_tips([{"commence_time": "2026-11-02T01:00:00Z"}], d)   # 19:00 CST
    assert tips == [ct(2026, 11, 1, 19)]
    slots = schedule.slots_for(d, tips)
    sgo_open = next(s for s in slots if s.key == ("sgo", "2026-11-01", "open"))
    assert sgo_open.opens.astimezone(UTC) == datetime(2026, 11, 1, 16, tzinfo=UTC)
    settle = next(s for s in slots if s.key == ("sgo", "2026-11-01", "settle"))
    assert settle.opens.astimezone(UTC) == datetime(2026, 11, 2, 16, tzinfo=UTC)   # 10:00 CST the next morning


# --- providers -------------------------------------------------------------

class FakeResp:
    def __init__(self, body, status=200, headers=None):
        self.status, self.headers, self._body = status, headers or {}, json.dumps(body).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_sgo_sends_key_in_header_and_paginates(monkeypatch):
    monkeypatch.setenv("SGO_API_KEY", "SGOKEY123")
    seen = []
    pages = iter([FakeResp({"data": [{"eventID": "a"}], "nextCursor": "c2"}),
                  FakeResp({"data": [{"eventID": "b"}], "nextCursor": None})])

    def fake(req, timeout):
        seen.append(req)
        return next(pages)
    monkeypatch.setattr(providers.urllib.request, "urlopen", fake)
    r = providers.sgo_events({"leagueID": "NBA"})
    assert r.status == 200 and [p["data"][0]["eventID"] for p in r.body] == ["a", "b"]
    assert all("SGOKEY123" not in q.full_url for q in seen)
    assert seen[0].get_header("X-api-key") == "SGOKEY123"
    assert "cursor=c2" in seen[1].full_url


def test_sgo_without_pagination_stops_after_first_page(monkeypatch):
    monkeypatch.setenv("SGO_API_KEY", "SGOKEY123")
    monkeypatch.setattr(providers.urllib.request, "urlopen",
                        lambda req, timeout: FakeResp({"data": [{"eventID": "a"}], "nextCursor": "c2"}))
    assert len(providers.sgo_events({}, paginate=False).body) == 1


def test_http_error_text_is_redacted(monkeypatch):
    monkeypatch.setenv("THE_ODDS_API_KEY", "ODDSKEY999")

    def fake(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {"X-Requests-Used": "5"},
                                     io.BytesIO(b"invalid key ODDSKEY999"))
    monkeypatch.setattr(providers.urllib.request, "urlopen", fake)
    r = providers.oddsapi_odds({"regions": "us"})
    assert r.status == 401 and r.body is None
    assert "ODDSKEY999" not in r.error and "***" in r.error
    assert r.headers["x-requests-used"] == "5"


def test_network_error_is_captured_not_raised(monkeypatch):
    monkeypatch.setenv("THE_ODDS_API_KEY", "NETKEY")

    def fake(req, timeout):
        raise TimeoutError("timed out")
    monkeypatch.setattr(providers.urllib.request, "urlopen", fake)
    r = providers.oddsapi_events()
    assert r.status is None and r.body is None and "TimeoutError" in r.error


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("SGO_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SGO_API_KEY"):
        providers.sgo_events({})


def test_truncated_response_is_captured_not_raised(monkeypatch):
    monkeypatch.setenv("THE_ODDS_API_KEY", "NETKEY")

    def fake(req, timeout):
        raise http.client.IncompleteRead(b"par")
    monkeypatch.setattr(providers.urllib.request, "urlopen", fake)
    r = providers.oddsapi_events()
    assert r.status is None and r.body is None and "IncompleteRead" in r.error


def test_sgo_keeps_earlier_pages_when_a_later_page_fails(monkeypatch):
    monkeypatch.setenv("SGO_API_KEY", "SGOKEY123")
    answers = iter([FakeResp({"data": [{"eventID": "a"}], "nextCursor": "c2"}), TimeoutError("slow")])

    def fake(req, timeout):
        a = next(answers)
        if isinstance(a, Exception):
            raise a
        return a
    monkeypatch.setattr(providers.urllib.request, "urlopen", fake)
    r = providers.sgo_events({})
    assert r.status == 200 and [p["data"][0]["eventID"] for p in r.body] == ["a"]
    assert r.error.startswith("partial: page 2 failed: TimeoutError")


def test_sgo_stops_on_a_repeated_cursor(monkeypatch):
    monkeypatch.setenv("SGO_API_KEY", "SGOKEY123")
    calls = []

    def fake(req, timeout):
        calls.append(req.full_url)
        if len(calls) > 5:
            raise AssertionError("pagination did not stop")
        return FakeResp({"data": [{"eventID": "a"}], "nextCursor": "same"})
    monkeypatch.setattr(providers.urllib.request, "urlopen", fake)
    r = providers.sgo_events({})
    assert r.status == 200 and len(r.body) == 2 and r.error.startswith("partial: pagination stopped")


def test_sgo_stops_at_the_page_cap(monkeypatch):
    monkeypatch.setenv("SGO_API_KEY", "SGOKEY123")
    n = iter(range(1000))
    monkeypatch.setattr(providers.urllib.request, "urlopen",
                        lambda req, timeout: FakeResp({"data": [], "nextCursor": f"c{next(n)}"}))
    r = providers.sgo_events({})
    assert len(r.body) == providers.SGO_MAX_PAGES and r.error.startswith("partial: pagination stopped")


def test_sgo_non_object_body_is_an_error(monkeypatch):
    monkeypatch.setenv("SGO_API_KEY", "SGOKEY123")
    monkeypatch.setattr(providers.urllib.request, "urlopen", lambda req, timeout: FakeResp([1, 2]))
    r = providers.sgo_events({})
    assert r.status == 200 and r.body is None and "list" in r.error


def test_oddsapi_non_list_body_is_an_error(monkeypatch):
    monkeypatch.setenv("THE_ODDS_API_KEY", "ODDSKEY999")
    monkeypatch.setattr(providers.urllib.request, "urlopen", lambda req, timeout: FakeResp({"message": "x"}))
    r = providers.oddsapi_odds({"regions": "us"})
    assert r.status == 200 and r.body is None and "dict" in r.error


# --- flatten ---------------------------------------------------------------

def load_fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def sgo_env():
    return {"request_id": "rid1", "provider": "sgo", "slot": "open", "captured_at_utc": "2026-10-21T15:00:00Z",
            "payload": [load_fixture("sgo_events.json")]}


def test_parse_odd_id_handles_players_and_combo_stats():
    assert flatten.parse_odd_id("points+rebounds-LEBRON_JAMES_1_NBA-game-ou-under") == (
        "points+rebounds", "LEBRON_JAMES_1_NBA", "game", "ou", "under")
    assert flatten.parse_odd_id("points-all-1h-ou-over") == ("points", "all", "1h", "ou", "over")
    assert flatten.parse_odd_id("garbage") == (None,) * 5


def test_sgo_rows():
    rows, schema = flatten.flatten(sgo_env())
    assert schema is flatten.SGO_SCHEMA and len(rows) == 5   # EVT1: 4, EVT2 (no odds): 0, EVT3 (junk): 1
    fd = next(r for r in rows if r["bookmaker_id"] == "fanduel" and r["event_id"] == "EVT1")
    assert fd == {
        "request_id": "rid1", "slot": "open", "captured_at_utc": "2026-10-21T15:00:00Z", "minutes_to_tip": 660.0,
        "event_id": "EVT1", "starts_at_utc": "2026-10-22T02:00:00.000Z",
        "home_team": "Los Angeles Lakers", "away_team": "Boston Celtics",
        "odd_id": "points-home-game-sp-home", "stat_id": "points", "stat_entity_id": "home", "period_id": "game",
        "bet_type_id": "sp", "side_id": "home", "bookmaker_id": "fanduel",
        "odds_american": -110.0, "line": -3.5, "available": True, "last_updated_at": "2026-10-21T15:00:00.000Z",
        "open_odds": -108.0, "open_line": -2.5, "close_odds": -112.0, "close_line": -4.0}
    dk = next(r for r in rows if r["bookmaker_id"] == "draftkings")
    assert dk["odds_american"] == 100.0 and dk["open_odds"] is None and dk["close_line"] is None
    half = next(r for r in rows if r["bookmaker_id"] == "betmgm")
    assert (half["period_id"], half["stat_entity_id"], half["line"]) == ("1h", "all", 112.5)
    prop = next(r for r in rows if r["bookmaker_id"] == "caesars")
    assert (prop["stat_id"], prop["stat_entity_id"], prop["line"], prop["available"]) == (
        "points+rebounds", "LEBRON_JAMES_1_NBA", 33.5, False)
    junk = next(r for r in rows if r["event_id"] == "EVT3")
    assert junk["stat_id"] is None and junk["odds_american"] is None
    assert junk["minutes_to_tip"] is None and junk["home_team"] is None


def test_oddsapi_rows_preserve_signs_and_books():
    env = {"request_id": "rid2", "provider": "oddsapi", "slot": "close@1830",
           "captured_at_utc": "2026-10-22T01:40:00Z", "payload": load_fixture("oddsapi_odds.json")}
    rows, schema = flatten.flatten(env)
    assert schema is flatten.ODDSAPI_SCHEMA and len(rows) == 6
    spreads = {r["outcome_name"]: (r["point"], r["price_american"]) for r in rows if r["market_key"] == "spreads"}
    assert spreads == {"Los Angeles Lakers": (-3.5, -110.0), "Boston Celtics": (3.5, -110.0)}
    assert {r["bookmaker_key"] for r in rows} == {"betrivers", "fanatics"}
    assert rows[0]["minutes_to_tip"] == 20.0 and rows[0]["point"] is None   # h2h has no point


def test_flattened_rows_write_to_parquet(tmp_path):
    rows, schema = flatten.flatten(sgo_env())
    store.write_parquet(tmp_path / "x.parquet", rows, schema)
    table = pq.read_table(tmp_path / "x.parquet")
    assert table.num_rows == 5 and table.schema.equals(flatten.SGO_SCHEMA)


# --- CLI -------------------------------------------------------------------

NOW = datetime(2026, 10, 21, 15, 15, tzinfo=UTC)   # 10:15 CT: the SGO open slot is due


def seed_schedule(root, fetched=NOW):
    events = load_fixture("oddsapi_events.json")
    store.atomic_write(root / "schedule" / "2026-10-21.json",
                       json.dumps({"fetched_at_utc": cli.iso(fetched), "events": {e["id"]: e for e in events}}).encode())


def fake_providers(monkeypatch, status=200):
    page, odds, calls = load_fixture("sgo_events.json"), load_fixture("oddsapi_odds.json"), []
    ok = status == 200

    def sgo(params, paginate=True):
        calls.append(("sgo", {**params, "_paginate": paginate}))
        return providers.Response(status, {}, [page] if ok else None, "" if ok else f"HTTP {status}")

    def oddsapi(params):
        calls.append(("oddsapi", params))
        return providers.Response(status, {"x-requests-used": "12", "x-requests-remaining": "488"},
                                  odds if ok else None, "" if ok else f"HTTP {status}")
    monkeypatch.setattr(providers, "sgo_events", sgo)
    monkeypatch.setattr(providers, "oddsapi_odds", oddsapi)
    return calls


def test_run_due_captures_logs_and_is_idempotent(tmp_path, monkeypatch):
    seed_schedule(tmp_path)
    calls = fake_providers(monkeypatch)
    assert cli.run_due(tmp_path, NOW) == 0
    log = store.read_log(tmp_path)
    assert sorted((r["provider"], r["slot"], r["outcome"]) for r in log) == [
        ("sgo", "open", "ok")]
    sgo_row = next(r for r in log if r["provider"] == "sgo" and r["outcome"] == "ok")
    assert sgo_row["objects"] == "3" and sgo_row["error"] == ""
    raw = tmp_path / sgo_row["payload_path"]
    env = store.read_raw(raw)
    assert env["request"]["params"]["startsAfter"] == "2026-10-21T15:15:00Z"
    assert env["request"]["params"]["started"] == "false"
    assert pq.read_table(raw.with_name(raw.name.replace(".json.gz", ".parquet"))).num_rows == 5
    n = len(calls)
    assert cli.run_due(tmp_path, NOW + timedelta(minutes=5)) == 0
    assert len(calls) == n and len(store.read_log(tmp_path)) == 1


def test_captured_at_is_response_time_not_run_start(tmp_path, monkeypatch):
    seed_schedule(tmp_path)
    fake_providers(monkeypatch)
    later = NOW + timedelta(seconds=90)
    monkeypatch.setattr(cli, "utcnow", lambda: later)
    cli.run_due(tmp_path, NOW)
    oks = [r for r in store.read_log(tmp_path) if r["outcome"] == "ok"]
    assert len(oks) == 1
    for r in oks:
        assert r["attempted_at_utc"] == cli.iso(NOW)
        assert store.read_raw(tmp_path / r["payload_path"])["captured_at_utc"] == cli.iso(later)
        assert cli.iso(later).replace(":", "-") in r["payload_path"]


def test_dry_run_reports_without_network_or_writes(tmp_path, capsys):
    seed_schedule(tmp_path, fetched=NOW - timedelta(hours=5))   # stale: a real run would refetch
    before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")}
    assert cli.run_due(tmp_path, NOW, dry_run=True) == 0
    out = capsys.readouterr().out
    assert "due sgo 2026-10-21 open" in out and "stale" in out
    assert {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")} == before


def test_auth_failure_exits_nonzero_and_logs_error(tmp_path, monkeypatch):
    seed_schedule(tmp_path)
    fake_providers(monkeypatch, status=401)
    assert cli.run_due(tmp_path, NOW) == 1
    errors = [r for r in store.read_log(tmp_path) if r["outcome"] == "error"]
    assert {r["provider"] for r in errors} == {"sgo"} and all(r["http_status"] == "401" for r in errors)
    assert not list(tmp_path.glob("*/date=*/*"))


def test_flatten_failure_keeps_raw_and_logs_ok(tmp_path, monkeypatch):
    seed_schedule(tmp_path)
    fake_providers(monkeypatch)

    def boom(env):
        raise KeyError("payload")
    monkeypatch.setattr(flatten, "flatten", boom)
    assert cli.run_due(tmp_path, NOW) == 1
    ok = [r for r in store.read_log(tmp_path) if r["outcome"] == "ok"]
    assert len(ok) == 1 and all(r["error"].startswith("flatten: KeyError") for r in ok)
    assert all((tmp_path / r["payload_path"]).exists() for r in ok)


def test_partial_sgo_capture_is_kept_and_fails_the_run(tmp_path, monkeypatch):
    seed_schedule(tmp_path)
    page = load_fixture("sgo_events.json")
    monkeypatch.setattr(providers, "sgo_events", lambda params, paginate=True: providers.Response(
        200, {}, [page], "partial: page 2 failed: TimeoutError: slow"))
    monkeypatch.setattr(providers, "oddsapi_odds", lambda params: providers.Response(200, {}, [], ""))
    assert cli.run_due(tmp_path, NOW) == 1
    sgo = [r for r in store.read_log(tmp_path) if r["provider"] == "sgo" and r["outcome"] == "ok"]
    assert len(sgo) == 1 and sgo[0]["error"].startswith("partial:") and (tmp_path / sgo[0]["payload_path"]).exists()


def test_multi_page_sgo_capture_counts_and_flattens_every_page(tmp_path, monkeypatch):
    seed_schedule(tmp_path)
    page = load_fixture("sgo_events.json")
    monkeypatch.setattr(providers, "sgo_events", lambda params, paginate=True: providers.Response(200, {}, [page, page], ""))
    monkeypatch.setattr(providers, "oddsapi_odds", lambda params: providers.Response(200, {}, [], ""))
    assert cli.run_due(tmp_path, NOW) == 0
    sgo = next(r for r in store.read_log(tmp_path) if r["provider"] == "sgo" and r["outcome"] == "ok")
    assert sgo["objects"] == "6"
    parquet = (tmp_path / sgo["payload_path"]).with_name(Path(sgo["payload_path"]).name.replace(".json.gz", ".parquet"))
    assert pq.read_table(parquet).num_rows == 10


def test_wrong_shape_sgo_response_is_logged_as_one_billed_object(tmp_path, monkeypatch):
    seed_schedule(tmp_path)
    monkeypatch.setattr(providers, "sgo_events", lambda params, paginate=True: providers.Response(
        200, {}, None, "unexpected body: list"))
    monkeypatch.setattr(providers, "oddsapi_odds", lambda params: providers.Response(200, {}, [], ""))
    cli.run_due(tmp_path, NOW)
    errs = [r for r in store.read_log(tmp_path) if r["provider"] == "sgo" and r["outcome"] == "error"]
    assert len(errs) == 1 and errs[0]["objects"] == "1" and errs[0]["error"] == "unexpected body: list"


def test_quota_guards_skip_requests(tmp_path, monkeypatch):
    seed_schedule(tmp_path)
    calls = fake_providers(monkeypatch)
    store.append_log(tmp_path, {"provider": "sgo", "attempted_at_utc": "2026-10-01T15:00:00Z", "objects": 2300,
                                "slot": "old", "outcome": "ok"})
    store.append_log(tmp_path, {"provider": "oddsapi", "attempted_at_utc": "2026-10-21T20:00:00Z", "quota_used": 448,
                                "slot": "schedule", "outcome": "ok"})
    cli.run_due(tmp_path, NOW)
    assert calls == []
    assert [r["provider"] for r in store.read_log(tmp_path) if r["outcome"] == "skipped_quota"] == ["sgo"]


def test_sgo_guard_reads_account_wide_usage_before_the_local_log(tmp_path, monkeypatch):
    seed_schedule(tmp_path)
    calls = fake_providers(monkeypatch)
    monkeypatch.setattr(providers, "sgo_objects_used", lambda: 2300)   # the betting app spent most of it
    cli.run_due(tmp_path, NOW)
    assert calls == []
    assert [r["provider"] for r in store.read_log(tmp_path) if r["outcome"] == "skipped_quota"] == ["sgo"]


def test_sgo_objects_used_reads_the_usage_endpoint(monkeypatch):
    monkeypatch.setenv("SGO_API_KEY", "k")
    body = {"data": {"rateLimits": {"per-month": {"max-entities": 2500, "current-entities": 1234}}}}
    monkeypatch.setattr(providers.urllib.request, "urlopen", lambda req, timeout: FakeResp(body))
    assert REAL_SGO_OBJECTS_USED() == 1234
    monkeypatch.setattr(providers.urllib.request, "urlopen", lambda req, timeout: FakeResp({"data": {}}))
    assert REAL_SGO_OBJECTS_USED() is None   # unreadable shape: the caller falls back to its own log


def test_schedule_refresh_merges_by_event_id(tmp_path, monkeypatch):
    seed_schedule(tmp_path, fetched=NOW - timedelta(hours=3))
    moved = {"id": "OA2", "commence_time": "2026-10-22T00:00:00Z",
             "home_team": "New York Knicks", "away_team": "Cleveland Cavaliers"}
    monkeypatch.setattr(providers, "oddsapi_events",
                        lambda: providers.Response(200, {"x-requests-used": "20"}, [moved], ""))
    sched, rc = cli.refresh_schedule(tmp_path, date(2026, 10, 21), NOW)
    assert rc == 0 and set(sched["events"]) == {"OA1", "OA2"}            # OA1 dropped by provider, kept
    assert sched["events"]["OA2"]["commence_time"] == "2026-10-22T00:00:00Z"   # tip change picked up
    assert cli.load_schedule(tmp_path, date(2026, 10, 21)) == sched
    assert store.read_log(tmp_path)[-1]["quota_used"] == "20"


def test_corrupt_schedule_cache_self_heals(tmp_path, monkeypatch):
    path = tmp_path / "schedule" / "2026-10-21.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"fetched_at_utc": "2026-10-21T2', encoding="utf-8")   # truncated mid-write
    monkeypatch.setattr(providers, "oddsapi_events",
                        lambda: providers.Response(200, {}, load_fixture("oddsapi_events.json"), ""))
    fake_providers(monkeypatch)
    assert cli.run_due(tmp_path, NOW) == 0
    assert set(cli.load_schedule(tmp_path, date(2026, 10, 21))["events"]) == {"OA1", "OA2"}


def test_corrupt_schedule_cache_is_logged_even_when_the_refetch_fails(tmp_path, monkeypatch):
    path = tmp_path / "schedule" / "2026-10-21.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(providers, "oddsapi_events", lambda: providers.Response(None, {}, None, "URLError: down"))
    cli.run_due(tmp_path, NOW)
    notes = [r["error"] for r in store.read_log(tmp_path) if r["slot"] == "schedule"]
    assert any(n.startswith("corrupt schedule cache 2026-10-21.json") for n in notes)
    assert any(n == "URLError: down" for n in notes)


def test_malformed_event_from_events_endpoint_is_dropped(tmp_path, monkeypatch):
    good = {"id": "OA2", "commence_time": "2026-10-22T23:30:00Z", "home_team": "A", "away_team": "B"}
    monkeypatch.setattr(providers, "oddsapi_events",
                        lambda: providers.Response(200, {}, [good, {"commence_time": "x"}, "junk", None], ""))
    sched, rc = cli.refresh_schedule(tmp_path, date(2026, 10, 21), NOW)
    assert rc == 0 and set(sched["events"]) == {"OA2"}


def test_manual_sgo_capture_uses_the_production_request_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path))
    calls = fake_providers(monkeypatch)
    assert cli.main(["--capture", "sgo"]) == 0
    params = calls[0][1]
    assert params["limit"] == "50" and params["_paginate"] is True
    assert params["startsAfter"] == "2026-10-21T15:15:00Z" and params["startsBefore"] == "2026-10-22T15:15:00Z"
    assert store.read_log(tmp_path / "raw" / "odds")[0]["slot"] == "manual"


def test_rebuild_parquet_matches_original(tmp_path, monkeypatch):
    seed_schedule(tmp_path)
    fake_providers(monkeypatch)
    cli.run_due(tmp_path, NOW)
    paths = sorted(tmp_path.glob("*/date=*/*.parquet"))
    before = [pq.read_table(p) for p in paths]
    for p in paths:
        p.unlink()
    assert cli.rebuild_parquet(tmp_path) == 0
    assert len(paths) == 1 and all(pq.read_table(p).equals(t) for p, t in zip(paths, before))


def test_outage_before_first_run_still_logs_started_games_missed(tmp_path, monkeypatch):
    # Yesterday's cache already listed today's noon game; no run happened until after it tipped,
    # so today's first /events call no longer returns it. Its slots must surface as missed.
    noon = {"id": "EARLY", "commence_time": "2026-10-21T17:00:00Z", "home_team": "A", "away_team": "B"}
    store.atomic_write(tmp_path / "schedule" / "2026-10-20.json",
                       json.dumps({"fetched_at_utc": "2026-10-20T15:00:00Z", "events": {"EARLY": noon}}).encode())
    monkeypatch.setattr(providers, "oddsapi_events", lambda: providers.Response(200, {}, [], ""))
    cli.run_due(tmp_path, datetime(2026, 10, 21, 19, 30, tzinfo=UTC))   # 14:30 CT
    missed = {r["slot"] for r in store.read_log(tmp_path) if r["outcome"] == "missed"}
    assert missed == {"open"}
