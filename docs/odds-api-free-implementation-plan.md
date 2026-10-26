# NBA Prediction — The Odds API Free-Tier Collector Implementation Plan

Status: draft

Scope: `repos/nba_prediction` — local, forward-only NBA spread collection and a provenance-preserving adapter.
Companion proposal: [odds-api-free-collection-plan.md](odds-api-free-collection-plan.md) (reviewed 2026-10-06).

## Goal

Collect reproducible, free-tier The Odds API NBA pre-game spread observations from the first successful run onward. Persist raw payloads, normalized bookmaker rows, and request/quota audit records outside Git. Produce a separate, provenance-labeled candidate table for future research without changing the existing historical-odds file or the frozen experiment loader.

## Review outcome and scope corrections

The companion proposal is directionally sound. Its provider assumptions were rechecked against the current official documentation:

- The Starter plan currently includes 500 monthly credits and excludes historical odds.
- `GET /v4/sports/basketball_nba/events` is quota-free and returns event IDs, teams, and commence times.
- A bulk `/odds` call with one market (`spreads`) and one region (`us`) costs one credit, regardless of the number of games returned.
- API responses include the `x-requests-used`, `x-requests-remaining`, and `x-requests-last` headers.

Implementation must make the following corrections/clarifications:

1. Do not add this collector to `dataPrep.pipeline.run_pipeline()`. Its existing modes fetch/rebuild historical NBA artifacts, and normal experiments must never make a paid API call.
2. Do not alter `data/odds/nba_2008-2025.csv`. It is a historical source with a different schema and lineage.
3. Do not connect the output to `experiment.py` in this change. `load_data()` is marked frozen and currently mirrors/defaults missing prices; that conflicts with the requested matched-only provenance rules. A later, separately reviewed research path can consume the adapter output.
4. Use CSV, not Parquet, for version one. Pandas is already installed and the repository does not declare a Parquet engine. The schemas must remain stable enough to migrate later if needed.
5. Audit every HTTP attempt, including failures; headers are nullable when no HTTP response exists. Never record the API key or an unredacted request URL.
6. Use a local lock so overlapping Task Scheduler invocations cannot duplicate one eligible paid capture.

## Constraints and non-goals

- Collection is disabled unless an explicit collector CLI mode is invoked. No import-time network access.
- The API key is read only from `THE_ODDS_API_KEY` in the process environment. `.env.example` documents its name but this repository does not automatically load `.env` files.
- Data paths are derived from `NBA_DATA_DIR`, falling back to `<repo>/data`, matching the existing project convention. Raw and normalized outputs are ignored by Git.
- The collector uses `basketball_nba`, `us`, `spreads`, and American prices by default. It sends no unrelated markets or regions.
- No live API call, Task Scheduler registration, subscription change, historical backfill, or production deployment is included without explicit user approval.
- The adapter emits research candidates only. It must not claim that a missing, mirrored, defaulted, late, or post-tip value is a matched market observation.

## Proposed file layout

```text
src/odds_api/
  __init__.py
  config.py          # dataclass + environment/data-root resolution
  client.py          # redacted HTTP boundary, timeout/retry classification
  storage.py         # lock, atomic writes, CSV schemas, idempotent upserts
  normalize.py       # payload validation and bookmaker/outcome normalization
  schedule.py        # Central-time due/missed decision logic
  adapter.py         # game matching and opening/pre-tip candidate selection
  collect.py         # argparse entry point / orchestration
tests/
  fixtures/odds_api/ # synthetic payloads only
  test_odds_api_*.py
```

Runtime data layout under `NBA_DATA_DIR`:

```text
raw/odds_api/2026-10-21T15-00-00Z_us_spreads_<request_id>.json
odds/odds_api_spread_snapshots.csv
odds/odds_api_request_audit.csv
odds/odds_api_schedule_audit.csv
odds/odds_api_model_candidates.csv
logs/odds_api_collector.lock
```

The request ID is generated before each paid request (UUID or content-independent equivalent) and is included in the raw filename, the request audit, and every normalized row. A timestamp alone is not sufficient for uniqueness.

## Data contracts

### Request audit

Write one row per attempt with `request_id`, `attempted_at_utc`, `request_kind` (`events` or `odds`), sanitized parameters, `http_status`, `outcome`, `error_class`, `raw_payload_path`, `event_count`, `normalized_row_count`, and nullable quota headers. Do not put the API key in any column, error message, or log output.

### Raw payload

Write only a successful response body. Use a temporary file in the target directory and atomic rename after JSON validation. Save the exact provider payload plus a minimal non-secret envelope containing `request_id`, capture timestamp, and sanitized request dimensions.

### Normalized snapshots

Use the fields proposed in the companion plan, plus `sport_key`, `provider_captured_at_utc`, and `normalization_version`. `provider_captured_at_utc` is the collector time; `market_updated_at_utc` remains the bookmaker's reported timestamp. Keep only a spread market whose two outcomes unambiguously match the provider's home and away names. Reject, audit, and do not manufacture a row for malformed, duplicated, or one-sided markets.

The idempotency key is `(request_id, event_id, bookmaker_key, market)` for raw normalized rows. Reprocessing the same payload must leave the snapshot table byte-for-byte logically unchanged after its deterministic sort.

### Model candidate table

Emit one selected observation per joined game and candidate type (`opening`, `pre_tip`) with `game_id`, `candidate_type`, `odds_tier`, `source`, `selection_rule`, `bookmaker_key`, `captured_at_utc`, `commence_time_utc`, side-specific spread/price fields, and the raw `request_id`. Preserve the provider names and the normalized join keys in audit columns.

`odds_tier` is limited to `matched`, `late`, and `unmatched` for this adapter. It never emits `mirrored` or `default` because those are not provider observations.

## Implementation approach

1. **Establish the isolated package and configuration.** Add `src/odds_api` with an immutable configuration dataclass. Defaults: disabled, `America/Chicago`, `basketball_nba`, `us`, `spreads`, `american`, timeout 20 seconds, a bounded transient retry policy, daily paid-capture cap 5, and monthly stop at 450 used credits. Validate numeric offsets, allowed-book format, writable data root, and absent/blank key before a live call. Add `THE_ODDS_API_KEY=` to `.env.example` with guidance that it must be exported to the process.

2. **Implement safe storage before networking.** Create parent directories only under the resolved data root; add `.gitignore` rules for fallback `data/raw/odds_api/`, `data/odds/odds_api_*.csv`, and the lock. Implement a stale-safe exclusive lock, atomic JSON/CSV replacement, explicit schemas, UTC ISO-8601 serialization, deterministic sorting, and upsert-by-key. Unit-test the filesystem work with `tmp_path`.

3. **Build the redacted client and quota guard.** Use the standard library HTTP client so no new runtime dependency is required. Keep the key in the request query only at the transport boundary; all returned request metadata is sanitized. Before a paid odds request, read the most recent successfully parsed `x-requests-used`; reject the request when it is at least 450, and reject when the daily cap is already consumed. Retry only connection timeouts, 429, and 5xx responses with bounded exponential backoff; audit each attempt. Do not retry malformed JSON, 4xx credential/configuration failures, or an exhausted quota.

4. **Normalize fixtures and live payloads.** Add a pure function that traverses event → bookmaker → `spreads` market → outcomes and maps outcomes by exact provider team name. Preserve signs rather than deriving one side from the other. Validate timestamps and numeric points/prices. Add fixture cases for multiple books, an absent market, swapped outcome order, malformed outcomes, duplicate payload processing, and a provider-bookmaker timestamp.

5. **Implement schedule eligibility as a pure policy.** A quota-free events refresh supplies the current event IDs and commence times. The policy returns a decision record rather than calling the odds endpoint itself:
   - an opening capture is due once per Central calendar day at or after 10:00 AM only when at least one unstarted NBA event exists;
   - a pre-tip capture is due once for each distinct configured tipoff batch at its 20-minute mark, up to the paid daily cap;
   - a delayed run may make one immediate capture only if the nearest unstarted event is at least five minutes away; otherwise it records the batch as missed;
   - completed/missed/paid-capture decisions are persisted in the schedule audit so restarts and duplicate scheduled invocations are idempotent.

   Treat Central daylight-saving transitions and a changed provider commence time as test cases. Batch by the exact UTC commence timestamp, not by a rounded local time.

6. **Provide an explicit CLI, not a background service.** `python -m src.odds_api.collect --dry-run` reads local audits and reports the next due action without credentials or network. `--run-due` obtains the quota-free event schedule, applies the policy, and makes at most one permitted odds capture. `--capture-once` is an operator-only paid action that still enforces the key, quota, cap, lock, and audit rules. Print sanitized summaries suitable for Task Scheduler history. Document an optional five-minute Windows Task Scheduler trigger that runs `--run-due`; creation of that task remains manual and outside code.

7. **Build the adapter independently of the collector.** Read only the normalized CSV and an explicitly supplied processed-schedule path. Normalize NBA team aliases in a single, tested mapping; join by date plus home/away with a fail-closed ambiguity report. For each game, first prefer the configured reference book if it has a valid snapshot; otherwise calculate the median only from complete allowed-book observations captured at the same selection point. Select opening as the earliest valid snapshot at least two hours before tipoff and pre-tip as the latest valid snapshot at least five minutes before tipoff. Write nonqualifying records as `late`/`unmatched`, never as matched substitutes.

8. **Test the public behavior and documentation.** Add pytest coverage for configuration/key absence, dry-run no-network behavior, quota stopping, 429/5xx retry classification, sanitized audit rows, atomic/idempotent persistence, schedule decisions, DST handling, response parsing/sign preservation, team matching, reference-book/median selection, timing tiers, and no mutation of historical odds artifacts. Update `README.md` and `data/README.md` with local-only usage, data locations, and the fact that forward collection cannot validate historical seasons.

## Decisions required before implementation

1. Confirm the reference-book rule: `pinnacle` first when a complete, timely market is present; otherwise median across a fixed allowlist. Pinnacle availability must be checked on the account/region rather than assumed.
2. Confirm whether `pre_tip` should retain the proposed rule (latest snapshot at least five minutes before tipoff) or additionally require an upper bound such as 20 minutes before tipoff. The latter may reduce usable data on delayed jobs.
3. Confirm whether the daily cap of five includes the opening capture. This plan assumes **yes**, so a day can contain one opening plus at most four pre-tip captures.
4. Choose and document a lawful, separately labeled source if research needs 2023–26 historical spreads. Otherwise limit the first evaluation dataset to forward-collected games.

## Verification

- Automated: `python -m pytest tests/test_odds_api_*.py -q` with synthetic fixtures and temporary data roots.
- Automated: `python -m src.odds_api.collect --dry-run` with no key set; assert that it neither opens the network client nor writes data.
- Automated: run the parser twice on one fixture and assert no duplicate snapshot/audit row for the same raw request ID.
- Manual (no key): inspect CLI `--help`, sample dry-run output, audit CSV headers, and `git status --short`; no raw data, generated CSVs, or credentials may be tracked.
- Manual (requires explicit approval and a real key): run one `--capture-once`, confirm a single raw JSON file, one audit record, quota headers, opposing home/away points for a complete market, and a redacted console log. Re-run from the saved payload path to verify normalization idempotency without a network call.
- Integration (after sufficient forward collection): build candidates against an explicit schedule file and inspect unmatched/late reports before any separate experiment-loader change.

## Risks and rollback

- Free-plan limits, bookmaker availability, and API schemas can change. Keep the provider configuration and parser versioned, and recheck the official documentation immediately before the first live call.
- A 10:00 AM snapshot is an operational observation, not a bookmaker's true opening line. Persist both the policy label and actual capture timestamp.
- Task Scheduler can miss a run while the machine sleeps. The schedule audit must expose missed batches rather than silently relabeling late data as closing data.
- The current repository has local, untracked `data/csv/` artifacts. This implementation must not read, move, or add them to Git.
- Rollback is code-only: remove the scheduled task and stop invoking the CLI. Retain already collected raw/audit data locally for reproducibility; do not delete it automatically.
