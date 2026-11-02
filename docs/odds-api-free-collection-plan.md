# The Odds API Free-Tier Spread Collection Plan

**Project:** `repos/nba_prediction`  
**Created:** 2026-10-06  
**Scope:** Collect reproducible NBA pre-game spread snapshots going forward without a paid The Odds API subscription.

## Decision and constraint

Use The Odds API's live/upcoming odds endpoint with:

- sport: `basketball_nba`
- region: `us`
- market: `spreads`
- odds format: `american`

The free tier provides 500 credits per month. A request for one market in one region costs one credit, irrespective of the number of games returned. Historical endpoints are not available to free accounts, so this work **cannot** backfill NBA spreads for 2023 through the present from The Odds API. It begins a clean, timestamped collection from the first successful run onward.

Authoritative provider references:

- [Plans and free-tier limit](https://the-odds-api.com/)
- [NBA endpoint and response shape](https://the-odds-api.com/sports/nba-odds.html)
- [Quota-cost rules](https://the-odds-api.com/liveapi/guides/v4/index.html)
- [Historical access restriction](https://the-odds-api.com/historical-odds-data/)

## Collection policy

1. Register for a free API key. Store it only in the local `.env` as `THE_ODDS_API_KEY`; add the variable without its value to `.env.example`. Never place a key in source, committed data, logs, or a task definition.
2. Run a no-credit discovery request to the `/v4/sports/basketball_nba/events` endpoint each morning during the season. Use it only to identify event IDs, teams, and scheduled start times.
3. Capture spreads via `/v4/sports/basketball_nba/odds?regions=us&markets=spreads&oddsFormat=american` at:
   - one daily opening observation at 10:00 AM Central; and
   - 20 minutes before each distinct NBA tipoff-time batch, capped at five captures per calendar day.
4. If a job starts late, collect once immediately only when the nearest unstarted game is at least five minutes away; otherwise mark that batch as missed. Do not label a late observation as a closing line.
5. Track the response headers `x-requests-used`, `x-requests-remaining`, and `x-requests-last`. Stop scheduled pulls for the month at 450 credits to reserve 50 credits for diagnosis and reruns.

The expected seasonal usage is well below 500 credits: roughly 1 daily opening capture plus 2--5 tipoff-batch captures on an active day, or about 90--180 credits per 30-day month. The job must still enforce the hard 450-credit guard because schedule shape and provider behavior can change.

## Data contract

Keep raw provider payloads outside Git at `${NBA_DATA_DIR}/raw/odds_api/` using one JSON file per successful request. Use a UTC capture timestamp and deterministic filename, for example:

`2026-10-21T15-00-00Z_us_spreads.json`

Normalize each bookmaker/game snapshot to a parquet or CSV table at `${NBA_DATA_DIR}/odds/odds_api_spread_snapshots.*` with these fields:

| Field | Meaning |
| --- | --- |
| `captured_at_utc` | Timestamp of this API response |
| `event_id` | The Odds API event ID |
| `commence_time_utc` | Provider-supplied game start time |
| `home_team`, `away_team` | Provider team names, retained verbatim |
| `bookmaker_key`, `bookmaker_title` | Source sportsbook identity |
| `market_updated_at_utc` | Sportsbook market update timestamp |
| `home_spread`, `away_spread` | Spread points, with their provider signs |
| `home_price_american`, `away_price_american` | Spread prices |
| `region`, `market` | Request dimensions (`us`, `spreads`) |
| `request_id` | Stable identifier joining normalized rows to a raw payload and quota audit |

Store a separate request audit table with URL parameters excluding the API key, HTTP status, payload path, response row count, and all three quota headers. Treat the provider's event ID plus capture time and bookmaker key as the unique raw-observation key.

## Canonical model inputs

Do not overwrite the existing `nba_2008-2025.csv` data file or silently combine sources. Build a dedicated adapter that produces one provenance-labeled row per game:

1. Match game date and normalized home/away teams to the processed NBA schedule.
2. Select one configured reference book first (default: `pinnacle` when present); otherwise use the median spread across the allowed U.S. books.
3. Define `opening` as the earliest snapshot at least two hours before tipoff and `pre_tip` as the latest snapshot captured at least five minutes before tipoff.
4. Emit `odds_tier=matched` only when the selected snapshot meets that timing rule; record `unmatched` or `late` otherwise. Never substitute `-110` or mirror one side into a market-quality row.
5. Preserve `source=the_odds_api_free_live`, `bookmaker_key`, `captured_at_utc`, and selection rule in the model-ready output so research runs can filter by provenance.

The free plan can establish data for future seasons. A separate, clearly labeled historical source is still required to evaluate 2023--26 seasons; acquiring that is deliberately out of scope for this no-cost plan.

## Implementation steps

1. Inspect existing ingestion configuration and add a small provider module under `src/ingest/` following its established paths and configuration style.
2. Add a typed configuration object for region, market, allowed books, daily cap, credit stop threshold, and time offsets. Default all network collection to disabled so ordinary experiments never make live requests.
3. Implement the collector with explicit timeouts, retry only for transient failures, redacted logging, atomic raw-payload writes, and idempotent normalization.
4. Implement the quota guard before every paid request using the last known audit value, then update the audit from response headers after every attempt.
5. Add a local scheduler entry point that accepts `--dry-run`, `--capture-once`, and `--run-due`. Use Windows Task Scheduler or a manually started local process only after a dry run confirms it would make the intended request; do not add a cloud scheduler or external deployment.
6. Add source-adapter logic that joins collected records to NBA game IDs and produces opening/pre-tip candidates without modifying current historical odds artifacts.
7. Add synthetic JSON fixtures with no API key and tests for response parsing, sign preservation, deduplication, quota stop behavior, schedule eligibility, and late/missed classification.
8. Run a manual smoke test using a real key only after the user approves the network call. Verify the raw payload is outside Git, the key is redacted, quota headers are written, and one normalized snapshot has opposing home/away points.

## Acceptance criteria

- A dry run makes no network request and reports which capture, if any, is due.
- A single allowed capture creates one raw JSON payload, request-audit row, and normalized bookmaker/game rows with the same request ID.
- The collector never runs after the 450-credit safety threshold and does not expose the key in output.
- Re-running the same payload is idempotent.
- Model-ready odds retain source, bookmaker, capture time, and timing classification; no historical rows are misrepresented as having come from the free API.
- `git status` remains free of raw odds payloads, credentials, and generated data.

## Decisions needed before implementation

- Confirm the reference-book rule: Pinnacle-first with median-U.S.-book fallback is proposed.
- Confirm whether a captured snapshot 5--20 minutes before tipoff is acceptable as the operational definition of `pre_tip` on the free tier.
- Choose a separate lawful source for 2023--26 historical evaluation, or explicitly limit early research to forward-collected data.
