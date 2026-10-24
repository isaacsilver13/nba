# nba_prediction — Sharp-vs-soft line shopping (scope)

Status: draft (2026-10-23)
Scope: this repository's odds collector and a research ledger. Live alerts or bet-watch integration would be a later,
separate change in betting_aggregator.

## Goal

Find Illinois sportsbook prices that are better than a **fair price taken from a sharper market**, with no model.
Bet only when a book's price beats fair value by more than its vig and per-bet fee. Judge the strategy by CLV against
the fair close, under a pre-registration, before risking money.

This follows from stages 1–2 ([methodology.md](methodology.md)). The model adds nothing beyond the market's own
price, so any edge has to come from **disagreement between markets**: an Illinois book that is slow, shaded or an
outlier compared with a sharper reference.

## Why it might work, and why it might not

| For | Against |
|---|---|
| Retail books move at different speeds; one book is often an outlier. On 10/22 at 4:55pm CT, CLE@PHI was +125 at ESPN BET and +115 to +120 elsewhere: about +2% EV against the cross-book consensus. | Kalshi isn't Pinnacle. If the reference is not sharper than the books, "beating" it is noise. Consensus fair value built from the same soft books is partly circular. |
| No model to be wrong; the method is simple and auditable. | Books limit accounts that win this way, often quickly. The edge decays with account health. |
| Kalshi's mid is free, timestamped and liquid (1–2¢ spreads on game markets). | Illinois per-bet fees ($0.50 at FanDuel and DraftKings below their waiver thresholds) cost 5% on a $10 bet. Stakes must clear the waiver, or bets must go to fee-free books. |

## Step 0: fix the odds collector (prerequisite)

**Diagnosis (2026-10-23):**
- Since 10/15 the collector has captured 0 of 3 opening snapshots and 0 of 26 close snapshots. Only three afternoon
  props pulls succeeded.
- **Root cause:** the cron-job.org job that triggers it every 5 minutes (job 8613747) is **disabled**, and it still
  posts to the old repo name `isaacsilver13/nba_prediction`. The repo is now `isaacsilver13/nba`.
- Without that job, the only trigger is GitHub's backup schedule (`17,47 * …`), which actually fired 4–6 times a day
  at irregular hours. Every capture window opened and closed between runs.
- **The Odds API key is over quota** (504 of 500 used). bet-watch reads the same variable name, `THE_ODDS_API_KEY`,
  and is the likely consumer. Collector slots for that provider stay off until it resets or gets a separate key.

**Done 2026-10-23:**
- The job was re-pointed, re-enabled, and its hours set to 09:00–23:55 Central. It dispatched on time (HTTP 204), and
  the close slot fired at 14:50 CT.
- SGO still returned **empty**. Close slots ask for games within tip ± 5 minutes, and The Odds API's pre-game tip
  times run about 10 minutes after the official tip that SGO uses.
- Fix: branch `fix/odds-collector-sgo-close-pad` widens the pad to 15 minutes. It must merge to `main` before it
  takes effect.

**Original fix list:**
1. Point cron-job.org job 8613747 at `…/repos/isaacsilver13/nba/actions/workflows/odds-collector.yml/dispatches`
   and re-enable it. **This is your call: it is persistent config.**
2. Verify:
   - one dispatch returns HTTP 204
   - the next scheduled slot shows `outcome=ok` in `requests.csv`
   - `gh run list` shows runs about every 5 minutes inside the window
3. Confirm the `settle` capture (next morning) fills SGO's per-book `close_odds`. It has never run this season.
   - If it does, every slate costs about two captures: each book's open (`open_odds`), the snapshots, and its
     close.
   - If it doesn't, close captures must land in the 20-minute pre-tip window. The fixed trigger makes that feasible.

**What SGO gives (checked on the 10/22 capture):**
- Per-book prices for DraftKings, FanDuel, Caesars, BetMGM and ESPN BET (all available in Illinois), plus Bovada,
  Unibet, PointsBet and William Hill.
- Each row carries `open_odds` (the book's own opening price) and `last_updated_at`.
- Markets covered: moneylines, spreads, totals, quarters and halves, player props, double-doubles.
- Filter moneylines on `stat_id == "points"` to drop first-to-score and similar markets.
- **Not covered:** Fanatics, bet365 and BetRivers.

## Step 1: research ledger (moneyline first)

`tools/line_shop_ledger.py` rebuilds a ledger from immutable inputs, following the stage 2 design: SGO snapshots,
plus Kalshi `KXNBAGAME` hourly candles pulled after the fact.

- **Rows:** one per (snapshot, game, side, Illinois book).
  - Price, `last_updated_at` and the snapshot time.
  - Fair probability: Kalshi mid in the candle covering the snapshot, plus a de-vigged consensus of the other books
    as a secondary reference.
  - EV after the book's fee at a configured stake.
  - Fair close: Kalshi last pre-tip mid, plus the consensus close.
  - CLV and result.
- **Validity:** a row is valid only if the price was seen before tip, and the Kalshi quote used for fair value comes
  from before the snapshot.
- **Rollout order:** moneylines (cleanest reference), then spreads and totals (needs a Kalshi spread/total series or
  consensus only; to check), then player props (Kalshi has `KXNBAPTS`, `REB`, `AST` and `3PT` ladders).

## Step 2: pre-registered forward test

- **Bet rule:** best Illinois price with EV ≥ X% against the Kalshi mid, after fee, at stake S. X and S are set in the
  pre-registration, before any forward data is seen.
- **Primary metric:** mean CLV against the Kalshi close mid. Secondary: CLV against the consensus close, per-book
  breakdown, and ROI.
- **Gates (the methodology's):** stop if CLV < 0 after 100 bets; real money only at CLV ≥ +1% over 200 bets.
- **Before the forward test:** a first read of opportunity frequency (how often any Illinois book beats Kalshi by more
  than X%) from the first week of captures.

## SGO quota budget

SGO's free tier is 2,300 objects a month, one object per game per capture.

| Plan | Captures per slate | Objects/day (8 games) | Objects/month |
|---|---|---|---|
| Today's schedule: open, props, about 3 close groups, settle | ~6 | ~48 | ~1,450 |
| Line shopping: 08:00, 11:00, 14:00, 17:00 CT, about 3 close groups, settle | ~8 | ~64 | ~1,900 |

Busy nights (12–15 games) push past the cap, and the collector's quota guard stops at 2,300. More snapshot times need
a paid tier or Kalshi-only fair prices with fewer book snapshots.

## Constraints and non-goals

- No paid data, and no automatic bet placement.
- No betting_aggregator change in this scope.
- Kalshi is the fair-price reference, not a betting venue, while its Illinois status is in litigation.

## Decisions needed

1. **Re-enable and re-point cron-job.org job 8613747** (persistent config change).
2. **Stake size.** It decides whether FanDuel and DraftKings are usable: their fees are waived at ≥ $25 (FanDuel)
   and ≥ $50 singles (DraftKings). Below that, only fee-free books (BetMGM, Caesars) make sense.
3. **Snapshot schedule:** keep today's or move to the line-shopping schedule above.
