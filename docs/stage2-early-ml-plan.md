# nba_prediction — Stage 2: early-line moneyline test

Status: complete — part A failed its gate; part B not built (2026-10-23, [results](stage2-retro-results.md))
Scope: this repository. Methodology: [methodology.md](methodology.md), stage 2.

## Goal

Test whether the line-free model's moneyline probability beats **early** prices: does the market move toward the
model between entry and tip (CLV)? Pricing uses curve C from stage 1. The work runs in two parts:

- **A. Retrospective gate on 2025-26**, using Kalshi's free hourly game-winner prices. Build nothing forward unless
  this passes.
- **B. Forward ledger.** One pre-tip row per priced moneyline, rebuilt from immutable, timestamped inputs.

## What exists today (checked 2026-10-23)

| Piece | State |
|---|---|
| Line-free predictions (`tools/predict_today.py`) | Runs daily on Actions at 09:30 CT. Publishes `predictions-<date>.json` with `generated_at` to the private data repo's `state-<date>` release, then POSTs it to bet-watch. Working. |
| Odds collector | **Not capturing moneylines.** Since 10/15 it has 0 of 3 open slots and 0 of 26 close slots; runs land hours apart, so the 5-minute cron-job.org trigger is apparently not firing. The Odds API is over quota (504 of 500 used, −4 left). Only 3 SGO afternoon props pulls succeeded. |
| Kalshi `KXNBAGAME` | Game-winner markets open about a week before tip. History for 2025-26 is free and needs no key. Hourly candles can be pulled after the fact, so no capture can be missed. |
| betting_aggregator paper ledger | Pre-tip, quote-linked moneyline paper portfolio (11 profiles, 3% net EV, quarter Kelly). Local and isolated only. **Records no close or CLV.** Out of scope here. |

### Evidence check

- The 10/08 spike (+4.3% CLV at the open, +1.4% at 10am) ran with the raw `home_margin` target, after the 10/07
  leakage audit, so it is not contaminated by the target leak below. It still has three limits:
  - It ran on an uncommitted working tree.
  - Its "open" price was posted before the previous night's games finished, which is look-ahead for teams on a
    back-to-back.
  - Its data ends 2025-03.
- Since 10/20 `experiment.py` predicts the market's error and adds back the **closing** spread
  ([experiment.py:644](../experiment.py)). So `tools/encompass_check.py` now produces predictions that contain the
  close. **It must not be used for any CLV replication.** Use `predict_today.py`'s raw-margin path instead.

## Decisions (2026-10-23)

- Replicate on 2025-26 first, as a gate.
- Entry times: **~06:00 CT** (right after the previous night's games are ingested) and **10:00 CT**.
  - Predictions use no same-day information, so a single run at ~05:45 CT serves both entries.
  - This moves the cron-job.org time from 09:30 to ~05:45 CT, which is **your action**.
  - bet-watch's 10:00 CT job still finds the file.
- The reference close is Kalshi's last pre-tip mid. Illinois book prices are added whenever the collector captures
  them.

## Approach

### A. Retrospective gate (2025-26)

1. **Pre-register** `docs/stage2-preregistration.md` before pulling any prices.
   - Entries: the first Kalshi candle at or after 06:00 CT, and the first at or after 10:00 CT, on game day.
   - Bet rule: model EV ≥ 3% at the entry ask.
   - Primary metric: mean gross CLV of bets at the 06:00 entry, `p_close_mid / ask − 1`, with SE clustered by date.
     Pass if the 90% CI is above 0.
   - Secondary metrics:
     - CLV at 10:00
     - CLV net of Kalshi's taker fee, `0.07·P·(1−P)`
     - CLV slope
     - Elo placebo
     - curve C vs Φ
2. **Kalshi pull.** Add `KXNBAGAME` to `tools/kalshi_history.py`, with hourly candles over the whole pre-game day
   rather than just the last candle.
   - Take tip times from the ESPN schedule, joined on date and teams. Don't infer them from Kalshi expiry times.
   - Filter on a two-sided book and spread ≤ 5¢.
3. **Line-free walk-forward predictions for 2025-26.** New `tools/noline_walkforward.py`.
   - Reuse `predict_today._fit` and `EXCLUDE`, the raw `home_margin` target, and `experiment.py`'s fold sizes.
   - Features for date d come only from games before d (the slate builder `run_features --slate`, covered by the
     `RUN_PIT=1` tests).
   - Save `pred_margin`, plus the model's residual sd `s` fit on earlier folds only.
4. **Win probability from curve C.** `p = Σ_{x>0} pmf(pred_margin, s, ecdf_C, no_tie=True)`.
   - Reuse `pmf`, `fit_scale` and `design` from `tools/curve_calibration.py`. Expose the stage 1 standardized ECDF
     as `curve_c(train_df)`, fit on close residuals from seasons before 2025-26.
   - Log Φ(pred/s) next to it.
   - Expect the two to differ by under 1 pp; stage 1 showed shape matters little near the center.
5. **Score once** and write `docs/stage2-retro-results.md`.
   - Fail → record it, update memory, and stop. Part B is not built.

### B. Forward ledger (only if A passes)

6. **Earlier predictions.** You change the cron-job.org daily-predictions job to ~05:45 CT. No code change is needed:
   the `complete-<date>.txt` marker keeps the 15:05/16:05 UTC backup triggers idempotent.
7. **`tools/ml_ledger.py`** rebuilds the ledger from immutable inputs, not from a live writer.
   - Inputs:
     - `predictions-<date>.json` release assets
     - Kalshi live and historical candles
     - SGO snapshots from the data repo, when present
   - A row is valid only if `generated_at < price_seen_at < tip`.
   - Fields follow the methodology's ledger schema, with entries E06 and E10.
     - Close: last pre-tip price, both the Kalshi mid and the same book's close.
     - Results: the ingested final scores.
     - Fees: Kalshi's taker formula and the Illinois per-book fees from config.
   - Output: `$NBA_OUTPUTS_DIR/ledger/ml_ledger.csv`, plus a summary of mean CLV ± SE by entry and venue, n bets, and
     stop/go status.
   - Rebuilding from timestamped inputs makes "pre-tip" structural: a late run cannot add a look-ahead row.
8. **Forward pre-registration** uses the same rule as A. It starts on the first slate after steps 6–7 ship.
   - Stop if mean CLV < 0 after 100 bets.
   - Go to real money if CLV ≥ +1% at ≥ 200 bets.
9. **Collector (optional, parallel).** You check why the cron-job.org odds trigger isn't firing (its key is in
   Bitwarden). The Odds API stays off until its monthly reset, so SGO alone covers Illinois books until then.

## Constraints and non-goals

- No change to `experiment.py`'s FROZEN section or `program.md`.
- No betting_aggregator changes. Feeding it curve-C probabilities and adding a close and CLV to its paper ledger is
  a separate follow-up in that repo.
- No paid data. Kalshi is a measurement venue here, not a recommendation to bet there while its Illinois status is
  in litigation.

## Verification

- Unit tests (no data):
  - ledger validity: a row is rejected when the price precedes `generated_at` or follows tip
  - CLV and fee arithmetic
  - curve-C win probability: symmetric at pred = 0, increasing in pred, exactly 0.5 at a pick'em
- `RUN_PIT=1 python -m pytest tests -q`, covering the slate features used by the walk-forward.
- Spot-check: for one 2025-26 date, `noline_walkforward` matches `predict_today --date D --check` on a fold-start
  date.
- `python -m pytest tests -q` passes.

## Risks / open items

- **Kalshi liquidity at 06:00 CT may be thin.** If fewer than ~300 bets pass the spread filter, the retrospective
  test is underpowered. Report n before scoring.
- **Fees.** Kalshi's taker fee is about 3.5% of cost at 50¢, so net CLV will be much lower than gross CLV. Gross CLV
  tests information; Illinois books, with flat fees, decide bettability.
- **Venue shift.** The spike measured CLV against TeamRankings' aggregate, while this test uses Kalshi. A pass or fail
  here may partly reflect the venue.
