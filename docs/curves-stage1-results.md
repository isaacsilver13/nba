# Stage 1 results: market-anchored spread and total curves (2026-10-23)

Pre-registration: [curves-stage1-preregistration.md](curves-stage1-preregistration.md). Code:
`tools/curve_calibration.py`, tests in `tests/test_curve_calibration.py`.

## Verdict

**The shape thesis is not supported for full-game spreads or totals.** On the holdout, a curve whose width and shape
depend on the game prices alt lines no better than a generic fixed-width curve. The two curves differ by under 1.5
percentage points on 90% of alt rungs, which is less than alt-line vig. Both locked curves are well calibrated.

| Holdout 2023–2025 (n = 3,960) | Margin (pick C) | Total (pick C) |
|---|---|---|
| H1: RPS gain over generic A, 90% CI | 0.0020 [−0.0025, +0.0065]: **FAIL** | −0.0004 [−0.0044, +0.0038]: **FAIL** |
| Calibration: max \|z\| over rungs ±15 | 1.47: **PASS** | 2.50: **PASS** |
| Room: \|p_C − p_A\| on rungs ±3..10, median / 90th pct | 0.65 / 1.44 pp: **FAIL** (needs ≥ 2) | 0.46 / 0.91 pp: **FAIL** |

H1 and room both fail, so under the pre-registered rules **the shape thesis is parked for full-game spreads and
totals**. The pre-registration did not say which curve to use for pricing when H1 fails. The choice is below.

## Development (seasons 2012–2022, n = 13,901)

- C had the lowest mean RPS for both quantities, so it was locked for the holdout.
  - Margin: A 6.9385, B 6.9367, C 6.9340. C's gain over A was 0.0045 [0.0023, 0.0067].
  - Total: A 9.9197, B 9.9200, C 9.9168. C's gain over A was 0.0029 [0.0001, 0.0057].
- Development already showed the room problem: 90th percentiles of 1.91 pp (margin) and 1.56 pp (total).
- The generic curve A miscalibrates margin rungs in development (max |z| 4.74) and only borderline in the holdout
  (2.78). C stays under 2 in both.

## What the curves say about the closing line

- **The close is centered.** Mean margin residual in development is −0.13 (se 0.11); in the holdout it is +0.09
  (se 0.21).
- **The width barely depends on the game.** Fitted log-sd slopes are about +0.03 per 10 points of total and −0.03 per
  5 points of spread, for both margin and total. A game with a total 30 points higher has a margin sd only about 9%
  wider. Larger spreads have slightly *narrower* outcomes.
- **Totals have an overtime tail.** The largest total residuals (+84 to +112) are all overtime games. The empirical
  curve C captures this tail; a normal curve cannot.
- **The spread curve reproduces the moneyline.** P(home win) from curve C has a Brier score of 0.2044 vs 0.2041 for
  the de-vigged closing moneyline in development (holdout 0.2238 vs 0.2234). The moneyline in this file is missing
  for most 2023–2025 games, so the holdout comparison covers only 664 of 3,960 games.

## Not pre-registered (observations only)

- The over at the closing total hit slightly more often than curve C predicts:
  - Development: 49.9% observed vs 48.8% predicted (z = +2.47).
  - Holdout: 51.0% observed vs 49.0% predicted (z = +2.50).
  - Mean total residuals are +0.39 (development) and +0.56 (holdout) points.

  A 51% over rate is below the 52.4% break-even at −110, and this was not a registered test. Treat it as a lead at
  most.

## Decisions

- **Pricing curve: C** (conditional empirical) for both margin and total. It is no worse than A on RPS, better
  calibrated on margin rungs, and the only one of the three that carries the overtime tail. This is a choice made
  after the holdout, not a registered result.
- **Shape is parked** for full-game alt spreads and totals. Any remaining alt-line edge would have to come from books
  pricing alt lines worse than a generic fair curve, for example with heavy or inconsistent vig. Only forward
  per-book alt prices can show that. Compare each book's alt ladder with curve C, and drop books whose alt vig
  exceeds the possible gain.
- **Next priority:** stage 2, the early-line moneyline forward test. It uses curve C's P(M > 0) to convert between
  spreads and moneylines.
- **Untested derived markets:** halves and quarters (the Voulgaris case) and player ladders. Stage 1 says nothing
  about them.

## Deviation from the pre-registration

The three games with no closing spread were dropped from the total analysis as well as the margin analysis, because
model B's width uses |spread|. The first development run crashed on them; no result was produced from it.

## Reproduce

```bash
python tools/curve_calibration.py
python tools/curve_calibration.py --holdout --pick-margin C --pick-total C
```

Set `NBA_DATA_DIR` to the main checkout's `data` folder when running from a worktree. Each run takes about a minute.
