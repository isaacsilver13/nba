# Results: does the lineup increment predict the closing line's error?

Run 2026-10-22 under `docs/lineup-market-encompassing-preregistration.md` (frozen before the run). Full output:
`outputs/ablation/lineup_market_encompass.json` (not committed). Regression:
`(home_margin - market) = a + b*E + g*D`, E = baseline minus market, D = +lineup minus baseline, errors clustered by date.

| Block | g (lineup increment) | se | one-sided p | games |
|---|---|---|---|---|
| Pooled 2022-2025 (primary) | +0.354 | 0.125 | 0.002 | 4,379 |
| 2022 / 2023 / 2024 / 2025 | +0.26 / +0.17 / +0.41 / +0.60 | 0.22-0.32 | 0.004-0.22 | 702-1,230 each |
| 2025-26 (replication) | +0.095 | 0.226 | 0.34 | 1,230 |

Baseline `b` is about 0 in every pooled block (+0.02 overall), so the baseline adds nothing beyond the close; the lineup
increment is the only part with a signal. The increment moves a prediction by 1.5 points (sd).
`+tracking` pooled g = +0.07 (p 0.26) and negative in 2025 and 2025-26; `+both` pooled g = +0.19 (p 0.02), negative in 2025-26.

**Registered verdict: +lineup supported** (pooled g > 0 with p < 0.05, and g > 0 in 2025-26).

## How much to believe it
- The 2025-26 replication meets the registered sign rule but its standard error (0.23) is larger than its estimate
  (0.095): it is consistent with both zero and the pooled 0.35. It does not independently confirm.
- This test reuses the predictions from the RMSE ablation, so it is a second look at the same data, not new data.
- Post-hoc, not registered: dropping early-season games (either team with fewer than 5 window games) leaves g = +0.23
  (p 0.03, n 4,941), so the result is not just a season-opener artifact.
- Size: g = 0.35 on a 1.5-point increment is roughly a 0.5-point edge on the games where the increment is typical,
  and an NBA point is worth about 3 percentage points of win probability, so about 1.5 points of win probability, against
  about 2.4 points of vig at -110. Statistically notable, economically marginal; no betting claim is made.

## Next test
A forward test on 2026-27 games, pre-registered before they accumulate, with the lineup features frozen as they are.
That needs 2026-27 play-by-play for the stint step first.
