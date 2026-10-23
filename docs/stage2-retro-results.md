# Stage 2 retrospective results: early-line moneyline CLV, 2025-26 Kalshi (2026-10-23)

Pre-registration: [stage2-preregistration.md](stage2-preregistration.md). Plan:
[stage2-early-ml-plan.md](stage2-early-ml-plan.md), part A. Code: `tools/ml_clv_retro.py` (`preds`, `prices`, `score`);
tests in `tests/test_ml_clv_retro.py`. This was the one scoring run.

## Verdict: FAIL

**The line-free model has no information beyond Kalshi's early game-day price in 2025-26.** Bets it likes lose
value between entry and tip by about the cost of crossing the spread, and the market does not move toward the model
at all. Under the pre-registered decision, stage 2 stops here and part B (the forward ledger) is not built.

| Primary (E06, curve C) | Value |
|---|---|
| Bets (EV ≥ 3% at the ask) | 896 of 1,049 priced games |
| Mean gross CLV, `mid_close / ask − 1` | **−1.50%** (se 0.48%, 90% CI [−2.29%, −0.71%]) |
| Result | **FAIL** (CI entirely below 0) |

## Secondary (pre-registered, no decision attached)

| Entry | Probability | n bets | Gross CLV (90% CI) | Net of Kalshi fee | Realized ROI (90% CI) |
|---|---|---|---|---|---|
| E06 | curve C | 896 | −1.50% [−2.29, −0.71] | −5.69% | −2.3% [−10.2, +5.7] |
| E06 | Φ | 898 | −1.36% [−2.15, −0.58] | −5.54% | −3.5% [−11.5, +4.5] |
| E06 | Elo placebo | 847 | −2.48% [−3.23, −1.72] | −6.72% | −6.0% [−14.5, +2.4] |
| E10 | curve C | 908 | −1.40% [−2.11, −0.69] | −5.59% | −3.7% [−12.1, +4.6] |
| E10 | Φ | 906 | −1.34% [−2.06, −0.62] | −5.52% | −5.2% [−13.2, +2.7] |
| E10 | Elo placebo | 864 | −2.31% [−3.00, −1.62] | −6.57% | −7.9% [−16.2, +0.4] |

- **CLV slope** (logit move entry → close on the model's logit gap, all priced games):
  - E06: +0.008 (se 0.014, p 0.57).
  - E10: +0.010 (se 0.012, p 0.42).
  - The 10/08 spike found +0.38 from the open and +0.23 from 10am. Here the market's move is unrelated to the model.
- **Curve C vs Φ:** identical calibration over all 1,230 games (Brier 0.2066, log loss 0.6005 for both), as stage 1
  predicted.
- **Elo placebo:** worse than the model, as in the spike. The model is better than Elo but not better than the market.

## Context (not pre-registered)

- **The early market is already far better than the model.** On the 1,049 E06-priced games the Brier scores are:
  - model: 0.2074
  - Kalshi E06 mid: 0.1974
  - Kalshi close mid: 0.1959

  The 06:00 price captures most of what the close knows.
- **The bets are mostly underdogs because the model's probabilities are compressed.** The model's logit sd is 0.88,
  against 1.12 for the market. As a result, 79% of E06 bets are on the market underdog, and 540 of 896 bets claim
  EV ≥ 20%. A model whose probabilities are pulled toward 50% "finds" value on almost every underdog.
- **Gross CLV is about the spread.** The median half-spread at E06 is 1.0% of the ask; mean CLV is −1.5%. Bets pay the
  spread and get nothing back from line movement.
- **CLV by claimed EV** (E06):

  | Claimed EV | n | CLV |
  |---|---|---|
  | 3–6% | 74 | −1.6% |
  | 6–10% | 89 | −0.5% |
  | 10–20% | 193 | +0.5% |
  | ≥ 20% | 540 | −2.4% |

  There is no monotone pattern.

## Why this differs from the 10/08 spike

The test cannot separate these explanations; all are plausible.

1. **Venue and timing.** Kalshi game markets open about a week before tip and trade continuously. By 06:00 on game
   day the price is mature. The spike's "open" was TeamRankings' first snapshot the night before, a stale,
   aggregate retail number.
2. **Season.** The spike's data ended 2025-03; markets may have sharpened.
3. **Spike measurement limits** recorded in the plan: an uncommitted working tree, an aggregate price source of
   unknown origin, and an overnight open posted before the previous night's games finished.

What this run does establish: against a liquid, continuously traded early price, the current model adds nothing,
and the model is not good enough to be the source of an edge on full-game moneylines.

## Decisions

- **Stage 2 stops.** No forward ledger. Moving the daily-predictions cron to 05:45 CT is no longer needed for this
  test.
- **Both edge theses for full-game markets are now unsupported:** shape (stage 1) and model-driven timing (stage 2).
  The methodology's anchoring default `w = 0` stands for the moneyline too.
- **Candidate next direction (needs your decision; not started):** use the sharp price itself rather than a model.
  - Treat Kalshi's mid, or a consensus, as the fair price.
  - Bet an Illinois book only when its price beats that fair price by more than its vig and fee. This is the classic
    sharp-vs-soft line-shopping approach.
  - It needs per-book prices captured at the right times, and the odds collector is currently missing every slot.

## Deviations and notes

- Kalshi now reports prices in dollars (`0.7400`), not cents. Ratios such as CLV and EV are unchanged.
- The ESPN tip parser skipped events whose teams have no abbreviation yet (TBD playoff slots). It crashed on them in
  the `prices` step before any CLV was computed.
- Tip times: Kalshi and ESPN differ by more than 15 minutes on 1.9% of 1,448 games, so Kalshi times are allowed as
  the fallback. ESPN covered the 2025-26 regular season.
- 1,229 of 1,230 games have both Kalshi markets; 1,050 have a close that passes the quote filter.
- Before scoring, the line-free predictions were checked against `predict_today --check` on 2025-12-31, a fold
  start. They match to within 0.0044 points (JSON rounding).

## Reproduce

```bash
python tools/kalshi_history.py KXNBAGAME --candles --hours 24 --name day
python tools/ml_clv_retro.py preds
python tools/ml_clv_retro.py prices
python tools/ml_clv_retro.py score
```

Set `NBA_DATA_DIR` and `NBA_OUTPUTS_DIR` to the main checkout's folders when running from a worktree.
