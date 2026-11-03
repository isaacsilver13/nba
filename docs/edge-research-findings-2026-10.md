# Edge research: moneyline CLV and Kalshi player props (2026-10-08)

Two spikes after the 2026-10-08 encompassing test showed experiment.py's spread model adds ~nothing beyond the
closing spread. Question for both: is there an exploitable edge, measured by CLV / encompassing rather than noisy ROI?

> **Correction (2026-11-03): the moneyline CLV "go" below is retracted. It is a leak artifact.**
> `experiment.py` trains on the market's error (`home_margin + spread_signed`) and adds the closing line back
> (`_run_fold`), so `pred_ensemble` is the closing line plus a ~3pt model deviation (corr with the line 0.92, 85% of
> its variance). Dropping `spread_signed` from the *features* does not remove it from the *prediction*, so
> `p_model = Φ(pred/s)` is "close + noise" and mechanically predicts open→close moves, with an edge that decays as the
> entry nears the close (the +4.3% / +1.4% / 0 pattern). Re-run on the same 4,020 games (2022-02 to 2025-04):
>
> | Variant | CLV at open | CLV at 10am |
> |---|---|---|
> | Model as run in Spike 1 | +4.4% | +1.5% |
> | Placebo: closing line + *shuffled* model deviations (no information) | +4.0 to +4.5% | +1.1 to +1.4% |
> | Raw-margin target, no line in features or target | **−2.0%** | **−2.9%** |
>
> The leak-free model has log loss 0.635 vs the market's 0.605 and RMSE 14.2 vs 13.4, so it has no early-price edge.
> The Elo placebo was never a fair control (Elo is simply a worse forecaster). The CLV SE of ~0.25–0.3% below is for
> n≈2,700; at n=200 it is ~0.9%, and confirming +1% CLV would take ~1,400 bets for 2 SE. Kalshi results (Spikes 2–3)
> are unaffected. Spike 1 is kept for the record only. Reproduce the clean run by patching `_run_fold` to fit
> `home_margin` directly with no `- line_te` offset, then `python tools/encompass_check.py --preds <that csv>`.

**Bottom line**

| Question | Verdict |
|---|---|
| Does the model beat the moneyline **close**? | No. Encompassing slope 0.01 (se 0.09). |
| Does it beat **earlier** moneyline prices (CLV)? | ~~Yes, robustly.~~ **No (retracted 2026-11-03).** The apparent CLV (+1.4% at 10am, +4.3% at open) comes from the closing line baked into the prediction; a closing-line + shuffled-noise placebo reproduces it, and a leak-free model gets CLV −2% to −3%. |
| Does a simple points model beat Kalshi **early** (3–24h before tip) prop prices? | **No.** Same result as pre-tip: encompassing slope −0.05 to +0.10, all n.s.; taker ROI −6% to −18%. See Spike 3. |
| Does a simple points model beat Kalshi pre-tip prop prices? | **No.** Kalshi's pre-tip mid is well calibrated and better than the model (Brier 0.169 vs 0.175); encompassing slope +0.008 (se 0.058); taker ROI −7% to −12% after fees. |

**Recommendation**

- **~~Go: forward-test early moneylines~~ — retracted (see Correction).** There is no demonstrated early-price edge.
  Keep the odds collector logging timestamped per-book prices; that data is still what any future test needs. A
  forward test is only worth running on a model whose prediction contains no closing line, and with a bar sized to the
  real SE (~0.9% at n=200, so ~1,400 bets for 2 SE at +1% CLV).
- **No-go: building a full player-prop model aimed at Kalshi pre-tip prices.** A minutes × rate baseline with
  teammate-out adjustment is close to what the market already prices. The market beats it, and fees plus the ~3¢
  regular-season spread remove any margin. A fuller model would mostly need *faster information* (late scratches,
  minutes restrictions), not better modeling.
- **Done (Spike 3): early Kalshi prop prices.** Same answer as pre-tip: no edge. Props stay closed.

---

## Spike 1: moneyline CLV and encompassing

> **Superseded by the Correction at the top: the CLV results in this section are a closing-line leak artifact.**
> The "Leak found first" paragraph below removed the line from the features but not from the target/prediction.

**Data.** experiment.py's out-of-sample predictions, 14 walk-forward folds of 300 games (2021-12 to 2025-03).
Fold 1 is dropped because it has no earlier data to calibrate on, leaving 3,846 games with valid TeamRankings moneylines
(`data/csv/game_odds.csv`, overround 1.00–1.08 at all three entries; ~1.5% of games lack a quote or have a broken one). Prices:

- **open**: TR's first snapshot (it equals the earliest timestamped row 99% of the time), median ~9pm ET the night before.
- **10am**: last snapshot at or before 10:00 ET on game day (9:00 EST in winter, because `game_date` is stored as
  04:00Z).
- **close**: TR's close.

All three are de-vigged multiplicatively.

**Model win probability.** `p = Φ(pred_ensemble / s)`, with `s` fit by probit on *earlier folds'* home-win
outcomes (s ≈ 13–15.5). The `sigma_ensemble` in experiment.py is calibrated for spread *cover* and cannot be reused
for win probability.

**Leak found first.** experiment.py uses `spread_signed` (the **closing** spread) and `is_home_favorite` as model
features. Any comparison against prices earlier than the close is therefore look-ahead. All CLV numbers below come
from re-running the walk-forward with features removed at runtime (no edit to experiment.py):

- **A**: drops `spread_signed` and `is_home_favorite`.
- **B**: A plus the same-day injury features (`pregame_missing_top1/2`, `pregame_minutes_out`,
  `pregame_usage_out`), which may only be known after the open.

A and B agree, so the B (strictest) numbers are reported.

| Test (variant B, n=3,846) | Result |
|---|---|
| Log loss: model / open / 10am / close | 0.612 / 0.608 / 0.604 / 0.597 |
| (b) Encompassing vs close: `(win − p_close) ~ (p_model − p_close)` | slope **+0.008** (se 0.090), so no information beyond the close |
| (a) CLV slope from open: `logit move open→close ~ logit(p_model) − logit(p_open)` | **+0.381** (se 0.012); placebo (shuffled) +0.004 |
| CLV slope from 10am | **+0.230** (se 0.010) |
| EV>3% bets at open (n=2,748) | mean CLV **+4.3%** (se 0.36%), ROI +4.0% (se 2.7%) |
| EV>3% bets at 10am (n=2,729) | mean CLV **+1.4%** (se 0.26%), ROI +3.2% (se 2.8%) |
| Elo-only placebo, EV>3% at open / 10am | CLV −1.3% / −3.4%, ROI −5.2% / −4.7% |
| Joint regression of move on model gap + Elo gap | b_model 0.377 (se 0.012), b_elo 0.009 (se 0.013) |

The CLV slope from the open is stable across all 13 scored folds (0.31–0.45; every fold more than 7 SE from 0).

"Mean CLV" is `p_close_fair × decimal_odds_at_entry − 1`: the expected return *if the close is efficient*. It
already pays the vig at the entry price. Its SE is roughly 10× smaller than the ROI's, which is why it is the
primary metric.

**Interpretation.** The model is a fundamentals forecaster that knows roughly what the market will know by tip.
The open sits farther from fundamentals than the close does, and the model predicts which way it moves. A dumb
fundamentals predictor (Elo) does *not* pay after vig, so this is not just "openers are stale". The edge decays
through the day: about +4% overnight, about +1.4% by mid-morning, and zero at the close. The money is in betting
**early**.

**Caveats.**

- **Unknown book.** TeamRankings' price source is not documented (bouncing +160/+170 values suggest an aggregate
  across books). Illinois retail books may not offer these numbers at these times, and early limits are low.
- **Fixed entry times.** "10am" is a fixed clock time; a real bettor's entry time will differ.
- **Older seasons.** Data ends 2025-03; markets may have sharpened since.
- **Spread result is now weaker.** Without the closing-line feature, the spread-market encompassing slope falls
  from 0.146 to 0.081 (se 0.064, n.s.). The earlier small "edge vs close" was partly the model echoing the line.

**How to verify forward.** Have the odds collector log timestamped per-book moneylines from the overnight open
through tip. Paper-bet EV>3% games at the first price actually available to you, and track CLV against the close.
At n≈200 bets the CLV SE is ~0.9%, which is NOT enough to tell +1.4% from 0 (that needs ~700 bets for 2 SE).

Reproduce: `python tools/encompass_check.py --exclude spread_signed,is_home_favorite,pregame_missing_top1,pregame_missing_top2,pregame_minutes_out,pregame_usage_out --out outputs/oos_noline.csv`
(~3 min). Add `--preds outputs/oos_noline.csv` to re-report from the cache. Note: the numbers above were produced
with the main checkout's working-tree `experiment.py` (uncommitted `settle_signed` changes), not this branch's
copy.

---

## Spike 2: Kalshi player points props (KXNBAPTS), 2025-26

**Data.** Kalshi's settled-market history is free and needs no API key: `https://api.elections.kalshi.com/trade-api/v2`,
endpoints `/historical/markets?series_ticker=` and `/historical/markets/{ticker}/candlesticks` (the historical cutoff
is 2026-08-09, so all of 2025-26 is served there). Market counts pulled:

| Series | Markets |
|---|---|
| KXNBAPTS | 23,562 |
| KXNBAREB | 22,656 |
| KXNBAAST | 17,745 |
| KXNBA3PT | 16,619 |
| KXNBA1HTOTAL | 4,707 |

Coverage runs 2025-11-18 to 2026-06-13. Candles were pulled for points only. Each market is a ladder rung,
"Player: N+ points".

- **Tip time.** `expected_expiration_time − 3h`. It equals `occurrence_datetime` in 3,429 of 3,463 markets that
  have both (85% lack `occurrence_datetime`), and implied tips cluster at 7:00/7:30/8:00pm ET.
- **Pre-game price.** The yes bid/ask at the close of the last hourly candle ending at or before tip.
- **Filters.** Two-sided book and spread ≤ 10¢: 17,373 of 22,043 markets with candles. Median spread 3¢ (1¢ in
  the playoffs).
- **Settlement.** Matched to a boxscore row for a player who played: 17,311 markets, 1,018 games, 173 players.
  Settlement agrees with boxscore points 100%. A player who is active but never plays settles at the last pre-game
  fair price (Kalshi rules), so those markets carry ~0 P&L and are excluded.

**Fees.** The series metadata says `fee_type: quadratic`, `fee_multiplier: 1`, i.e. the taker fee is
`ceil(0.07 · C · P · (1−P))`. Modeled as `0.07 · P · (1−P)` per contract, which is the large-order limit;
single-contract rounding is worse. Third-party sources say most sports markets have no maker fee. That could not be
confirmed from Kalshi's own schedule, and maker fills are not simulated.

**Model** (scratch, not kept):

- **Projection.** Projected minutes (EWMA, half-life 5 games) × points per minute (EWMA, half-life 15 games), plus
  `b × share × points of absent regulars`. A "regular" has EWMA minutes ≥ 20 and played one of the team's last 3
  games; absent means missing from this game's boxscore.
- **Distribution.** Negative binomial with variance `a · μ^p`.
- **Fitting.** `b` (0.33), a linear mean recalibration, and `(a, p) = (8.2, 0.70)` were fit on 2022-23..2024-25
  and frozen. All features use strictly earlier games.
- **Projection quality.** 2025-26 out of sample, all players: RMSE 5.98 points vs 6.05 without the teammate
  adjustment; bias +0.03.
- **Leak note.** "Absent regulars" comes from the game's own boxscore, which is approximately pre-tip injury news,
  but late scratches leak. The variant without that adjustment (`noadj`) is reported too; it does no better.

**Results** (n = 17,311 market rungs; SEs clustered by game):

| Metric | Kalshi mid | Model | Model (no teammate adj.) |
|---|---|---|---|
| Brier | **0.1694** | 0.1753 | 0.1764 |
| Log loss | **0.510** | 0.525 | 0.528 |
| Encompassing `(y − mid) ~ (p − mid)` | — | +0.008 (se 0.058) | −0.008 (se 0.055) |

Calibration by decile:

- **Kalshi** (predicted / hit rate): .05/.05, .10/.09, .16/.14, .24/.22, .33/.31, .43/.41, .52/.50, .62/.61,
  .72/.71, .84/.85.
- **Model:** .07/.07, .12/.09, .17/.15, .23/.23, .30/.32, .38/.42, .46/.49, .56/.58, .66/.70, .82/.83. It is
  slightly under-confident in the middle.
- **YES overpricing:** YES hits 1.2¢ less often than the mid implies (se 0.45¢), a small "overs" bias.

Flat taker bets, 1 contract per signal (YES at the ask, NO at 1 − bid, net of fee), return per $ risked:

| Signal | n | ROI | One bet per player-game: n | ROI |
|---|---|---|---|---|
| model edge > 0 | 11,437 | −11.8% (se 1.8%) | 4,627 | −10.9% (se 2.4%) |
| model edge > 3¢ | 6,689 | −8.2% (se 2.4%) | 3,302 | −7.4% (se 2.7%) |
| model edge > 6¢ | 3,740 | −7.6% (se 2.9%) | 1,954 | −7.1% (se 3.1%) |
| no model: buy NO on every rung | 17,311 | −4.4% (se 1.1%) | — | — |

Blanket NO is positive only in the 2026 playoffs (May +2.6%, June +4.0%), when spreads tightened to 1¢. That is
two months, so treat it as noise.

**Reading.** With one season, the encompassing slope's 95% interval is roughly [−0.11, +0.12]. That rules out the
baseline carrying substantial information beyond Kalshi's pre-tip price, though not a tiny amount. Even a perfect
small edge would have to clear a ~1.5¢ half-spread plus a ~1.5¢ fee near 50¢. One season, ~1,000 games and 173
players is thin, and the intervals above are wide. The direction is not ambiguous, though: every model-driven
strategy loses by more than 2 SE.

Reproduce:

- `python tools/kalshi_history.py KXNBAPTS --candles` (~1.5 h at 8 req/s; resumable).
- The scratch scripts `prop_model.py` and `prop_eval.py` are throwaway. Their logic is summarized above.

---

## Spike 3: points model vs EARLY Kalshi prop prices (KXNBAPTS, 2025-26)

Question: the moneyline edge lives early, so does the same points model beat Kalshi's prop prices 3-24h before tip?

**Setup.** Hourly candles for the 48h before tip (23,562 markets). For each rung, the entry quote is the last two-sided
candle ending at least *h* hours before tip, no older than 3h, spread ≤ 10¢; the "close" is the same pre-tip quote
used in Spike 2. Model: the frozen Spike 2 projection **without** the teammate-absence adjustment (`mu_noadj`),
because absences are only known game-day. Rungs are the 17,311 from Spike 2 that have an early quote. SEs are
clustered by game.

| Entry | rungs / games | median spread | Brier early mid / model | encompassing slope | taker ROI, edge>3¢ | blanket NO |
|---|---|---|---|---|---|---|
| 24h | 3,549 / 148 | 3¢ | 0.1625 / 0.1705 | −0.046 (se 0.117) | −13.3% (se 5.8%) | −3.1% (se 2.3%) |
| 12h | 9,490 / 692 | 4¢ | 0.1688 / 0.1736 | +0.096 (se 0.079) | −6.4% (se 3.6%) | −4.5% (se 1.4%) |
| 6h | 11,699 / 864 | 4¢ | 0.1711 / 0.1763 | +0.063 (se 0.067) | −9.3% (se 2.8%) | −3.5% (se 1.3%) |
| 3h | 13,305 / 923 | 3¢ | 0.1721 / 0.1777 | +0.046 (se 0.063) | −8.6% (se 2.7%) | −3.5% (se 1.2%) |

- **Prices do drift toward the model, but not enough to trade.** Regressing the move to the close on the model-vs-early
  gap gives slope +0.063 (se 0.016) at 24h, +0.071 (0.009) at 12h, +0.038 (0.007) at 6h. The sign is right, but the
  mean absolute move is only ~2¢ and the round trip costs ~1.5-2¢ of half-spread plus ~1.7¢ of fee near 50¢.
- **CLV net of cost is negative everywhere:** valuing the position at the close mid against entry cost including fee
  gives −5.7% to −9.5% per $ for every edge threshold and horizon.
- **Early quotes are not softer than late ones.** Early-mid Brier is within 0.002 of the close-mid Brier and the
  model is 0.005-0.008 worse at every horizon. Spreads are no tighter early (3-4¢).
- **Caveats.** The 24h sample is only 148 games (illiquid early); intervals are wide there. The model has no game-day
  information by construction. One season.

**Verdict: no-go on a prop model against Kalshi at any horizon.** A prop edge, if any, would need information the
market lacks (late scratches, minutes restrictions), not a better minutes × rate model.

Reproduce: `python tools/kalshi_history.py KXNBAPTS --candles --hours 48 --name early48 --skip-markets` (~1.7 h at the
rate limit; resumable; writes `KXNBAPTS_early48.csv`). The evaluation script `prop_eval_early.py` is throwaway;
its logic is summarized above.
