# Betting methodology (2026-10-23)

This document is the operating methodology for the NBA betting work. It replaces the earlier framing, which treated
a gap between the margin model and the closing line as a pick. Every new market, model or experiment should say which
part of this document it serves.

## Why the reset

| Finding | Source |
|---|---|
| The box-score margin model adds nothing beyond the moneyline close: encompassing slope +0.008 (se 0.090). Against the closing spread, without closing-line features, the slope is 0.081 (se 0.064, n.s.). | [edge-research-findings-2026-10.md](edge-research-findings-2026-10.md), Spike 1 |
| The same model, without closing-line features, shows real CLV against **early** moneylines. EV>3% bets average +4.3% CLV (se 0.36%) at the overnight open and +1.4% (se 0.26%) at game-day ~10am ET. An Elo-only placebo loses. | same, Spike 1 |
| A minutes × rate points model loses to Kalshi's prop ladders. Brier is 0.1694 for the Kalshi mid vs 0.1753 for the model, and taker ROI is −7% to −12% after fees. Early Kalshi prices show the same result. | same, Spikes 2–3 |
| Lineup and tracking features improve RMSE a little, but the primary pre-registered hypothesis was not supported. The lineup increment's edge on the closing spread is about 0.5 points, roughly 1.5 pp of win probability, against about 2.4 pp of vig at −110. | [lineup-tracking-ablation-results.md](lineup-tracking-ablation-results.md), [lineup-market-encompassing-results.md](lineup-market-encompassing-results.md) |

The conclusion: out-predicting the closing line on the main market is not where the money is. Timing is, and
possibly the **shape** of the outcome distribution, which books need to price alt lines and ladders.

## The methodology

> Price every market from one joint distribution of the game. Anchor that distribution's center to the sharpest
> available market price. Bet only where a book's price disagrees with the distribution for a structural reason,
> and only at an entry time you can actually use. Judge every bet by CLV in a forward ledger, under a pre-registration
> written for that market.

There are two edge theses, in priority order:

1. **Timing.** Early prices are farther from fundamentals than the close. Bet early, at the overnight open or before
   about 10am CT; this is confirmed as feasible. Every pre-registration names an early entry time.
2. **Shape.** Books derive alt spreads, alt totals, halves and prop ladders from the main line using generic
   conversions. A better model of the distribution's width, skew and tails can find rungs priced off that shape,
   even when the center is correct.

## Influences

| Analyst | Core idea | What this methodology takes |
|---|---|---|
| Bob Voulgaris | Found mispricings in derived markets: books split full-game totals into halves with naive rules, while late-game fouling made second halves score differently. His later simulator was built mainly on minutes and coaching rotations. | Hunt in derived markets (alt lines, ladders, halves). Minutes projection is the most important input for props. |
| Evan Miya (college) | Bayesian Performance Rating: box-score prior plus lineup on/off → player ratings → team strength from who plays. | Player ratings from the existing lineup stints, rolled up to team strength by projected minutes. An injury is a minutes change, not a dummy variable. |
| Bart Torvik (college) | Opponent-adjusted efficiency and tempo with recency weighting, published daily and explainable. | Keep the team layer simple: possessions × points per possession, opponent-adjusted, with explicit recency decay. |
| Warren Sharp (NFL) | Situational and matchup context, early lines, respect for the market, attention to why lines move. | Bet early and measure by line movement (CLV). Log situational context (rest, travel, back-to-backs) without letting it drive picks before it is tested. |
| Lev Akabas | Clear, visual, quantitative explanation of sports and markets. No betting-specific technique is borrowed. | Reporting standard: every claim ships with a chart and an uncertainty band. |

## The engine: one simulation, many markets

```
player ratings (box-score prior + stint on/off)
        │
minutes projection (rotation, injuries, rest, blowout risk)     ← most important input
        │
team possessions × efficiency (opponent-adjusted, recency-weighted)
        │
Monte Carlo game sim → joint draws of (home pts, away pts, per-player min/pts/reb/ast/3pm)
        │
market anchoring: move the sim's center toward the sharp no-vig line; keep the sim's shape
        │
price every offered line → fair probability → EV after vig and Illinois per-bet fees
        │
bet rule (EV threshold, entry time) → ledger → CLV + calibration
```

Every market is a function of the same game. Separate models can contradict each other, for example a team favored
by 8 while its star is projected for few minutes. One set of joint draws keeps every price consistent and carries the
correlations: blowout → starters sit → player unders; minutes ↔ double-double.

The engine is a target, not a first step. Stage 1 below uses an empirical distribution around the closing line and
needs no simulator.

### Anchoring rule

For each market quantity (margin, total, player stat mean):

```
center = market_fair + w × (model − market_fair)
```

- `market_fair` is the sharp reference's no-vig line: Pinnacle-style consensus where it is collected, otherwise the
  median across books.
- `w` is the slope from that market's encompassing regression, `(outcome − market) ~ (model − market)`, estimated
  walk-forward.
- **Default `w = 0`.** Trust the market's center until a test says otherwise. The model then contributes only the
  shape (width, skew, tails, correlations), which is the shape thesis in its pure form.

## Fair-price formulas

Notation: `M` = home margin, `T` = game total, `X` = a player's stat, `F` = the anchored CDF of the relevant quantity
(empirical or simulated).

| Market | Fair probability |
|---|---|
| Spread, home −s (half point) | P(M > s) = 1 − F_M(s) |
| Spread, integer s | win P(M > s), push P(M = s); price on P(win) / (1 − P(push)) |
| Moneyline, home | P(M > 0). NBA games cannot tie, so no push |
| Total over t (half point) | P(T > t) = 1 − F_T(t) |
| Prop ladder "N+" (integer stat) | P(X ≥ N) = 1 − F_X(N − 1). With a continuous approximation, use the continuity correction 1 − F(N − 0.5) |
| Prop over/under at line L.5 | P(X > L.5) = P(X ≥ L + 1) |
| Double-double | Share of sim draws where at least two of {pts, reb, ast, stl, blk} are ≥ 10 |

EV per $1 staked at decimal odds `d` is `p × d − 1 − fee / stake`, using the Illinois fee for that book and stake.

## Per-market plan

| Market | Edge hypothesis | History available now | Must collect forward |
|---|---|---|---|
| **Spread curve** (all alt spreads) | Margin width depends on total, pace and favorite size, while books use near-generic alt ladders | Closing spread/total and final scores, 2008–2025 (`data/odds/nba_2008-2025.csv`) | Timestamped per-book alt spread prices |
| **Moneyline** | Early prices are stale (measured). Spread↔ML conversion errors | TeamRankings ML open/intraday/close (`data/csv/game_odds.csv`) | Per-book ML from open to tip |
| **Total curve** (all alt totals) | Total variance scales with pace and total; halves and quarters are priced naively | Closing totals and quarter scores 2008–2025; 2H lines through mid-2023 | Alt total prices |
| **Player props and alt ladders** | Low rungs on role players depend on minutes risk (blowout, fouls, DNP). Faster injury news | Kalshi ladders for pts/reb/ast/3pm, 2025-26 (`data/kalshi/`); box scores | Book prop ladders (SportsGameOdds free tier) |
| **Double-double** | None. This is a **forecast, not a bet market** (prices are unlikely to be available) | Box-score outcomes (`dd2` column) | Nothing; outcomes come from box scores |

### Double-double forecasts

The model outputs P(double-double) per player per game and logs it before tip. It is judged only on calibration
against actual `dd2` outcomes: Brier score, log loss and a reliability curve, with no CLV, EV or ROI. The output is
kept for later use. If prices become available, it can become a bet market under its own pre-registration.

### Honest priors

- Moneyline timing: positive evidence.
- Spread and total curves: untested.
- Props: negative evidence at Kalshi. Any new props test must target a different hypothesis (minutes-risk rungs,
  injury timing), not a better center.
- Double-doubles: calibration only.

## Evaluation rules

These rules apply to every market.

1. **Ledger first.** Write one append-only row per priced line before tip. A row written after tip is invalid.
2. **Primary metric: CLV** against the sharp close, at the entry time you actually use. Its standard error is about
   10× smaller than ROI's.
3. **Calibration.** Reliability curves for every binary rung; PIT histograms or CRPS for full distributions. Check
   the tails, not just the rungs near the main line; a curve is only as good as its tails.
4. **Encompassing.** Fit `(outcome − market) ~ (model − market)` per market. A slope near 0 means the model's center
   adds nothing; keep `w = 0`.
5. **ROI last,** net of per-bet fees, and only at books you can use in Illinois. Fees as of 2026-10: FanDuel $0.50
   (waived ≥ $25), DraftKings $0.50 (waived on singles ≥ $50), Fanatics $0.25. Kalshi is supplementary while its
   Illinois status is in litigation.
6. **Pre-register** before looking, using the template below. No re-tuning on the same data.
7. **Real-money gate:** CLV ≥ +1% over at least about 200 bets in that market.

### Ledger schema

One row per (game, market, line, book, snapshot):

| Field | Meaning |
|---|---|
| `logged_at_utc` | When the row was written; must be before `tip_utc` |
| `game_id`, `tip_utc` | Game identity and scheduled tip |
| `market` | `spread`, `ml`, `total`, `prop`, `dd` |
| `player_id`, `stat` | Props and double-doubles only |
| `side`, `line` | e.g. `home`, −6.5; `over`, 224.5; `over`, 9.5 |
| `book`, `price_decimal`, `price_seen_at_utc` | The offer we priced |
| `fair_prob`, `model_version` | Our probability and the code/config that produced it |
| `anchor_w`, `market_fair_line` | Anchoring inputs |
| `ev`, `fee`, `would_bet`, `stake` | Decision |
| `close_price_book`, `close_price_sharp` | Filled after tip |
| `result` | win / loss / push, or the actual stat; filled after the game |

Double-double rows leave the price, EV and bet fields empty.

### Pre-registration template

```
Market:            (e.g. alt spreads, all rungs within ±10 of the main line)
Hypothesis:        (one sentence: structural reason the book is wrong)
Entry time:        (e.g. first price available after 06:00 CT on game day)
Books:             (Illinois books in scope)
Model + version:   (frozen config, anchor_w)
Bet rule:          (EV threshold after fees, max bets per game, stake rule)
Primary metric:    mean CLV vs sharp close, with SE
Secondary:         calibration by rung bucket; ROI net of fees
Sample size:       (n bets, or date range)
Stop rule:         (e.g. stop if CLV < 0 after 100 bets; go if ≥ +1% at ≥ 200)
Written:           (date, before any data in the window is seen)
```

## Stages

| Stage | Work | Why this order |
|---|---|---|
| 0 | Forward ledger; confirm the odds collector captures alt lines and props | No free history exists for alt or prop prices, so nothing can be proven without it |
| 1 | **Market-anchored spread and total curves** from the empirical distribution of (result − closing line), conditioned on spread size and total. Calibrate on 2008–2025, then price forward alt lines against it. **Done 2026-10-23:** curves are calibrated, but the shape thesis is not supported for full-game lines ([results](curves-stage1-results.md)). Curve C is the pricing curve. | Uses data already on disk; tests the shape thesis cheaply; the moneyline falls out at k = 0 |
| 2 | Early-line moneyline forward test using the stage 1 curve. **Retrospective gate failed 2026-10-23:** on 2025-26 Kalshi prices, CLV at 06:00 CT was −1.5% and the CLV slope ≈ 0 ([results](stage2-retro-results.md)). Forward ledger not built. | Strongest existing lead |
| 3 | Minutes projection and player stat distributions → prop ladders. Report low rungs and minutes-risk cases separately | Minutes are the bottleneck for props; reuses stints and box scores |
| 4 | Joint simulation → P(double-double), logged before tip and scored on calibration only | Needs stage 3's distributions plus their correlation |

## Parked

- Lineup and tracking RMSE work. It improves accuracy on the main line, which the market already prices. Re-open it
  only with a new dated pre-registration on unseen games, without re-tuning the window of 10 or the shrinkage of 300.
  See [lineup-tracking-ablation-results.md](lineup-tracking-ablation-results.md).
- Reading model-vs-close gaps as picks.

## Existing tools to reuse

- `tools/encompass_check.py`: encompassing regression and walk-forward predictions without closing-line features.
- `tools/kalshi_history.py`: free Kalshi ladder history.
- `src/odds_collector/`: forward per-book prices ([odds-collector.md](odds-collector.md)).
- `data/processed/lineup_stints.csv`: 5-man stints for player on/off ratings.
- `tools/snapshot_injury_report.py`: daily injury-report snapshots for injury timing.
