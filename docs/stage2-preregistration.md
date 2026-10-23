# Stage 2 pre-registration: early-line moneyline CLV, 2025-26 retrospective (Kalshi)

Written 2026-10-23, before any Kalshi `KXNBAGAME` price or candle was pulled. Plan:
[stage2-early-ml-plan.md](stage2-early-ml-plan.md), part A. The only Kalshi data seen so far is four live 2026-27
market headers and three 2025-26 tickers, all without prices.

## Question

Does the line-free margin model predict where Kalshi's game-winner price moves between an early entry and tip? This
replicates the 10/08 spike (+4.3% CLV at the open, +1.4% at 10am, TeamRankings, data through 2025-03) on a later
season and a different venue, using the current point-in-time feature pipeline.

## Sample

- 2025-26 regular-season games (`season` 2026 in the games table) with a Kalshi `KXNBAGAME` market for each team.
- Game dates are matched on the ticker date plus the team pair.
- Playoffs are excluded; the model is trained on the regular season only.

## Model (frozen)

- **Margin:** `predict_today.walk_forward` over all completed regular-season games. The features are
  `experiment.build_features` minus `predict_today.EXCLUDE`: no closing line, favourite flag, spread prices or
  same-day absences. The target is the raw `home_margin`, and folds are `experiment.TRAIN_SIZE` / `TEST_SIZE`. Each
  2025-26 game gets the out-of-sample prediction of the fold that contains it.
- **Win probability, curve C (primary):**
  - `p_home = P(M > 0)` from `curve_calibration.pmf(pred, s, G, no_tie=True)`.
  - `G` is the stage 1 standardized empirical CDF: margin residuals around the close, seasons 2008–2025, divided by
    model B's sd.
  - `s` is fit by maximum likelihood on home-win outcomes of out-of-sample predictions in folds **before** the
    game's fold, requiring at least 250 games. This is the same as the spike's probit scale, with `G` in place of Φ.
- **Win probability, Φ (reference):** `Φ(pred / s_Φ)` with `s_Φ` from `encompass_check.fit_probit_scale` on the same
  earlier folds.
- **Elo placebo:** logistic on `elo_diff`, fit on the same earlier folds (`encompass_check.fit_logistic`).

## Prices

Hourly Kalshi candles for each team's market (`yes` = that team wins). Prices are in cents.

- **Tip:** the scheduled tip time.
  - Source: ESPN's schedule if its cached data covers the game.
  - Otherwise Kalshi's `expected_expiration_time − 3h`, which is checked against ESPN on the games where both exist.
  - If the two disagree by more than 15 minutes on over 5% of games, Kalshi times are not used.
- **Entries:** E06 and E10 are the candles ending at 06:00 and 10:00 Central on game day.
  - Each uses that candle's closing `yes_bid` and `yes_ask`.
  - An entry is used only if it ends at least 30 minutes before tip.
- **Close:** the last candle ending at or before tip, valued at its mid `(bid + ask) / 2`.
- **Quote filter (entry and close):** `0 < bid < ask < 100` and `ask − bid ≤ 5¢`.
  - A game is priced at an entry only if both teams' markets pass the filter at that entry and at the close.

## Bet rule (frozen)

- For each team T at entry: `EV_T = p_T × 100 / ask_T − 1`.
- Bet the side with the larger EV if `EV ≥ 0.03`, at most one side per game.
- No fee in the rule.

## Metrics

**Primary:** mean gross CLV of E06 bets, where `CLV = mid_close_T / ask_T − 1`.
- SE is clustered by game date; 90% CI = mean ± 1.645 SE.
- **Pass:** the CI's lower bound is above 0.
- **Inconclusive:** fewer than 150 E06 bets. Neither pass nor fail is declared, and the bet count is reported before
  the CLV is computed.
- **Fail:** otherwise.

**Secondary (reported, no decision attached):**
- The same CLV for E10 bets.
- Net CLV after Kalshi's taker fee: `fee = 7 × P × (1 − P)` cents per contract at entry, with P = ask/100, and
  `CLV_net = mid_close / (ask + fee) − 1`.
- CLV slope over all priced games: `logit(mid_close) − logit(mid_entry) ~ logit(p_model) − logit(mid_entry)`, home
  side, with date-clustered SE, at each entry.
- The Elo placebo and the Φ reference under the same bet rule: mean CLV and n.
- Calibration of curve C vs Φ on outcomes: Brier score and log loss over all 2025-26 games.
- Realized ROI of the bets (noisy, shown for completeness).

## Not allowed

- No re-tuning of the 3% threshold, the entry times, the spread filter or the model after seeing CLV.
- One scoring run. A crash or a data bug may be fixed and re-run; any such fix is listed in the results doc with its
  reason.

## Decision

- **Pass:** build part B (the forward ledger).
- **Fail:** stop stage 2 and record the result.
- **Inconclusive:** stop. The forward ledger can only start after you decide on it explicitly.
