# Stage 1 pre-registration: market-anchored spread and total curves

Written 2026-10-23, before computing any residual statistic on these data. Methodology:
[methodology.md](methodology.md), stage 1. Only column names and row counts of the data file have been looked at.

## Question

The closing line is the center (anchor weight `w = 0`). Does a curve whose **width and shape** depend on the game
(closing total and spread size) price alt lines better than a **generic** curve with one fixed width? This is the shape
thesis. If the conditional curve barely differs from the generic one, alt-line shape is not worth pursuing.

## Data

`$NBA_DATA_DIR/odds/nba_2008-2025.csv`: closing spread, total and moneylines with final scores, 23,118 games.
`season` is the end year of the season (2025 = 2024-25). All games are used, including playoffs. Scores include
overtime, as sportsbooks settle. Three rows with no spread are dropped from the margin analysis.

- Home line `L_M` = +spread if the home team is favored, otherwise −spread (expected home margin). `L_T` = total.
- Residuals: `e_M = home margin − L_M`, `e_T = total points − L_T`.

## Models

All three are centered on the closing line. They differ only in width and shape.

| Model | Width | Shape |
|---|---|---|
| **A, generic** | One standard deviation fit on training seasons | Normal |
| **B, conditional normal** | `log sd = b0 + b1·(L_T − 210)/10 + b2·|L_M|/5`, fit by maximum likelihood on training | Normal |
| **C, conditional empirical** | B's sd | Empirical distribution of training residuals standardized by B's sd |

Probabilities for integer outcomes are taken as `P(X = x) = F(x + 0.5) − F(x − 0.5)`. For the margin, `P(M = 0)` is
set to 0 and the rest renormalized, since NBA games cannot tie.

## Walk-forward split

- Expanding window by season: each test season is scored by a model fit on all earlier seasons.
- **Development:** test seasons 2012–2022. Used to pick between B and C (lower mean RPS) separately for margin and
  total. The pick is then locked.
- **Holdout:** test seasons 2023–2025, run once after the pick is locked.

## Metrics

- **RPS** (ranked probability score) over integer outcomes, per game: the discrete form of CRPS. Lower is better.
- **Rung calibration:** for offsets `d ∈ {−15, …, +15}`, rung `X > L + d`. The predicted probability conditions on
  no push. Per `d`: mean predicted vs observed, z = (observed − predicted) / sqrt(Σ p(1−p)) × n.
- **Room versus generic:** `|p_pick − p_A|` on rungs `|d| ∈ {3, …, 10}`, summarized by median and 90th percentile.
- **Secondary, margin only:** P(home win) from the curve vs the de-vigged closing moneyline. Brier score of each on
  the holdout. Descriptive only.

## Decision rules (holdout, applied separately to margin and total)

1. **H1, shape beats generic:** mean RPS of the picked model is lower than A's, with the 90% game-bootstrap interval
   of the difference above 0.
2. **Calibration:** every rung offset has |z| ≤ 3 for the picked model.
3. **Room:** the 90th percentile of `|p_pick − p_A|` on rungs within ±3..10 is at least 2 percentage points. Alt-line
   vig is at least that large, so a smaller difference cannot be bet even if the books used model A.

Go to forward alt-line pricing for a quantity only if all three hold. If 1 and 2 hold but 3 fails, the curve is
still used as the pricing curve (it is the better calibrated one), but the shape thesis is parked for that quantity.

## Caveats stated in advance

- Model A is a stand-in for a book's generic conversion. Books' real alt prices are unknown until forward collection.
- The source of these closing lines is undocumented, so "closing" is assumed, not verified.
