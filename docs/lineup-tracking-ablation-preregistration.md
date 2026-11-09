# Pre-registration: lineup and tracking features vs the daily margin model

Written 2026-10-22, **before any ablation result was computed**. The script that produces results
(`tools/lineup_tracking_ablation.py`) records this file's SHA-256 and refuses to run if the file's
`Frozen` marker below is missing. Any change to a rule after results exist requires a new dated
pre-registration, and the old one stays.

Frozen: yes

## Question
Do pre-game lineup features (`engineer_lineup_features`) and player-tracking features
(`engineer_tracking_features`) improve out-of-sample home-margin prediction over the current daily
model? This is a model-quality test, not a betting-edge claim.

## What is compared (all four use the identical games, folds, models and seeds)
Baseline = `predict_today.walk_forward` with `experiment.build_features(...)` minus `predict_today.EXCLUDE`
(the line-free daily model: TRAIN_SIZE 2,500 / TEST_SIZE 300, lgb_v1 + lgb_v2 + elasticnet, inverse-RMSE weights).

| Variant | Added columns (nothing else changes) |
|---|---|
| +lineup | `lu_top_unit_net_diff`, `lu_top_unit_share_diff`, `lu_starter_unit_net_diff`, `lu_starter_unit_share_diff`, `lu_unit_concentration_diff` (home minus away), `lu_games_min` (min of home/away `lu_games`) |
| +tracking | `trk_dist_per_min_diff`, `trk_speed_diff`, `trk_touches_per_min_diff`, `trk_passes_per_min_diff`, `trk_uncontested_fga_share_diff`, `trk_rim_fga_per_min_diff`, `trk_rim_fg_allowed_diff`, `trk_reb_chances_per_min_diff`, `trk_games_min` |
| +both | the union of the two |

Feature parameters were fixed before this test and are not tuned: window 10 same-season games, shrink 300
possessions (lineups), top 8 rotation / shrink 50 rim attempts / rim prior 0.65 (tracking). Missing values stay
missing (no filling beyond what the existing models already do).

## Blocks (season = ending year, as in the data: 2025 means 2024-25)
- **Development (diagnostic only, nothing is tuned on it):** out-of-sample games from seasons 2022, 2023, 2024.
- **Holdout (the decision block):** out-of-sample games of season **2025** (2024-25), 1,230 games.
- **Second check (descriptive):** season 2026 (2025-26), **+lineup only**; tracking columns are empty for that season
  so +tracking and +both are not defined there.
The holdout was not used to choose any feature, window, shrink value or threshold. Disclosure: before registering,
the author computed simple correlations of each home-minus-away feature with the actual home margin over all seasons
(including the holdout) as a smell check; no parameter or feature set was changed in response.

## Primary metric
**Out-of-sample RMSE of the ensemble home-margin prediction on the holdout block**, compared with the baseline on the same
games. The primary candidate is **+both**. +lineup and +tracking are secondary and are reported regardless of the
outcome of +both.

## Decision rules (a variant is "supported for further work" only if ALL hold; otherwise "not supported, keep baseline")
1. **Holdout effect:** RMSE(variant) is at least **0.05 points lower** than RMSE(baseline), and the one-sided 95% lower
   bound of the paired improvement (below) is **greater than 0**.
2. **No development regression:** on the development block, RMSE(variant) is not more than **0.02 points higher** than
   baseline.
3. **Calibration:** on the holdout, Brier score of the home-win probability (baseline and variant each calibrated with
   `encompass_check.fit_probit_scale` on their own out-of-sample margins) is not worse than baseline by more than
   **0.001**.
4. **Opening months:** RMSE over October-November games, development + holdout pooled, is not more than **0.05 higher**
   than baseline.

## Uncertainty
Paired difference in squared error per game (baseline minus variant), summarized as the RMSE difference; block bootstrap
over **game dates** (resample dates with replacement, keep all games of a date), 2,000 draws, seed 20261022. Report the
two-sided 90% interval (its lower end is the one-sided 95% bound).

## Always reported (not used for the decision)
MAE; RMSE by season; coverage (share of OOS games where each added feature is non-missing); feature importance is
not used to select anything. All four variants are reported even if only one is the candidate. No betting metrics.

## Not allowed after seeing results
Changing windows, shrinkage, feature lists, folds, models, seeds or thresholds and re-running the same test. A new idea
is a new pre-registration, tested on data not used here.
