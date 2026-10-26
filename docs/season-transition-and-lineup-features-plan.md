# NBA Prediction — Season transitions and player availability

Status: draft; local proof of concept run 2026-10-22
Date: 2026-10-22
Scope: `repos/nba_prediction` feature pipeline and daily margin predictions

## Goal

Make the 09:30–10:00 America/Chicago daily prediction reflect what is known about
each team's current rotation, while handling old team and player observations
explicitly at a new season. Measure whether the changes improve held-out margin
predictions before using them in daily picks.

## Current behavior and motivating case

- `tools/predict_today.py` refits the line-free margin model on the latest 2,500
  completed games. Its selected features contain no head-to-head matchup statistic.
  A last-meeting display is context, not a separately weighted model input.
- Team Elo retains 75% of its prior-season deviation. Several 5-, 10-, and
  20-game team windows continue across seasons. Season-to-date features are
  built, but the daily model's static allowlist selects none of them.
- The selected explicitly player-based inputs are `top8_points_diff` and
  `prev_game_minutes_out`. The former uses player form attached to the previous
  box-score-listed team snapshot. The latter carries the previous game's
  absence minutes forward without a season or age limit. Same-day official
  inactive features are excluded from the morning prediction.
- In the local 2026-10-22 PHI–CLE pregame reconstruction, Cleveland's previous
  regular-season game was 2026-04-12. The resulting `prev_game_minutes_out`
  differential was -44.6 (PHI minus CLE), and `top8_points_diff` was 12.05.
  These are input values, not measured effects on a prediction. The latest
  regular-season PHI–CLE meeting in the local table was 2026-03-09; 19 PHI
  and 17 CLE regular-season games followed it through 2026-10-20.

## Constraints and non-goals

- Every historical row must use only information available by the same local
  morning cutoff as the daily run. Do not use the target game's participants,
  minutes, DNP comments, final injury status, or a roster snapshot captured
  after that cutoff. An undated roster is not historical evidence.
- Keep missing availability distinct from a confirmed healthy/unchanged
  rotation. Preserve player IDs across trades, but attribute team-level
  rotation only to the player's team as known at the cutoff.
- Keep the existing pipeline contracts (`run(config, inputs)`), raw-data and
  cache boundaries, and the `# FROZEN` section of `experiment.py`. New feature
  meanings get new column names; old artifacts cannot silently supply them.
- The first implementation uses existing local history. A live injury,
  transaction, or projected-lineup provider is a separate source decision:
  verify coverage, timestamps, terms, rate limits, and cost before integration.
  Do not introduce a paid feed or live write as part of this plan.
- This is a model-quality and data-quality project, not a claim of betting edge.
  No inference-only patch to the current pick: train and serve each candidate
  feature with the same definition.

## Approach

### 1. Freeze the baseline and audit the season boundary

1. Record the code revision, input artifact hashes/date range, selected daily
   features, training folds, and baseline out-of-sample predictions. Keep
   generated datasets, models, and results out of Git.
2. Produce a per-game diagnostic for each team's latest observation date,
   previous observation season, available player-history count, and source of
   any roster/absence claim. Inspect opening weeks, long layoffs, traded
   players, and games after a last-day rest rotation.
3. Confirm where team windows cross seasons and which calculated season-to-date
   columns actually enter `experiment.build_features`. Document the current
   PHI–CLE case as a reproducible fixture, without using its result.

**Gate:** The baseline and diagnostic can be recreated from a recorded input
manifest. The diagnostic distinguishes missing, stale, and known-current data.

### 2. Fix stale previous-game signals with existing data

1. In `engineer_advanced_features.py`, add a newly named prior-absence feature
   whose value is unknown when the previous team game belongs to another
   season. Add previous-game age and season-boundary indicators. Preserve the
   existing `prev_game_minutes_out` for comparison; exclude it from a candidate
   model only when the replacement is evaluated.
2. Replace the single previous-game roster assumption behind
   `top8_points_diff` with a candidate, separately named rotation summary
   derived strictly from earlier games: aggregate player minutes/form over
   multiple recent team games, retain each player's last-seen date, and expose
   coverage/staleness. Do not label this an expected lineup when current roster
   membership is unknown. Compare it with a team-only fallback on an opening
   day such as Cleveland's 2026-10-22 game.
3. Evaluate team carryover independently: current Elo shrink and cross-season
   rolling windows versus season-aware alternatives that retain an explicit
   prior-season prior and a current-season sample-size indicator. Check the
   misleading `blowout_pct_season` name/definition: it currently expands by
   team across all seasons. Do not change several feature families in one
   experiment.

**Gate:** Synthetic tests show no target-game roster or score can affect its
own feature row. The PHI–CLE fixture no longer treats April's absence list as
a known October absence. Point-in-time reconstruction agrees with the full
build on sampled dates across season boundaries.

### 3. Decide whether a current-lineup source is viable

1. Specify the minimum snapshot: game/date, team, stable player ID, status or
   projected availability, expected minutes if supplied, source, retrieval
   timestamp, and the exact cutoff. Store a missing/uncertain state and allow
   provider corrections without rewriting the earlier snapshot.
2. Compare official or licensed sources for historical as-of coverage and
   daily reliability. Check terms, pricing, limits, identifiers, and whether
   a complete snapshot can be captured by the morning run. Prior-game
   box scores alone cannot prove today's availability.
3. If historical as-of snapshots are unavailable, begin forward collection
   without incorporating the feed into a historical backtest. A new
   snapshot-based model waits until there is enough genuinely out-of-sample
   history. The history-only candidate from step 2 can still be evaluated.

**Gate:** A written source decision states coverage, missingness, cost, and
earliest trustworthy evaluation date. No provider is selected by this plan.

### 4. Evaluate history-only changes and, if viable, a lineup-aware candidate

1. Evaluate the history-only candidates from step 2 whether or not a current
   lineup source passes step 3. Test the stale-absence replacement, rotation
   summary, and team carryover separately against the same baseline. A
   history-only improvement may be promoted on its own; label its roster
   estimate as uncertain at a new season.
2. If step 3 passes and enough as-of history exists, build a separate
   point-in-time projected rotation from players known at the cutoff and
   lagged minutes/form. Start with transparent team summaries (projected
   minutes coverage, retained prior-season minutes, unavailable prior
   contribution, and uncertainty). Defer five-player interaction or on/off
   features until lineup identity and minutes are reliable and sufficiently
   sampled.
3. Add candidate columns through the step-4/step-6 feature pipeline and
   `step6a` allowlist. Select them only through the editable
   `EXTRA_FEATURE_INCLUSIONS` / `EXTRA_FEATURE_EXCLUSIONS` config;
   `STATIC_FEATURES` is below the frozen boundary. Make
   `tools/run_features.py --slate` and `tools/predict_today.py` use the
   identical point-in-time definitions.
4. Compare one change at a time with the frozen baseline on identical
   chronological folds. Report overall and opening-month RMSE/MAE, home-win
   calibration (Brier or log loss and reliability), coverage/missingness,
   and error by long gap and roster turnover. If evaluating betting results,
   also separate matched from fallback odds, ROI uncertainty, drawdown,
   bet count, and daily exposure. Use a final untouched season or date block
   for the promotion decision; do not select a model on that block.

**Gate:** A history-only or lineup-aware candidate improves a prespecified
primary metric on the untouched block without a material regression in
calibration or high-uncertainty subgroups. Otherwise retain the current model
and publish the diagnostic only.

### 5. Release and monitor

1. Rebuild derived features and model artifacts from recorded raw inputs.
   Tie cache manifests to source revision, feature-definition version, and
   input content; reject a cache with mismatched provenance. Extend
   `tools/predict_today.py:model_version()` to include the feature-definition
   version: its current hash covers feature names and model configuration but
   misses same-name semantic changes. Run `python -m pytest tests -q`, `python
   tools/leak_screen.py`, the point-in-time test with `RUN_PIT=1`, and a local
   daily-prediction smoke test using a fixed past slate. No API-heavy ingest or
   full training run is part of plan authoring.
2. Compare new and old predictions in shadow mode, including the PHI–CLE
   opening-week scenario and missing-roster cases. Record freshness and
   coverage alongside each generated pick. Keep the existing model as the
   rollback path.
3. Seek approval before changing the scheduled GitHub job, its state bundle,
   any external feed, or the bet-watch prediction contract. A bet-watch
   last-meeting display should remain clearly labeled as historical context;
   changes to that separate project need their own scope.

**Gate:** Local checks and shadow comparisons pass, source coverage is
acceptable, and the daily output contract is reviewed before promotion.

## Verification

- **Automated:** unit fixtures for season rollover, last-day rest, missing
  roster, trades, DNP versus true absence, and multiple games on one date;
  point-in-time truncation/equality; `tests/test_predict_today.py`; full
  repository tests and leakage screen after feature changes.
- **Manual:** inspect feature provenance and values for 2026-10-22 PHI–CLE,
  one ordinary midseason game, and one high-turnover opening game. Check
  JSON `data_through`, feature schema version, and no unintended API calls.
- **Research:** preserve fold-level and subgroup metrics, model/config/input
  fingerprints, uncertainty intervals, and a holdout untouched during design.

## Risks / decisions needed

- A morning availability feed may have no usable historical as-of archive.
  In that case, do not backfill with later knowledge; collect forward and
  continue the history-only experiment separately.
- Early-season sample size and offseason transactions may make player
  summaries less reliable than team priors. Missingness and uncertainty must
  remain visible to the model and to diagnostics.
- The choice of primary promotion metric, holdout dates, minimum source
  coverage, and any external provider requires a decision before step 4's
  final evaluation. Record these before looking at final holdout results.

## Local proof of concept, 2026-10-22

`tools/season_rotation_poc.py` is an opt-in comparison on an isolated local
dataset refreshed through 2026-10-21. It leaves the scheduled daily-pick path
untouched. The candidate replaces the old previous-game absence differential
with an unknown value when either team's last game belongs to another season,
and replaces `top8_points_diff` with a ten-game minutes-ranked player-points
history. It adds observation-age and same-season-sample features. Missing
box scores are excluded from the player summary and window coverage is reported.
The conservative absence mask discards a valid current-season team observation
when its opponent's previous game is stale; step 2 should move to separate
home and away components before promotion.

On the exploratory walk-forward comparison (5,623 games), margin RMSE was
14.195 for the baseline, 14.195 for the season guard, and 14.204 for the
combined rotation candidate. October/November RMSE was 13.406, 13.428, and
13.472, respectively. These are model-development comparisons, not an
untouched holdout. The combined change has no demonstrated accuracy gain.

For the 2026-10-22 slate, baseline / season guard / combined candidate home
margin predictions were PHI over CLE +6.29 / +2.89 / -0.34 and OKC over DEN
+6.12 / +5.05 / +2.13. The combined prediction's Cleveland edge is only
0.34 points, effectively a toss-up. Both candidate games depend on player
history last observed as much as 193 days earlier. The player proxy does not
know tonight's confirmed starters, transactions, or injury availability.
Before generalizing, evaluate separate-team absence components, current-roster
coverage, team-only fallback, and a prespecified untouched block.
