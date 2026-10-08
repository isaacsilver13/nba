# Feature leakage audit and fix plan

Audit date: 2026-10-07. Scope: the 295 model features (`experiment.build_features`) plus every column in
`data/processed/df_model_3.csv`, traced through `src/ingest/dataPrep/steps/`.

Rule used: a feature is **clean** only if it could have been computed at bet time. That means its value for game
*g* depends only on games strictly before *g*, on facts known before tip-off (schedule, venue, closing line,
official inactive list), and on nothing that depends on how game *g* was played.

Three severities:
- **L1 (outcome leak)**: depends on how game *g* was played. Inflates backtests. Must fix before any result is trusted.
- **L2 (look-ahead)**: uses future games or whole-season statistics. Weaker than L1, still invalid.
- **D (defect)**: not a leak, but wrong, degenerate or duplicated. Fix while in the code.

## 1. Findings

### L1: outcome leaks

| # | Feature(s) | Where | Why it leaks | Evidence |
|---|---|---|---|---|
| 1 | `minutes_lost_last_game`, `usage_lost_last_game`, `missing_top1_flag`, `missing_top2_flag` (diff of home/away) | `engineer_advanced_features.py` ~L370-395 | Built from `did_play = minutes_played > 0` of **this** game, grouped by `GAME_ID`. Despite the "last_game" names, 88% of zero-minute rows are "DNP - Coach's Decision", which depends on game flow. | Lost minutes from coach DNPs correlate -0.42 with abs(margin). Out-of-fold rho vs market error: coach DNPs 0.345, pre-game inactives (DND/NWT) -0.006. Data screen: d_R2 on abs(margin) 0.138 / 0.112 / 0.017 / 0.007 against a noise cutoff of 0.0045. Now in `DENY_EXACT`. |
| 2 | `top8_points_diff` | `engineer_advanced_features.py` ~L536-548 | Top-8 players are picked by **this game's** minutes among players who played, so the set depends on the game's rotation and garbage time. The per-player points are properly lagged (r5). | Code read. Data screen shows no strong signal, so it is a mild leak, but it is the same mechanism as #1. |

### L2: look-ahead

| # | Feature(s) | Where | Problem |
|---|---|---|---|
| 3 | `team_style_vs_opponent_style` | `engineer_rolling_features.py` ~L249-266 | K-means on **full-season** team averages (includes the current and all future games of the season), then merged back by season. |
| 4 | `travel_1d_z`, `travel_3d_z`, `travel_7d_z`, `travel_10d_z` | `calculate_travel_fatigue.py` ~L188-193 | z-score uses the mean and std of the whole (home team, season) group, including future games. |
| 5 | `is_key_player` (feeds `key_players_out`, `injury_*`; not in the final table today) | `engineer_advanced_features.py` ~L400-425 | Player strength is the mean over the player's entire history, future included. |
| 6 | Sample selection | `calculate_elo_and_rest.py` L477 | `df = df[home_margin.abs() != spread]` drops push games (207 of 22,780, 0.9%) based on the final score. Those games vanish from every team's rolling history and from Elo. |

### D: defects found on the way

| # | Feature | Problem |
|---|---|---|
| 7 | `ats_roll_last20` | Cover is computed as `margin > team_spread` (`calculate_elo_and_rest.py` L283), but `team_spread` is the signed handicap, so the correct test is `margin + team_spread > 0`. The sign is wrong: code and truth agree on only 73% of games. The feature is lagged, so it is not a leak, but it does not measure ATS form. |
| 8 | `net_rating_trend` | Uses `net_rating_r15`, which is never built (windows are 5/10/20). `.get(..., 0)` silently turns it into `net_rating_r5` diff. It equals `net_rating_diff` in 100% of rows, and `net_rating_last_5` is the same again. |
| 9 | `after_ot_flag` | Constant 0: `ot_home`/`ot_away` are not in the source data, so they default to 0. |
| 10 | `home_margin_last5`, `away_margin_last5`, `margin_diff_last5`, `pts_diff_last5` | `.rolling().mean().shift(1)` after `groupby` shifts across team boundaries (first game of a team inherits the previous team's last window). Also the windows cover only the team's home (or away) games, and the away "margin" is the home-perspective margin. |
| 11 | `player_3p_pct_r{w}`, `player_2p_pct_r{w}`, `player_ft_pct_r{w}` | Not rolled at all: they are this game's raw percentages. Not in the final table today, so a latent leak. `2p_pct` formula is also wrong (operator precedence). |
| 12 | `oreb_rate`, `dreb_rate` (player level) | Computed as x/(x+x), always 0.5. |
| 13 | Early-season warm-up | Many rolling features `.fillna(0)`, so "no history" looks like "average team". Prefer NaN plus an indicator. |

### Verified clean

| Group | Why |
|---|---|
| 17 dynamic families x {r5, r10, r20, ewm5/10/20, s2d} (all `*_adv_*`, `*_ff_*`, `*_sc_*`, `*_usg_*`, `*_misc_*`, `*_ptrk_*`) and `home/away_team_net_*_r5/r10` | `shift(1)` before every rolling, ewm (`_ewm_rolling`) and expanding op. Data screen: no feature above noise. |
| `elo_diff`, `home/away_elo_roll_2Y`, `rolling_margin_diff_10` | Pre-game ratings are recorded before the update; the loop runs in date order; rolling uses `shift(1)`. |
| `rest_*`, `*_b2b`, `games_last_{3,5,7}_days`, `rest_weighted_load` | Strictly earlier dates only. |
| `win_streak`, `loss_streak`, `blowout_pct_*`, `margin_std_last_10`, `bench_net_rating_last_5` | Recorded before the update or shifted. |
| `travel_diff_*`, `fatigue_diff` | Distance from the previous venue to this one: schedule facts. |
| `spread_signed`, `is_home_favorite`, `payout_home/away` | Market inputs. Clean provided the line and price are captured before the bet is placed (see section 5). |
| `efg_diff`, `tov_pct_diff`, `oreb_pct_diff`, `ftr_diff`, `oliver_ff_composite`, `pyth_*` | Built in `experiment.add_oliver_features` from r10 rolled columns, so inherit their lag. |

### Method and limits

- Code read: every `rolling`, `ewm`, `shift`, `expanding`, `transform`, `ffill`, group `mean` and merge in `dataPrep/`.
- Data screen (`screen.py` in the session scratchpad): for every feature, small walk-forward LightGBM fit of
  abs(margin) and favorite-cover margin on [abs(line), feature]. A same-game leak shows up as extra R^2 on
  abs(margin) beyond the line alone. Positive controls (the four known leaks) were caught clearly.
- The screen only detects leaks tied to the game's *outcome*. A feature that encodes same-game *facts* unrelated
  to the result (for example #3 and #4) would pass it. Those were found by code reading only.

## 2. Fix design (fix, do not drop)

Principle: every feature keeps its meaning; only its **information set** changes to "as of the morning of the
game". Fixed versions are added under **new names** where the meaning shifts, so a stale cached column cannot be
used by mistake, and the old names stay in `DENY_EXACT`.

| Finding | Fix |
|---|---|
| 1 Absence features | Rebuild in two flavours from the boxscore `comment` field. **Pre-game**: lost r5 minutes/usage of players with status `DND` or `NWT` (official inactives, published before tip-off); top-1/top-2 flags use the lagged usage rank among players available *before* the game. **Previous-game**: players who missed the *previous* game (this is what the old names implied), via `shift(1)`. Exclude `DNP - Coach's Decision`, and treat `DNP - Injury/Illness` as unknown until we confirm when it is published. New names: `pregame_minutes_out`, `pregame_usage_out`, `pregame_missing_top1`, `pregame_missing_top2`, plus `prev_game_minutes_out`. |
| 2 `top8_points_diff` | Pick the top 8 by **lagged** r5 minutes among players not in the pre-game inactive list. Same value definition, no dependence on this game's rotation. |
| 3 Team style | Cluster on **prior** information: for each game date use style stats through the previous month end (first month: prior season). Add a test that the cluster of game *g* is unchanged if later games are deleted. |
| 4 Travel z-scores | Expanding z-score per (team, season) with `shift(1)`; fall back to the prior season's mean/std for the first games. |
| 5 `is_key_player` | Expanding (lagged) minutes and usage z-scores instead of full-history means. |
| 6 Push filter | Remove the row filter. Pushes stay in the history. The backtest already settles pushes at 0 (`compute_roi`). |
| 7 ATS cover | `ats_cover = (margin + team_spread > 0)`, keep `shift(1)` and `min_periods`. |
| 8 `net_rating_trend` | Add 15 to the rolling windows or use r20; drop the duplicate `net_rating_last_5`/`net_rating_diff` pair down to one name (keep the feature, remove the duplicate column). |
| 9 `after_ot_flag` | Derive OT from boxscore minutes (team minutes > 240) or the `ot_*` score columns, then lag by one game. |
| 10 Margin windows | Compute on the long (all games) team table with `transform(lambda x: x.shift(1).rolling(...))`, from the team's own perspective. |
| 11-12 Player pct/rate | Roll with `shift(1)`; fix the `2p_pct` precedence and the x/(x+x) rates. |
| 13 Warm-up | Leave NaN, add `*_has_history` indicator; LightGBM handles NaN. |

## 3. Verification (what makes the fixes provable)

1. **Point-in-time reconstruction test (gold standard).** Rebuild the feature table from data truncated at date D
   (outcomes of D and later removed; the schedule and closing lines for D kept). The rows for date D must equal the
   full-data rows for D, column by column, for a sample of dates across seasons. Any difference names a leaking
   column.
2. **Outcome-flip unit tests** on small synthetic fixtures for every function touched: flip the final score of
   game *g*; its own feature row must not change (only later games' rows may).
3. **Leakage screen as a test**: move `screen.py` into `tests/` (or `tools/`), run it on all features, fail if any
   feature's d_R2 on abs(margin) exceeds the noise cutoff (median + 4 MAD). Keep the four old absence features as the
   positive control.
4. **Re-run the pre-registered backtest** (`PREREG.txt` rules) only after 1-3 pass.
5. **Document the information cutoff** in `docs/` and `program.md`: "no column may use this game's boxscore", so
   the autoresearch loop cannot reintroduce one.

## 4. Order of work

| Phase | Work | Why this order |
|---|---|---|
| P0 | Findings 1 and 6 | Biggest effect on results and on which games exist; everything else depends on a correct sample. |
| P1 | Findings 2, 3, 4, 5 | Remaining leaks and look-ahead. |
| P2 | Findings 7-13 | Correctness and cleanup; changes values, so do before re-training. |
| P3 | Verification harness (section 3, items 1-3) | Built alongside P0-P2 and run after each phase. |
| P4 | Rebuild caches, re-run the backtest | Cached pipeline outputs (`load_cached_output`) must be invalidated first. |

All previous results (`results.tsv`, the earlier diagnosis numbers, the pre-registered run) are invalid for the
changed features and must be regenerated after P4.

## 5. Open questions for the owner

1. **Timing of absences versus the line.** Official inactives are published roughly 30 to 90 minutes before tip-off.
   Are the Kaggle and BetMGM lines captured before or after that? If before, the pre-game absence features carry real
   information the line lacks and can be bet on only after the list is out; if after, the market already prices them.
2. **`DNP - Injury/Illness`** (about 1,350 rows in the sample): in-game injuries or declared before tip-off? Default is
   to exclude them until confirmed.
3. **`payout_home/away`** as model features: they come from one book's price. They are available at bet time, but they
   let the model learn the book's vig behaviour instead of basketball. Keep or drop is a modeling choice; not a leak.

## 6. Implementation status (2026-10-07, branch `fix/feature-leakage`, uncommitted)

Decisions from the owner: ignore line timing; injury DNPs count as pregame; keep the price features.

| Finding | Status | What changed |
|---|---|---|
| 1 Absence features | Done | `pregame_missing_top1/2`, `pregame_minutes_out`, `pregame_usage_out`, `prev_game_minutes_out`. "Pregame" = not played, a reason is recorded, and it is not "Coach's Decision". Old names stay in `DENY_EXACT`. |
| 2 `top8_points_diff` | Done | Top 8 by lagged r5 minutes among players not pregame-out. |
| 3 Team style | Done | K-means refit each month on games before that month (prior season as fallback). |
| 4 Travel z-scores | Done | Expanding z-score of the home team's earlier games this season. |
| 5 `is_key_player` | Done | Top 5 by lagged minutes. Not used by the model today. |
| 6 Push filter | Done in step 2, **blocked at ingestion** | The 87 push games have no rows in `all_boxscores.csv`, so they still drop out at step 4. Needs a re-ingest (network). |
| 7 `ats_roll_last20` | Done | Cover = margin + handicap > 0; pushes count 0.5. |
| 8 `net_rating_trend` | Done | r5 minus r20 (there is no r15). `net_rating_diff` is now the 10-game gap, so the three net-rating columns are distinct. |
| 9 `after_ot_flag` | Done | Previous game's team minutes > 245 (9.9% of games). |
| 10 Margin/rest windows | Done | One all-games window per team (`add_team_form_features`), own perspective; rest and back-to-back use all games, not just home games. |
| 11-12 Player pct / rate | Done | Percentages rolled with `shift(1)`; rebound shares fixed. |
| 13 Warm-up zeros | **Not done** | Wide behavior change for little leakage value; revisit with the modeling work. |

New findings during implementation:
- **Duplicated boxscore rows (new, high impact).** `all_boxscores.csv` repeats 620 player-games 16 or 32 times (13,450 extra rows, 614 games). Step 3a summed them: 564 of 13,452 team-games had inflated totals (for example 379 points), which fed every dynamic rolling feature. In step 4 a duplicate row's lagged window included its own game. Both steps now call `helpers.dedupe_player_games`. After the fix, team points match the real score in 13,452 of 13,452 team-games. **Follow-up:** fix the fan-out in `fetch_player_boxscores.py`.
- **Step 6 could not run on its own.** It read `GAME_ID` as an int from the CSV and merged against a zero-padded string, giving all-NaN rolling features. Now normalized on both sides.
- **14 `usg_usage_pct` features are degenerate** (only 0 and 1). Not a leak; needs a look at the source column in `team_boxscores_v3.csv`.
- Step 6 sorts by date with an unstable sort, so row order of same-date games differs between runs (values are identical when keyed by `GAME_ID`).

### Verification results
| Check | Result |
|---|---|
| Unit tests | 30 pass (7 new in `tests/test_feature_leakage.py`: own-result invariance, future invariance, push games kept, all-games rest, ATS sign, pregame classification, duplicate collapse). |
| Point-in-time rebuild (`tools/pit_check.py`), 2023-01-04 (12 games) and 2024-01-05 (14 games) | **PASS**: every feature identical without later games and with the date's outcomes scrambled. |
| Same check on the original code (positive control) | **FAIL as expected**: flagged 10 columns (4 absence features, `top8_points_diff`, `team_style_vs_opponent_style`, 4 `travel_*_z`) and 1 game dropped by the push filter. |
| Leak screen (`tools/leak_screen.py`) on the rebuilt table | Max score 0.0002 against a 0.0054 cutoff; nothing flagged (the old leaks scored 0.11-0.14). |
| Pre-registered backtest, same rules as before, rebuilt features | **No edge.** Walk-forward rho 0.045-0.048 (break-even about 0.07); primary threshold: BetMGM-settled variant n=156, cover 51.9%, flat ROI -2.6%, CI [-17.4%, +12.2%]. Kaggle-only control: n=207, -4.7%. |

How to rebuild and re-verify: `python tools/run_features.py` (about 2.5 minutes), then `python tools/pit_check.py --date 2023-01-04`, `python tools/leak_screen.py`, `pytest tests`. Set `RUN_PIT=1` to include the slow point-in-time tests in pytest. Pre-fix tables are saved in `data/processed_backup_pre_leakfix/`.

## 7. Follow-up (2026-10-07): duplicates at the source, push games, usage features

**Duplicates.** The current ingestion code cannot produce them: each extra endpoint family is averaged per
(game, team, player) before it is merged (`fetch_player_boxscores.py`), and I added a base-table dedupe plus a hard
check (`_require_unique`) that raises if any consolidated table repeats a key. A test reproduces the situation (three
families returning two rows per player) and passes. The 13,450 duplicate rows in `all_boxscores.csv` therefore date from
an older build of that file (dated 2026-02-19, before this repo's single squashed import, so git cannot date the
fix). The file itself is now deduplicated (original saved as `data/processed_backup_pre_leakfix/all_boxscores_with_duplicates.csv`)
and every feature was rebuilt from it; the result is identical to the build that deduplicated downstream (0 of 318
columns differ). The downstream `dedupe_player_games` calls stay as cheap insurance.

Impact before the fix, measured on the saved team tables: 1,225 of 13,498 team-games (9.1%) had wrong totals (an
inflated game also corrupts its opponent's net stats). Through the rolling windows, `net_points_r5` was off by more than
5 points for 24% of team-games and `net_points_r20` for 50% (worst case about 500); season-to-date averages were off
by more than 1 point for 67%. All dynamic nba-api families were affected.

**Dynamic features re-tested on correct data** (walk-forward, raw-margin target, 2,100 out-of-fold games, standard
error of rho about 0.022): static only rho 0.037, dynamic only 0.039, all 0.031. Cleaning the data did not reveal hidden
signal: all three are at noise level and all raw models are about 0.43 RMSE worse than the line.

**The 87 "push" games.** `|margin| == spread` is not a push test. A push needs the favorite to win by exactly the spread.
Of the 87 games the old filter removed, 53 are true pushes (refunded, no effect on results) and 34 are games the
underdog won outright by exactly the spread. Those 34 are real, bettable outcomes that the old code silently deleted,
a small selection effect (0.5% of games, all underdog covers). The filter is gone in step 2, but none of the 87 has
rows in `all_boxscores.csv`, `team_boxscores_v3.csv` or the flat cache, so they still drop at step 4. `data/csv/player_boxscores.csv`
has all 87 (box score only: no DNP/inactive rows and no advanced or tracking metrics). The list is in
`data/processed/margin_equals_spread_games.csv`.

**`usg_usage_pct` (14 features: 7 windows x home/away).** Source: `usg_usage_percentage` in the team rows of
BoxScoreUsageV3. At team level it is 100% by definition (the team's usage share of the team), so it is the constant 1.0
in all 13,498 team-games, as are all 19 `usg_percentage_*` team columns and `adv_usage_percentage`. Rolled and
EWM-smoothed, the features reduce to 0 (fewer than 3 games of history) or 1. They carry no information. The player-level
version varies (mean 0.18; the top-2 players' usage per team-game averages 0.60, sd 0.09) and is what the feature
presumably meant to summarize.
