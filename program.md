<!-- autorun:state:start -->
## Current state (auto-updated by autorun.py)

**Best score so far:** `-0.058033`

**Recent experiment history:**
```
          timestamp   exp_id  ensemble_rmse  roi     score
2026-05-25T02:18:30 96fda43c        16.2266 -1.0 -0.058050
2026-05-25T02:21:22 612b7e0e        16.2161 -1.0 -0.058085
2026-05-25T02:23:56 612b7e0e        16.1940 -1.0 -0.058160
2026-05-25T02:25:36 96fda43c        16.2190 -1.0 -0.058075
2026-05-25T02:27:36 96fda43c        16.2190 -1.0 -0.058075
2026-05-25T02:29:32 96fda43c        16.2190 -1.0 -0.058075
2026-05-25T02:32:27 0ebec3c1        16.2315 -1.0 -0.058033
2026-05-25T03:52:42 612b7e0e        16.1940 -1.0 -0.058160
```

**Already tried this session — DO NOT repeat these changes:**
  Iter 1: score=-0.058160
    --- 
    +++ 
    -"""
    -NBA Prediction - Autoresearch Experiment Script
    -================================================
    -This is the file the AI agent modifies each iteration.
    -All tunable configuration lives in the AGENT-EDITABLE CONFIG block.
    -Data loading and metric calculation are FROZEN below the marked boundary.
<!-- autorun:state:end -->


# NBA Prediction — Autoresearch Program

## Goal

Maximize `score = flat_roi - flat_roi_se` where, over the bets the strategy places on held-out walk-forward folds:
- `flat_roi` = mean return per bet at a 1-unit stake (not compounded)
- `flat_roi_se` = its standard error, so a noisy edge on few bets scores lower
- fewer than 100 bets scores `-1.0` (too little evidence)

Win probabilities are calibrated walk-forward: each fold's sigma is scaled using only earlier folds, so
overconfidence is corrected rather than rewarded. `roi` (compound Kelly) and `ensemble_rmse` are still logged
as diagnostics but do not decide keep/revert. This is the single number that decides whether a change is kept or reverted.

---

## Your job each iteration

1. Read `experiment.py` carefully — understand what the current config does.
2. Propose **one focused change** to the `AGENT-EDITABLE CONFIG` section.
3. Return **ONLY the modified AGENT-EDITABLE CONFIG section** — no markdown fences,
   no explanations, no frozen code. Stop before the `# ═══` separator line.
4. Do **NOT** include anything below (or including) the `# FROZEN` comment line.

---

## What you may change

| Config variable | Typical range | Notes |
|---|---|---|
| `TRAIN_SIZE` | 1500–2500 | More data per fold = better stability |
| `TEST_SIZE` | 200–500 | Larger = more reliable eval |
| `EV_THRESHOLD` | 0.01–0.06 | Higher = bet only on strong edges |
| `FRACTIONAL_KELLY` | 0.20–0.75 | Lower = less variance, more conservative |
| `MAX_KELLY` | 0.05–0.15 | Hard cap per bet |
| `TOP_N_BETS_PER_DAY` | 2–10 | Fewer = more selective |
| `DAILY_MAX_EXPOSURE` | 0.10–0.40 | Total risk per game-day |
| `MODEL_SPECS` | any | Add, remove, or tune models |
| `ENSEMBLE_WEIGHTS` | see below | `"inverse_rmse"`, `"equal"`, or dict |
| `EXTRA_FEATURE_EXCLUSIONS` | list of strings | Drop noisy columns |
| `EXTRA_FEATURE_INCLUSIONS` | list of strings | Force-add columns |

**`ENSEMBLE_WEIGHTS` options:**
- `"inverse_rmse"` — weight each model by 1/RMSE (default, best models get more weight)
- `"equal"` — equal weight to all models
- `{"lgb_v1": 0.4, "lgb_v2": 0.3, "xgb_v1": 0.2, "elasticnet": 0.1}` — manual weights

---

## What you must NOT change

- Any code below `# FROZEN — do not modify anything below this line`
- Data loading logic (`load_data`)
- Walk-forward split logic (`walk_forward_splits`)
- Metric calculation (`compute_roi`, `calibrate_sigma`, `flat_stake_stats`, RMSE computation, `score` formula)
- Results logging (`write_results`, `params_summary`, the `results.tsv` format)

---

## Research directions to explore

> **CURRENT SITUATION (read this first):**
> Results before 2026-10-08 were produced with backtest bugs and same-game feature leakage (see
> `docs/feature-leakage-audit-and-fix-plan.md`) — do not compare against them. On the corrected features the model
> shows no edge over the closing line (flat ROI about 0% per bet, CI roughly ±10%). Improvements must show up as a
> higher `score`, which means a real, repeatable edge on enough bets. Lower `ensemble_rmse` helps only if it lifts
> `score`; the market line itself has RMSE about 13.35. Every config is scored on the same folds, so a small `score`
> gain over many tries is expected by chance alone.

### Priority 1 — Reduce RMSE (do these first)

1. **Tune lgb_v1 regularization** — current `min_child_samples=30` may be underfit.
   Try `min_child_samples=50, num_leaves=20` for a simpler, less-overfit tree.

2. **Tune lgb_v2** — try `num_leaves=31, min_child_samples=40, n_estimators=1000`
   (bring it closer to lgb_v1 to reduce ensemble variance).

3. **Add XGBoost** — uncomment the `xgb_v1` spec and set `n_estimators=300,
   learning_rate=0.05, max_depth=4, reg_alpha=0.1, reg_lambda=1.5`. A third
   diverse model often reduces ensemble RMSE.

4. **Feature exclusions** — drop noisy rolling box-score columns that add noise:
   `EXTRA_FEATURE_EXCLUSIONS = ["home_team_net_fgm_r5", "away_team_net_fgm_r5",
   "home_team_net_fga_r5", "away_team_net_fga_r5"]`

5. **Reduce ElasticNet alpha** — try `alpha=0.3, l1_ratio=0.7` for less
   regularization (ElasticNet may be underfitting on 300-feature input).

6. **Tune FF_WEIGHTS** — try `{"efg": 0.50, "tov": 0.20, "oreb": 0.15, "ftr": 0.15}`
   to put more weight on shooting quality (eFG% is the strongest Four Factor).

7. **Tune PYTH_EXPONENT** — try `13.91` (the value James used originally) or `14.0`.

### Priority 2 — Betting params (only relevant once RMSE < 15.5)

8. **Tighter EV threshold** — `EV_THRESHOLD = 0.04` to bet only on strong edges.

9. **Conservative Kelly** — `FRACTIONAL_KELLY = 0.25, MAX_KELLY = 0.06`.

---

## One change per iteration

Make exactly one logical change per experiment so the effect is interpretable.
If multiple parameters need adjusting (e.g. both Kelly fraction and EV threshold),
do them in separate iterations.
