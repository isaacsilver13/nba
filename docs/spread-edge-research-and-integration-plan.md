# NBA Prediction — Backtest Integration and Spread-Edge Research Plan

Status: draft

Scope: `repos/nba_prediction` only. This plan covers the corrected historical
backtest, its integration into `main`, and a controlled search for an
out-of-sample spread edge. It does not add or connect any external odds API.

## Goal

Safely integrate the backtest-correctness work, establish a reproducible
baseline, and evaluate a small number of pre-declared model changes without
mistaking noisy backtest returns for a betting edge.

## Current evidence

- `fix/backtest-alignment-calibration` is at `d6d3473` (`fix: backtest
  alignment, edge sign, spread odds, calibrated sigma, flat-stake score`). It
  fixes same-date prediction alignment, the home-spread edge sign, in-sample
  sigma estimation, and spread-price loading; it also changes selection to the
  flat-stake lower-bound score.
- All pre-fix experiment rows are archived in
  `results_archive_pre_fix_2026-10-07.tsv` and are not comparable evidence.
- The calibrated historical replay covers 14 chronological folds and 4,200
  scored games. A previous bundle showed a small positive flat ROI whose
  confidence interval included zero.
- A later rerun kept RMSE at 13.5464 but changed selected-bet count and
  flat-stake results. Before tuning, the team must identify whether this is a
  model/thread-order effect, a data-lineage difference, or a betting-selection
  tie/order effect.
- Existing untracked `data/` artifacts and Odds API planning documents are
  outside this plan. Do not stage, move, delete, or connect them.

## Constraints and non-goals

- Keep the `# FROZEN` portion of `experiment.py` unchanged unless a separately
  approved correctness bug is demonstrated with a regression test.
- Work only with local historical data; do not call or configure external odds
  APIs, create accounts, or add credentials.
- Do not commit raw/processed data, local output bundles, logs, caches,
  `.env` files, or credentials.
- Do not promote a strategy from one rerun, one fold, or a positive compounded
  Kelly path. The decision metric remains flat ROI minus its standard error,
  with matched-odds quality reported separately.
- One logical configuration change per experiment. No automated long-running
  autoresearch loop until the deterministic baseline gate passes.

## Approach

### 1. Inventory and review the integration candidate

1. From `repos/nba_prediction`, record `git status --short`, the current branch,
   `git merge-base main HEAD`, and `git diff --stat main...HEAD`.
2. Classify every changed/untracked path as either intended source, generated
   output, or pre-existing user work. Stage only reviewed source, test, and
   documentation files. In particular, exclude `data/all_odds.csv`, `data/csv/`,
   `data/sql/`, outputs, logs, and any credentials.
3. Run the focused regression suite with the repository `.venv`, including
   `tests/test_replay_run.py` and `tests/test_research_dashboard.py`; run a
   replay-manifest checksum check for a fresh bundle.
4. Obtain an independent code review of `d6d3473`, with special attention to
   chronology, target/spread sign conventions, leakage, matched/default odds
   tiers, and score units. Fix confirmed findings and rerun affected checks.

### 2. Commit and merge only the reviewed correction

1. If the source changes behind `d6d3473` are already complete and the working
   tree contains no intended source edits, preserve that commit rather than
   creating a duplicate. If review fixes are needed, create one focused follow-up
   commit with tests.
2. Inspect the branch diff against the current `main` immediately before
   integration. Resolve conflicts without touching ignored data or unrelated
   untracked files.
3. Open a PR to `main` (or use the repository's approved direct-merge process),
   attach the review and verification evidence, and merge only after the review
   gate passes. Do not force-push or rewrite history.
4. After the merge, check out/update `main` and capture the resulting commit SHA
   and clean tracked-file status. This establishes the research baseline.

### 3. Reproducibility gate before searching for an edge

1. Run two full, local `src.research_dashboard.replay_run --yes` replays from the
   same merged SHA, local data root, and configuration, each with its own run ID.
   Record Python/package versions, platform, CPU-thread settings, input paths,
   input file sizes/hashes, fold boundaries, and manifest checksums.
2. Compare predictions row-by-row (game ID, fold, prediction, sigma), selected
   side/count, and flat-stake statistics. The expected outcome is identical
   prediction and selection artifacts.
3. If artifacts differ, stop tuning. Diagnose deterministic settings available in
   the editable model configuration (for example fixed LightGBM threading and
   deterministic mode) and ordering/tie behavior, then add a regression test or
   a documented reproducibility check. Do not interpret either differing return
   as an edge.
4. Once identical, name that verified bundle pair the baseline and preserve their
   manifests locally. Its metrics—not archived pre-fix results—are the only
   comparator for later experiments.

### 4. Establish an untouched final evaluation period

1. Reserve the latest chronological season segment before making model choices;
   it must not be used to choose hyperparameters, feature exclusions, model
   weights, thresholds, or staking parameters.
2. Use earlier chronological folds for exploratory walk-forward comparisons and
   keep the holdout definition, input checksums, and candidate list in the run
   manifest/experiment notes.
3. Report the market-line RMSE, model RMSE, calibration, matched-odds coverage,
   flat ROI/standard error, coverage, drawdown, and daily exposure separately on
   both the exploratory period and untouched holdout.

### 5. Run a bounded, discriminating model search

1. Begin with three pre-declared, one-at-a-time changes in the editable config:
   - deterministic LightGBM configuration (only if the reproducibility gate
     requires it);
   - one LightGBM complexity/regularization change;
   - one feature exclusion or one diverse-model inclusion.
2. For each candidate, run the full chronological replay and capture a new,
   checksummed bundle. Compare it with the verified baseline on the same
   evaluation period; do not compare different data roots, fold definitions, or
   odds tiers.
3. Advance a candidate only if it has at least 100 selected bets, improves the
   flat-score metric, does not degrade calibration/market comparison materially,
   and remains non-negative on the untouched holdout. Compound Kelly ROI is a
   secondary diagnostic only.
4. If none of the bounded candidates clears these gates, stop model tuning and
   audit feature availability timing, team/date joins, market-line benchmark,
   and selection concentration before proposing another search direction.

### 6. Review together before any broader iteration

After the baseline and each candidate, bring a concise comparison to chat:

| Item | Baseline | Candidate | Decision |
| --- | ---: | ---: | --- |
| Model/market RMSE | | | |
| Calibration / sigma multiplier | | | |
| Matched-odds coverage | | | |
| Selected bets / coverage | | | |
| Flat ROI ± SE | | | |
| Drawdown / daily exposure | | | |
| Untouched-holdout result | | | |

Discuss whether the result supports one more isolated test, an input/market audit,
or stopping the search. The default conclusion remains “no demonstrated edge”
unless the holdout and uncertainty gates are satisfied.

## Verification

- Automated: focused replay/dashboard tests; artifact-manifest checksum checks;
  exact-replay comparison after the reproducibility gate is implemented.
- Manual: inspect branch diff and staged-file list; confirm no raw data, generated
  bundles, credentials, or Odds API integration are included.
- Review: independent review of correctness-sensitive backtest code before merge
  and of any subsequent modeling change.

## Risks and decisions needed

- Confirm whether integration should use a PR into `main` or an approved direct
  merge. This plan assumes a PR because the workspace requires independent review
  for meaningful changes.
- The different selected-bet counts across otherwise similar replays are a
  blocking reproducibility issue, not evidence that the strategy improved or
  worsened.
- Historical odds provenance limits remain. Defaulted/mirrored/unmatched rows
  must never be presented as matched market outcomes.
