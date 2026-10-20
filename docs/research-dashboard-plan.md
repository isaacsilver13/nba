# NBA Prediction — Research Dashboard Handoff

Status: draft

Scope: `repos/nba_prediction` only. This is a local, research-only Streamlit dashboard; it is not a live picks product and must not call external data APIs.

## Goal

Replace the current mixed-lineage dashboard with an auditable research dashboard. A user must be able to select a single research run and see its provenance, data/odds quality, model performance, probability calibration, and betting-simulation results without accidentally combining artifacts from different runs.

## Current state and evidence

- `dashboard.py` already implements a Streamlit UI, but reads independent legacy files: `results.tsv`, `outputs/kelly_bets_summary.csv`, `outputs/copilot_betting_summary.csv`, `outputs/copilot_model_rmse.csv`, and the newest `outputs/copilot_analysis_odds_*` directory. These files do not share a run ID.
- `experiment.py` writes only aggregate metrics to `results.tsv`; it does not emit a dashboard-ready run bundle. Its `# FROZEN` boundary must remain untouched.
- Generated data and output artifacts are intentionally ignored by Git. Do not add them to source control.
- The current local `python` executable is broken (points to a missing Python 3.14 install). The available Anaconda interpreter has incompatible pandas/NumPy packages. Environment repair is required before runtime validation.
- Current artifact names contain historical aliases such as `copilot_*`; preserve them only behind compatibility adapters, not in the new dashboard’s public UI or new artifact contract.

## Constraints and non-goals

- Do not run ingestion, a full experiment/backtest, `autorun.py`, or any API-backed operation without the user’s explicit approval.
- Preserve chronological walk-forward evaluation and train-only fitting/calibration. Do not use dashboard work to retune a model.
- Do not edit below `# FROZEN` in `experiment.py`.
- Do not silently replace missing/mirrored/default odds with normal market-quality rows. Quality tiers must be visible and filterable.
- Do not make live betting recommendations, deploy the app, add authentication, or commit datasets, model binaries, output CSVs, credentials, or caches.
- Keep the existing `dashboard.py` command working; a split into small dashboard modules is preferred if it makes the data contract testable.

## Proposed architecture

### 1. Canonical read model: run manifests

Add a source-controlled module, e.g. `src/research_dashboard/run_catalog.py`, that discovers manifest files under:

```
outputs/research_runs/<run_id>/manifest.json
```

The dashboard reads manifests only through this module. Each manifest must include:

- `schema_version`, `run_id`, `created_at`, `run_kind`, and an immutable source/config fingerprint;
- input lineage: input paths (relative when possible), file modification time, size, optional hash, row/date coverage, and data-schema version;
- evaluation definition: target, feature-set fingerprint, chronological fold boundaries, and holdout definition;
- an artifact map with relative paths, row counts, required columns, and checksum/size for each CSV or JSON;
- odds-quality counts for `matched`, `mirrored`, `default`, and `unmatched` rows;
- explicit warnings and a `status` (`complete`, `partial`, or `failed`).

Use paths relative to the individual run directory. Reject path traversal and missing required files. A partial or invalid run may be listed, but cannot power performance charts.

### 2. Legacy compatibility adapter

Add a read-only adapter that can expose existing `results.tsv` and legacy `outputs/copilot_*` / `kelly_*` files as a clearly labeled **Legacy / unbundled** catalog entry. It must record each file path, timestamp, and incompatible/missing fields rather than pretending they were generated together.

Do not copy, rename, or modify legacy output files. The legacy adapter gives users a useful transition screen while new run bundles are adopted.

### 3. New run-bundle producer outside the frozen experiment

Add a small explicit command, e.g. `python -m src.research_dashboard.capture_run`, that creates a manifest around a completed, user-specified set of existing artifacts. It must:

1. Require explicit artifact paths or a named source profile; never choose “latest” files silently.
2. Validate schemas, row counts, dates, duplicate game/bet keys, and odds-quality columns before writing the manifest.
3. Write only into a new `outputs/research_runs/<run_id>/` directory and never overwrite a completed run.
4. Mark runs `partial` when a required artifact is missing or incompatible.

This produces a reproducible dashboard bundle without changing `experiment.py`’s frozen evaluation code. A future request to have `experiment.py` natively emit bundles requires a separate review of the frozen-boundary contract.

### 4. Dashboard pages

Refactor `dashboard.py` into a thin Streamlit entry point plus testable loaders/view helpers (or retain one file only if that is substantially simpler). Implement these views:

1. **Run Health** — selected run’s status, creation time, data coverage, configuration fingerprint, fold definition, warnings, and artifact freshness. No charts if provenance is invalid.
2. **Run Comparison** — compare complete run bundles in one table and in separate model-error, flat-stake uncertainty, selected-tier ROI, and full-run return/drawdown charts. Keep full-run and odds-tier-filtered metrics visibly separate, weight bankroll-relative P&L in common units, expose missing provenance fingerprints, and do not collapse reproducibility runs or imply unlike historical runs are directly comparable.
3. **Experiments** — aggregate experiment history from `results.tsv`, with model/config filters. Label this as an experiment log, separate from a selected run’s results.
4. **Model Quality** — fold-level RMSE/MAE if present, residual diagnostics, coverage, and calibration reliability plot with bin counts. Clearly label unavailable metrics rather than infer them.
5. **Odds & Join Quality** — matched/mirrored/default/unmatched counts and rates, duplicate-key warnings, and filters that exclude non-matched odds from performance metrics by default.
6. **Betting Simulation** — candidate count, selected bets, coverage, win rate, ROI, bankroll curve, drawdown, Kelly distribution, and daily exposure. Settled results only; label all units and starting bankroll.
7. **Artifacts** — downloadable manifest and selected run tables; do not expose filesystem paths outside the local app.

Global controls: run selector, date range, odds-quality filter, and “include legacy/unbundled data” toggle (off by default). Use an explicit refresh button and cache invalidation based on manifest/artifact modification time.

## Implementation sequence

1. **Repair and document the local environment.** Create a fresh project `.venv` with a supported Python version (3.11 or 3.12), install pinned/compatible dependencies, and add a short local setup section to `README.md`. Confirm imports for `pandas`, `streamlit`, `plotly`, `scikit-learn`, `lightgbm`, and `xgboost`. Update `requirements.txt` if a directly imported package is missing. Do not change global Python installations.
2. **Write small, synthetic fixtures.** Add only minimal CSV/JSON fixtures under `data/sample/` representing: a complete matched-odds run, a partial run, a duplicate game key, and a legacy unbundled state. Fixtures must contain no production data.
3. **Implement the manifest schema, catalog, and validation.** Use dataclasses or typed dictionaries; make error messages actionable. Add focused tests for schema validation, path safety, run discovery, and rejection of invalid artifact metadata.
4. **Implement `capture_run`.** It must create a complete fixture run from explicit files and produce a manifest that the catalog can load. Test idempotency/no-overwrite behavior.
5. **Refactor the Streamlit dashboard.** Build the Run Health page first, then Model Quality, Odds Quality, and Betting Simulation. Keep the legacy adapter isolated and visibly labeled.
6. **Add the experiment-log page.** Load `results.tsv` defensively, including malformed JSON parameters and missing/empty logs. Do not equate its rows with bundled run results.
7. **Validate manually against fixtures.** Launch Streamlit locally, select each fixture, check empty/partial/error paths, and inspect all charts for correct date and filter behavior. Do not run real training as part of this work.
8. **Document usage and limitations.** Explain run capture, artifact requirements, legacy state, and the difference between historical research results and live betting decisions.

## Data contract: minimum artifact schemas

The concrete column names should match the existing pipeline where possible. Do not invent values to satisfy the schema.

| Artifact | Required information | Notes |
| --- | --- | --- |
| `metrics.json` | aggregate RMSE/MAE, calibration metric when available, strategy metrics, fold count | Metrics carry units and denominators. |
| `fold_metrics.csv` | fold identifier/time window, rows, RMSE/MAE | Time windows must be ordered and non-overlapping in test data. |
| `bets.csv` | game ID, game date, selected side, odds-quality tier, probability, edge, stake/Kelly fraction, outcome, P&L, bankroll/drawdown | Reject duplicate `(run_id, game_id, bet_side)` rows unless an explicit market timestamp key exists. |
| `odds_audit.csv` | game ID/date, join result/tier, source/price availability | Report tiers separately; no defaulted odds in “matched” totals. |
| `model_metrics.csv` | model ID and aggregate/fold errors | Optional for models that do not emit component-level metrics. |

## Statistical guardrails to preserve

- Keep folds chronological; fit imputers, feature selection, scalers, and calibration only on prior training data.
- Keep model-quality evaluation separate from betting performance. Never optimize one opaque score alone.
- Present RMSE/MAE, calibration/probability quality, ROI, drawdown, bet coverage, Kelly sizing, and per-day exposure independently.
- Treat default or mirrored odds as a distinct quality tier; default-odds simulations are diagnostics, not market-realistic evidence.
- Do not claim an improved model from a single run/fold. Holdout conclusions require comparable runs and preserved manifests.

## Verification

- Automated:
  - Unit tests for manifest parsing/validation, safe relative artifact resolution, catalog ordering, and legacy adapter labels.
  - Tests proving a malformed/partial run cannot render performance charts as complete.
  - Tests for odds-tier filters and duplicate game/bet-key detection.
  - `python -m py_compile` on dashboard modules and the capture command.
- Manual:
  - `streamlit run dashboard.py` using the repaired `.venv`.
  - Verify the complete fixture renders all pages and figures.
  - Verify partial and duplicate-key fixtures show warnings/no misleading ROI.
  - Verify legacy artifacts appear only with the toggle enabled and remain marked unbundled.
  - Verify no network requests occur while browsing the dashboard.

## Definition of done

- A user can select one complete fixture run and trace every shown metric to declared artifacts in its manifest.
- A user cannot accidentally aggregate unrelated legacy files into one result.
- Missing, stale, duplicate, and lower-quality odds inputs are explicit in the UI.
- The app launches from a documented isolated environment and works without API credentials or network access.
- README includes local launch/capture instructions and research-only limitations.

## Risks / decisions needed

- Existing legacy artifacts may lack all fields needed for a complete bundle. Keep them in the adapter; do not fabricate provenance.
- The frozen `experiment.py` contract intentionally prevents native bundle output. Revisit that only in a separately approved modeling/evaluation change.
- Before using the dashboard for real decisions, independently validate the probability calibration and odds join logic on a time-separated holdout.
