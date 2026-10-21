# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project boundary

NBA data-ingestion, feature-engineering, modeling, betting simulation, dashboard, and autoresearch workflows. Run Python commands from this repository root so `src` imports and relative data paths resolve consistently.

## Commands

```
pip install -r requirements.txt
python experiment.py                                   # run one experiment/backtest
streamlit run dashboard.py                              # launch dashboard
python autorun.py --dry-run                             # autoresearch loop, no LLM calls
python autorun.py --groq --max-iters 20                 # autoresearch loop, LLM-driven config edits
```
`autorun.py` resolves `experiment.py`, `program.md`, `results.tsv`, and `experiments/` relative to its own location, so it can be invoked from any working directory. Data locations are overridable via `NBA_DATA_DIR` / `NBA_OUTPUTS_DIR` env vars (default `./data`, `./outputs`).

Tests: `python -m pytest tests -q` (CI runs them). They need no local data; the slow point-in-time leakage check (`tools/pit_check.py`, ~2 min, needs `data/processed`) runs with `RUN_PIT=1`. After changing feature code, also run `python tools/leak_screen.py` — see `docs/feature-leakage-audit-and-fix-plan.md`.

Do not commit raw or processed datasets, generated outputs, model artifacts, runtime logs, or credentials — small reproducible fixtures belong under `data/sample/`.

## Architecture: the autoresearch loop

This repo's central mechanism is `autorun.py` driving an LLM (Groq or OpenAI, per `requirements.txt`) to iteratively edit `experiment.py`:

- `experiment.py` is split by two markers autorun.py parses directly: `# AGENT-EDITABLE CONFIG` (an agent/LLM may freely rewrite this block — training size, test size, and other tunables) and `# FROZEN` (everything from this line down — data loading, walk-forward split logic, leakage filters, target construction, Kelly ROI calc, results logging — is a correctness boundary, never a tuning knob). `autorun.py` refuses to proceed if it can't locate `CONFIG_MARKER` in the file, rather than silently guessing a boundary.
- Each iteration: autorun.py sends the current AGENT-EDITABLE CONFIG (plus `program.md`'s rules and recent score history) to the LLM, expects back *only* a replacement CONFIG block, splices it into `experiment.py`, runs the experiment, and scores it as `flat_roi - flat_roi_se` (flat-stake return per bet minus its standard error, probabilities calibrated walk-forward; `-1.0` under 100 bets) — higher is better. `program.md` is auto-updated by autorun.py with a `<!-- autorun:state:start/end -->` block recording best score and recently-tried changes (to avoid the LLM repeating them) — this block is machine-managed, don't hand-edit it.
- `results.tsv` is the append-only log of every experiment run (`experiment.py`'s FROZEN logging function writes to it).
- When working on the modeling code directly (not via the autorun loop): keep walk-forward splits chronological, fit any preprocessing/feature-selection/transforms on the training fold only, and report RMSE, calibration, odds-match quality, ROI, drawdown, and Kelly behavior separately rather than treating one run as proof a change helped.

## Data pipeline

- `src/ingest/dataPrep/config.py` owns shared DataPrep defaults and per-step overrides; `src/ingest/dataPrep/pipeline.py` owns orchestration; individual steps live under `src/ingest/dataPrep/steps/`.
- `experiment.py` is the only model/backtest code. Earlier model scripts (`src/ingest/models/`, `src/ingest/old/`) and autorun snapshots were removed 2026-10-08; they were built on leaky features and are in git history if needed.
- For lineup ingestion, official boxscore-listed players count as appearances (including DNPs) — don't add a separate player-stat download path for this.
- Prefer read-through caches and full artifact reconstruction over partial-cache assumptions; check cache completeness, file dates, and column names before re-fetching. Preserve existing cache paths/schemas unless a migration is explicit.

## Working rules

- `README.md` owns setup and entrypoints; `program.md` + `experiment.py` own autoresearch constraints (the `FROZEN` section and results format are protected).
- Preserve join audits, and identify fallback/default odds rows before interpreting betting results.
- One focused experiment change at a time; record the exact config and artifact paths.
- Inspect the owning module, neighboring tests, and current artifact schema before editing. Make the smallest change, run the narrowest import/compile/test/dry-run after each edit, and don't claim a historical fix still works without current evidence.
- Project slash commands in `.claude/commands/`: `/nba-model-review` (leakage/join/metric review before proposing one change) and `/nba-dataprep` (one narrow DataPrep slice).
