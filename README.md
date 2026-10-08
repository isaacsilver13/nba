# nba_prediction

This repository contains the NBA model training, backtesting, and dashboard workflow.

## Quick start

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure data locations (optional):

```bash
set NBA_DATA_DIR=C:\\path\\to\\shared-data\\nba
set NBA_OUTPUTS_DIR=C:\\path\\to\\shared-data\\nba\\outputs
```

If these are not set, defaults are `./data` and `./outputs` under this repo.

4. Run an experiment:

```bash
python experiment.py
```

5. Launch dashboard:

```bash
streamlit run dashboard.py
```

6. Run the autoresearch loop from any working directory:

```bash
python /path/to/nba_prediction/autorun.py --dry-run
python /path/to/nba_prediction/autorun.py --groq --max-iters 20
```

`autorun.py` resolves `experiment.py`, `program.md`, `results.tsv`, and
`experiments/` relative to its own repository, so callers do not need to
change directories first. Do not commit raw or processed datasets, generated
outputs, model artifacts, or API credentials; use `data/sample/` for small
reproducible fixtures.

## Research dashboard (local, research-only)

Historical research results only — not live picks. No network or API access needed.

```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt pytest
.venv\Scripts\streamlit run dashboard.py            # or: set NBA_OUTPUTS_DIR=data\sample for the synthetic sample run
.venv\Scripts\python -m pytest tests -q
```

The dashboard shows one **run bundle** at a time (`outputs/research_runs/<run_id>/manifest.json` plus `metrics.json`,
`fold_metrics.csv`, `predictions.csv`, `bets.csv`, `odds_audit.csv`). Create one from existing artifacts with explicit paths
(never auto-picks "latest"; refuses to overwrite; missing/bad artifacts make the run `partial` and disable its charts):

```bash
python -m src.research_dashboard.capture_run --run-id r1 --metrics m.json --fold-metrics f.csv \
    --predictions p.csv --bets b.csv --odds-audit o.csv
```

Required columns are in `src/research_dashboard/manifest.py` (`ARTIFACTS`). Odds tiers: matched / mirrored / default /
unmatched; the dashboard defaults to matched-only. The Experiment Log tab reads `results.tsv`, whose rows are not
results of the selected run. Real runs are bundled with `replay_run` (below); `data/sample/` holds a synthetic run
(`python -m tests.sample_run` regenerates it).

## Feature pipeline

`python tools/run_features.py` rebuilds the model table (`data/processed/df_model_3.csv`, ~40 s) from the ingested box
scores. `python -m src.ingest.dataPrep.pipeline --mode ingest|features|full` is the full entry point; the ingest steps
download from stats.nba.com and need `nba_api`. See `src/ingest/dataPrep/PIPELINE_README.md`.

### Capturing a real run (replay)

```bash
.venv\Scripts\python -m src.research_dashboard.replay_run --yes
```

Calls the real `experiment.run_experiment()` (trains the configured models on local data, no network, does not touch
`results.tsv` or `experiment.py`) while recording per-row predictions, bets and odds tiers, then writes a bundle via
`capture_run`. It aborts if its ROI differs from `experiment.py`'s or its odds-join replica disagrees with `load_data`.
Note `p_home` is P(home covers the spread) as `compute_roi` defines it, not P(home wins).
