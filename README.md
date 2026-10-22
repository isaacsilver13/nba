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
unmatched; the dashboard defaults to matched-only. Legacy `results.tsv` / `outputs/copilot_*` files appear only behind the
"legacy / unbundled" toggle (`legacy_dashboard.py`) and share no run ID. Existing pipeline outputs do not yet record a
per-bet odds tier; use `replay_run` below to bundle a real run. `data/sample/` holds a synthetic run
(`python -m tests.sample_run` regenerates it).

The **Run Comparison** tab compares complete bundles in one table plus full-run model-error, flat-stake uncertainty,
selected-tier ROI, and full-run return/drawdown charts. Full-run and odds-tier-filtered metrics are labeled separately.
It keeps reproducibility runs visible and warns when historical bundles lack source/config/data fingerprints or may
reflect different code, data, calibration, sign, or odds logic; compare only runs with compatible provenance.

## Feature pipeline

`python tools/run_features.py` rebuilds the model table (`data/processed/df_model_3.csv`, ~40 s) from the ingested box
scores. `python -m src.ingest.dataPrep.pipeline --mode ingest|features|full` is the full entry point; the ingest steps
download from stats.nba.com and need `nba_api`. See `src/ingest/dataPrep/PIPELINE_README.md`.

## Daily predictions for bet_watch

`tools/predict_today.py` writes the model's view of one day's regular-season slate for bet_watch's 10:00 CT
`ModelPicksJob` (daily report plan, step 3a):

```bash
python tools/predict_today.py                             # today (US/Eastern date); schedule from stats.nba.com
python tools/predict_today.py --date 2024-03-10           # a past slate; schedule from the local games table
python tools/predict_today.py --date 2024-01-04 --check   # also compare with that date's walk-forward fold
```

Output: `$NBA_PREDICTIONS_DIR/<YYYY-MM-DD>.json`. The default is `<NBA_OUTPUTS_DIR>/predictions/`, i.e.
`outputs/predictions/` in this repo, and bet_watch reads the same folder. The file is replaced atomically.

```json
{"slate_date": "2024-03-10", "model_version": "margin-noline-df6993e1",
 "generated_at": "2026-10-21T14:30:05+00:00", "data_through": "2024-03-09",
 "games": [{"game_date": "2024-03-10", "home": "MIA", "away": "WAS",
            "pred_margin": 13.53, "p_home_win": 0.8064, "sigma": 13.97}]}
```

- `home` and `away` are stats.nba.com tricodes (`GSW`, `NOP`, `NYK`, `SAS`, `UTA`, `WAS`, …). These are bet_watch's
  NBA `teams.abbr`.
- `game_date` is the NBA's US/Eastern schedule date.
- `pred_margin` is home minus away points.
- `p_home_win = Φ(pred_margin / s)`, with `s` fit by probit on the walk-forward out-of-sample predictions.
- `sigma` is their residual SD. P(home covers handicap L) = `Φ((pred_margin + L) / sigma)`.
- `data_through` is the last completed game the features saw.
- No regular-season games (or a failed scoreboard fetch): nothing is written, and the script exits 0.

**Model.** The script uses experiment.py's models, ensemble weights and 2,500-game rolling window, refit each run on
the newest completed games. It is kept free of the closing line in two ways:
- **Excluded features:** the closing spread, the favourite flag, the spread prices (`payout_*`) and the same-day
  inactive-list features.
- **Target:** the **raw home margin**. experiment.py learns the market's error and adds the closing spread back, so its
  predictions contain the close even when `spread_signed` is excluded. That would be look-ahead against a 10am
  price.

`--check` on a date that starts a walk-forward fold reproduces that fold's predictions exactly. Other dates use a newer
window, so their predictions differ.

**Slate features.** `python tools/run_features.py --slate DATE` builds the pregame rows from data strictly before
DATE: today's games get no score, line or box score. `pregame_missing_top1/2`, `pregame_minutes_out` and
`pregame_usage_out` are left NaN, because the inactive list comes out 30–90 minutes before tip. `RUN_PIT=1` tests check
that every other column equals the full build.

**Before it can run live**, the completed-games tables (`data/processed`) must be current through yesterday. The script
refuses to run if the last completed game is 10–100 or more than 240 days old. Today the tables end 2025-04-13.

**Scheduling** (Windows Task Scheduler, 9:30 CT daily, model only). **Do not register yet.** Register it after the
root-folder move `nba_prediction` → `codebase`, which changes the paths below:

```bat
schtasks /Create /TN "NBA predict_today" /SC DAILY /ST 09:30 /TR "cmd /c \"cd /d C:\Users\justj\nba_prediction\repos\nba_prediction && .venv\Scripts\python.exe tools\predict_today.py >> outputs\predict_today.log 2>&1\""
```

`/ST` is the PC's local time (Central). Set `NBA_PREDICTIONS_DIR` as a user environment variable (`setx`) only if
bet_watch should read somewhere other than `outputs\predictions`.

### Capturing a real run (replay)

```bash
.venv\Scripts\python -m src.research_dashboard.replay_run --yes
```

Calls the real `experiment.run_experiment()` (trains the configured models on local data, no network, does not touch
`results.tsv` or `experiment.py`) while recording per-row predictions, bets and odds tiers, then writes a bundle via
`capture_run`. It aborts if its ROI differs from `experiment.py`'s or its odds-join replica disagrees with `load_data`.
Note `p_home` is P(home covers the spread) as `compute_roi` defines it, not P(home wins).

## Odds collection

Forward-only NBA odds snapshots (props, 1H/quarter lines, moneylines) are collected by
`src/odds_collector` on GitHub Actions — setup and operations in [docs/odds-collector.md](docs/odds-collector.md).


## Causal evaluation and settlement

The top-player feature uses a previous complete team-date snapshot, including
roster membership. It is deliberately one team-date behind the per-player rolling
stats; target-game participants cannot affect it. Untimestamped absence features
and closing-line features are frozen exclusions, including explicit inclusions.
Rebuild processed features and revalidate before relying on old scores. Folds
round minimum row targets to complete dates, preserve row identity and include
the final partial test block. Same-day bets use one opening bankroll and settle
together. Replay records `day-opening-v1` and `bankroll_open`; older bundles keep
their original provenance. Autorun accepts validated literals and allowlisted
constructors with bounded workers/model parameters, ten iterations by default
and a maximum 1800-second per-experiment timeout. Environment credentials are preferred; legacy token-file fallback is loaded only
for an explicit live invocation. Importing the runner does not read token files. Fixture
tests need no API calls or model training.
