# nba_prediction

This repository contains the NBA model training, backtesting, and dashboard workflow.
The betting methodology (markets, edge theses, evaluation rules, stages) is in [docs/methodology.md](docs/methodology.md).

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
python tools/predict_today.py                             # today (US/Eastern date); schedule from ESPN
python tools/predict_today.py --date 2024-03-10           # a past slate; schedule from the local games table
python tools/predict_today.py --date 2024-01-04 --check   # also compare with that date's walk-forward fold
```

Output: `$NBA_PREDICTIONS_DIR/<YYYY-MM-DD>.json`. The default is `<NBA_OUTPUTS_DIR>/predictions/`, i.e.
`outputs/predictions/` in this repo. The file is replaced atomically. The scheduled run (below) POSTs it to bet-watch.

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
- No regular-season games: nothing is written, and the script exits 0. A failed ESPN fetch exits non-zero, also
  without a file.

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

**Daily ingest.** The completed-games tables (`data/processed`) must be current through yesterday: the script refuses
to run if the last completed game is 10–100 or more than 240 days old. `tools/ingest_daily.py` keeps them current:

```bash
python tools/ingest_daily.py                  # games from the last ingested date through yesterday, then run_features
python tools/ingest_daily.py --no-features    # append only
```

- **Source: ESPN's public JSON.** stats.nba.com and cdn.nba.com refuse this PC (Akamai 403 or timeouts, 2026-10-21).
- **What it appends:** final regular-season games to `nba_games_with_game_id.csv`, with the closing spread from ESPN's
  odds (ESPN BET / DraftKings). Player box scores, DNPs included, go to `all_boxscores.csv`.
- **Checked against NBA data** on 68 games (2025-04-06..13): scores, closing lines, every team counting stat and the
  pregame-absence flags match exactly, and every player maps to their NBA id.
- **Known differences** from NBA-sourced games:
  - ESPN minutes are whole numbers, so a team's minutes can be off 240 by 1–3.
  - `team_boxscores_v3.csv` gets only the team metrics the model reads, computed from ESPN's team totals:
    - The four factors and the offensive, defensive and net ratings.
    - eFG% and FT rate are exact. TOV% is within 0.002.
    - The NBA counts possessions and OREB% from play-by-play, so these two use linear fits from 2022-24. On 2025-04
      games they are unbiased: per-game error is about 1.3 possessions and 0.02 OREB%, and ratings are within 1.4.
    - The other V3 metrics, and the player-level `adv_*`, `ff_*` and `sc_*` columns, stay NaN.
  - GAME_IDs are `002<yy>9<nnnn>`, because ESPN has no NBA id.
- **Reruns are safe:** a game is skipped if its (date, home, away) is already in the table. The last 7 days are re-read
  each run, for postponed games.
- Raw responses are cached in `data/cache/espn/`.

**Scheduling: GitHub Actions** (`.github/workflows/daily-predictions.yml`). The PC is not involved:

```
cron-job.org 09:30 CT -> workflow_dispatch -> restore the newest state-* release -> ingest_daily -> predict_today
  -> POST outputs/predictions/<date>.json to $BET_WATCH_URL/internal/nba-predictions
  -> release state-<date> with the forecast JSON in the private data repo
GitHub schedule 15:05 and 16:05 UTC (= 10:05/11:05 CDT, 09:05/10:05 CST) = backup trigger
```

- **State.** `tools/state.py pack` tars the four inputs the scripts need (`data/processed/nba_games_with_game_id.csv`,
  `all_boxscores.csv`, `team_boxscores_v3.csv`, `data/raw/arena_locations.csv`, ~30 MB gzipped) into `state.tar.gz`.
  `all_boxscores.csv` is over GitHub's 100 MB file limit, so the archive is a **release asset** on release
  `state-YYYY-MM-DD` in the private data repo (`vars.ODDS_DATA_REPO`), newest 14 kept. Everything else is rebuilt each
  run: `run_features` writes the derived tables, and the ESPN cache is refetched.
- **Private forecast handoff.** On game days the same `state-YYYY-MM-DD` release also has
  `predictions-YYYY-MM-DD.json`, copied from the exact JSON accepted by bet-watch. No forecast
  asset is added on a no-game day. The code repository is public; keep forecast artifacts in
  the private data repository. The betting aggregator's isolated paper trial can download
  this release asset with an authenticated `gh` session and verify the slate before intake.
- **Idempotent.** From 2026-10-23 onward, `complete-YYYY-MM-DD.txt` on the private `state-<today>`
  release (US/Eastern date) means today is done. It is uploaded last, after bet-watch accepts the
  POST (or there were no games) and the state and forecast assets upload. A partial release is
  repaired by the backup trigger using the newest complete state. Earlier releases use the legacy
  release-exists marker. The workflow checks that `ODDS_DATA_REPO` is private before saving.
- **No games:** no file and no POST. The job exits 0 and still saves state.
- The pip versions in the workflow are pinned to the PC's `.venv`, so `model_version` and the predictions match a local
  run. Bump them together.

One-time setup (the owner does these):

1. **Seed state** from the PC, with `data/processed` current. Tag it with **yesterday's** date, never today's: a
   `state-<today>` release tells the workflow today is done.
   ```bash
   python tools/state.py pack --out state.tar.gz
   gh release create state-YYYY-MM-DD state.tar.gz -R isaacsilver13/nba-odds-raw --title state-YYYY-MM-DD --notes seed
   ```
2. **Repo secret and variable** (Settings → Secrets and variables → Actions). `ODDS_DATA_REPO` and `ODDS_DATA_TOKEN`
   already exist for the odds collector; the token's *Contents: Read and write* covers releases.
   - Secret `BET_WATCH_INTERNAL_SECRET`: bet-watch's `INTERNAL_JOBS_SECRET`.
   - Variable `BET_WATCH_URL` = `https://bet-watch.fly.dev`.
3. **cron-job.org job**, set up like the odds collector's (`docs/odds-collector.md` step 7, same dispatch token):
   `POST https://api.github.com/repos/isaacsilver13/nba/actions/workflows/daily-predictions.yml/dispatches`, body
   `{"ref":"main"}`, daily at 09:30 America/Chicago, failure notifications on. A successful dispatch returns HTTP 204.
4. **First run:** `gh workflow run daily-predictions.yml`, then `gh run watch`. Check that bet-watch has the file
   and that a `state-<today>` release exists.

**Pulling state to the PC.** This **overwrites** the four files under `$NBA_DATA_DIR`:

```bash
gh release download state-YYYY-MM-DD -R isaacsilver13/nba-odds-raw -p state.tar.gz
python tools/state.py unpack state.tar.gz
python tools/run_features.py      # rebuild the derived tables
```

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
