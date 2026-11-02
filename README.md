# NBA Research

A Python research pipeline for NBA features, walk-forward modeling, historical betting simulations, and a Streamlit results dashboard. It also contains daily forecast and forward-only odds-collection workflows.

## What is here

- Ingestion and feature engineering, with tests for point-in-time behavior.
- Chronological training and evaluation, calibration, odds-match audits, and bankroll accounting.
- Run bundles that keep results, configuration, and provenance together.
- A bounded AI-assisted configuration search. Generated Python is validated against an allowlist before an experiment runs.
- Daily forecasts that use a different target from the historical spread-residual experiment; the operations guide explains that distinction.

This is a research project. Backtests are sensitive to data quality, odds timing, and repeated tuning. A historical score is not proof of a tradable edge.

## Local setup

Python 3.12 is the CI baseline. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt pytest
python -m pytest tests -q
```

Explore the committed synthetic run without downloading live data:

```powershell
$env:NBA_OUTPUTS_DIR = "data/sample"
streamlit run dashboard.py
```

Real experiments require the processed dataset described in the [operations guide](docs/operations-guide.md). Raw data and forecast artifacts are not supplied by this public repository. Point-in-time tests that require local datasets are separate from the fixture suite.

## Where to start

- [Development and operations](docs/operations-guide.md): data paths, training, run capture, forecasts, and scheduling.
- [Odds collector](docs/odds-collector.md): forward-only collection and operational setup.
- [Feature pipeline](src/ingest/dataPrep/PIPELINE_README.md): ingestion and feature-building entry points.
- [Research rules](program.md): experiment constraints. Machine-managed history in that file is not a performance guarantee.

Use environment variables for provider credentials. Keep raw datasets, generated models, and live forecast artifacts out of Git.
