# Data Layout

Large datasets should live outside this repository in a dedicated shared-data location.

## Recommended structure

- Shared data root: `NBA_DATA_DIR`
- Required files under `NBA_DATA_DIR`:
  - `processed/df_model_3.csv`
  - `processed/nba_games_with_game_id_processed.csv`
  - `odds/nba_2008-2025.csv`

## Local development

- Keep only minimal sample files in `data/sample/` for CI and smoke tests.
- If `NBA_DATA_DIR` is not set, code falls back to this repo's `data/` directory.

## Parquet lake

Large tables live in a partitioned Parquet lake instead of SQLite/CSV copies. The layout and API are owned by
`src/ingest/datalake.py`; the lake sits at `$NBA_DATA_DIR/lake` (default `data/lake/`, git-ignored).

```
lake/play_by_play/season_year=2019-20/part-<uuid>-0.parquet     # partitioned by season, zstd compressed
lake/player_boxscores/season_year=.../
lake/team_boxscores/season_year=.../
lake/game_odds/part-<uuid>-0.parquet                            # small tables: one flat dataset
lake/games_index/   lake/games_schedule/
lake/_reports/                                                   # JSON reports from the tools below
```

Rules: `game_id` is a 10-character string (leading zeros kept), `season_year` is the text label such as `2019-20`
(for `play_by_play` it is derived from `game_id[3:5]`, pivot 96, and checked against `games_index`), compression is zstd.

### Build and verify

```powershell
python tools/sqlite_to_parquet.py                 # convert data/sql/*.sqlite in chunks, then verify every table
python tools/sqlite_to_parquet.py --verify-only   # re-verify later
python tools/compare_csv_to_lake.py               # is data/csv/*.csv an exact copy of the lake? (IDENTICAL / DIFFERENT)
```

The converter opens the SQLite files read-only, writes to a hidden staging folder, and replaces a lake table only
after every chunk is written (an existing table needs `--overwrite`). `verify` compares row counts, null counts,
numeric sums and text-length sums with the source. Nothing here deletes `data/sql`, `data/csv` or any backup.

`compare_csv_to_lake.py` also reports CSV rows whose field count differs from the header ("CSV is malformed"). Run on the real data, it found 5,292 such rows in `play_by_play.csv` (a description with an unquoted comma, e.g. `Double Technical - Castle, McDaniels (T.Ford)`, was split in two, shifting the later columns). The lake, which matches the SQLite source, holds those descriptions intact, so that CSV is the damaged copy and not a second source of truth.

Each run records its results in `lake/_reports/sqlite_to_parquet.json` (merged per table across runs). A table counts
as *verified* only if its last verification passed and nothing has been written to it since. Tools that used to read a
`data/csv` file (`tools/backfill_boxscores.py`, `tools/encompass_check.py`) read the lake only once the table is
verified and say which source they used; otherwise they keep reading the CSV (`--source lake|csv` forces one).
Lake writes are safe by construction: `write_table` overwrites via a staging folder (a failure keeps the old table)
and refuses to append data whose columns or types differ from the table's.

### Which engine

| Data size | Use |
|---|---|
| under ~1 GB after column/filter pruning | pandas: `datalake.read_pandas("play_by_play", columns=[...], filters=[("season_year", "==", "2019-20")])` |
| large joins and aggregates | DuckDB: `datalake.duck().execute("SELECT season_year, count(*) FROM play_by_play GROUP BY 1").df()` |
| outgrows one machine | PySpark (local mode here): `datalake.get_spark()` + `datalake.spark_read(spark, "play_by_play")` |

DuckDB and PySpark are optional: `pip install -r requirements-bigdata.txt` (PySpark 4 needs Java 17 or 21 on `PATH`
or `JAVA_HOME`). PySpark currently warns that it does not fully support pandas 3; the checks here avoid `toPandas()`.
Spark only reads the lake; all writes go through `datalake.write_table` (pyarrow), because Spark's local writes can
need Hadoop binaries on Windows. `python tools/spark_lake_check.py` confirms Spark and DuckDB agree on per-season counts.
