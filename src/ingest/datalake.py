"""Partitioned Parquet "lake" for the large NBA tables (play-by-play, box scores, odds, game index).

Layout, under ``<NBA_DATA_DIR>/lake`` (``NBA_DATA_DIR`` defaults to this repo's ``data/``, like every other tool)::

    lake/play_by_play/season_year=2019-20/part-<uuid>-0.parquet      # partitioned tables: one dir per season
    lake/player_boxscores/season_year=2019-20/...
    lake/team_boxscores/season_year=2019-20/...
    lake/game_odds/part-<uuid>-0.parquet                             # everything else: one flat dataset
    lake/games_index/...

Rules this module enforces:

* ``game_id`` is always a string (``0020000001``). An integer ``game_id`` means the leading zeros are already gone, so
  ``write_table`` refuses it instead of storing something unrecoverable.
* ``season_year`` is the text label (``2019-20``), stored as a hive partition column, and is always read back as a string.
* Parquet is written with zstd compression. Appends add new uniquely-named files, so converting in chunks never
  rewrites earlier chunks.

Only ``pandas`` and ``pyarrow`` (both in requirements.txt) are needed. ``duck()`` imports ``duckdb`` lazily, and
nothing here imports PySpark: install them only where you want them (see requirements-bigdata.txt).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import uuid
import warnings
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]

PARTITION_COL = "season_year"
# Tables big enough to be worth one directory per season. Everything else is stored flat.
PARTITIONED = frozenset({"play_by_play", "player_boxscores", "team_boxscores"})
COMPRESSION = "zstd"

_HIVE = ds.partitioning(pa.schema([(PARTITION_COL, pa.string())]), flavor="hive")


def lake_root() -> Path:
    """``$NBA_DATA_DIR/lake`` (default ``<repo>/data/lake``)."""
    return Path(os.environ.get("NBA_DATA_DIR", REPO / "data")).resolve() / "lake"


def table_path(table: str, root: Path | str | None = None) -> Path:
    if not table or Path(table).name != table or table in {".", ".."}:
        raise ValueError(f"invalid table name: {table!r}")
    return Path(root if root is not None else lake_root()) / table


def tables(root: Path | str | None = None) -> list[str]:
    """Names of the tables that currently hold at least one parquet file (hidden ``.``/``_`` dirs are skipped)."""
    base = Path(root if root is not None else lake_root())
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir()
                  if p.is_dir() and not p.name.startswith((".", "_")) and any(p.rglob("*.parquet")))


def _to_arrow(data) -> pa.Table:
    if isinstance(data, pa.Table):
        return data
    if isinstance(data, pd.DataFrame):
        return pa.Table.from_pandas(data, preserve_index=False)
    raise TypeError(f"write_table needs a pandas DataFrame or pyarrow Table, got {type(data).__name__}")


def _check_schema(table: str, tbl: pa.Table) -> None:
    names = tbl.schema.names
    if "game_id" in names and not (pa.types.is_string(tbl.schema.field("game_id").type)
                                   or pa.types.is_large_string(tbl.schema.field("game_id").type)):
        raise ValueError(
            f"{table}: game_id must be a string (got {tbl.schema.field('game_id').type}); an integer game_id has "
            "already lost its leading zeros. Read the source with game_id as text.")
    if table in PARTITIONED:
        if PARTITION_COL not in names:
            raise ValueError(f"{table} is partitioned by {PARTITION_COL}; the data has no such column")
        if tbl.column(PARTITION_COL).null_count:
            raise ValueError(f"{table}: {PARTITION_COL} has nulls; every row needs a season to be partitioned")


def _norm_type(t: pa.DataType) -> pa.DataType:
    return pa.string() if pa.types.is_large_string(t) else t


def _check_append(table: str, tbl: pa.Table, root) -> None:
    """Refuse an append whose columns or types differ from what the table already holds. Without this the dataset's
    schema comes from its first file, so a new column would be silently invisible on read."""
    path = table_path(table, root)
    if not path.is_dir() or not any(path.rglob("*.parquet")):
        return
    have = {f.name: _norm_type(f.type) for f in dataset(table, root).schema}
    new = {f.name: _norm_type(f.type) for f in tbl.schema}
    problems = []
    if set(have) - set(new):
        problems.append(f"missing columns {sorted(set(have) - set(new))}")
    if set(new) - set(have):
        problems.append(f"new columns {sorted(set(new) - set(have))}")
    problems += [f"{n}: table has {have[n]}, data has {new[n]}" for n in have if n in new and have[n] != new[n]]
    if problems:
        raise ValueError(f"cannot append to lake table {table!r}: " + "; ".join(problems)
                         + ". Use mode='overwrite' to replace the table with the new schema.")


def replace_dir(src: Path, dst: Path) -> None:
    """Move directory ``src`` to ``dst``, replacing ``dst`` only once ``src`` is complete (the old one is renamed aside
    first and restored if the move fails, so there is never a moment without a good copy)."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    old = None
    if dst.exists():
        old = dst.with_name(f".old-{dst.name}-{uuid.uuid4().hex[:8]}")
        dst.rename(old)
    try:
        src.rename(dst)
    except Exception:
        if old is not None:
            old.rename(dst)
        raise
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)


def _write_files(table: str, tbl: pa.Table, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ds.write_dataset(
        tbl, out, format="parquet",
        partitioning=_HIVE if table in PARTITIONED else None,
        file_options=ds.ParquetFileFormat().make_write_options(compression=COMPRESSION),
        basename_template=f"part-{uuid.uuid4().hex}-{{i}}.parquet",
        existing_data_behavior="overwrite_or_ignore",
    )


def write_table(table: str, data, *, mode: str = "overwrite", root: Path | str | None = None) -> Path:
    """Write ``data`` (DataFrame or Arrow table) as lake table ``table``.

    ``mode="overwrite"`` replaces whatever the lake already holds for this table (only that table's directory is
    touched): the new table is written to a hidden staging directory and swapped in, so a failure part-way leaves the
    old table intact. ``mode="append"`` adds files (how a large table is converted chunk by chunk) and refuses data
    whose columns or types differ from the table's. Returns the table directory.
    """
    if mode not in {"overwrite", "append"}:
        raise ValueError(f"mode must be 'overwrite' or 'append', got {mode!r}")
    tbl = _to_arrow(data)
    if PARTITION_COL in tbl.schema.names and table in PARTITIONED:
        # partition values must be text so "2019-20" never round-trips as a number
        tbl = tbl.set_column(tbl.schema.get_field_index(PARTITION_COL), PARTITION_COL,
                             tbl.column(PARTITION_COL).cast(pa.string()))
    _check_schema(table, tbl)

    out = table_path(table, root)
    if mode == "append":
        _check_append(table, tbl, root)
        _write_files(table, tbl, out)
        return out
    staging = out.parent / f".staging-{table}-{uuid.uuid4().hex[:8]}"
    try:
        _write_files(table, tbl, staging / table)
        replace_dir(staging / table, out)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return out


def dataset(table: str, root: Path | str | None = None) -> ds.Dataset:
    """The table as a pyarrow Dataset (lazy; nothing is loaded until you scan it)."""
    path = table_path(table, root)
    if not path.is_dir() or not any(path.rglob("*.parquet")):
        raise FileNotFoundError(f"no lake table {table!r} under {path.parent} (have: {tables(path.parent)})")
    return ds.dataset(path, format="parquet", partitioning=_HIVE if table in PARTITIONED else None)


def read_pandas(table: str, columns=None, filters=None, root: Path | str | None = None) -> pd.DataFrame:
    """Load (part of) a table as pandas. Always pass ``columns`` and/or ``filters`` for the big tables.

    ``filters`` uses the same form as ``pandas.read_parquet``: ``[("season_year", "==", "2019-20")]`` (AND), or a
    list of such lists (OR of ANDs). Partition filters skip whole season directories.
    """
    expr = pq.filters_to_expression(filters) if filters else None
    return dataset(table, root).to_table(columns=list(columns) if columns else None, filter=expr).to_pandas()


CONVERTER_REPORT = Path("_reports") / "sqlite_to_parquet.json"


def converter_report_path(root: Path | str | None = None) -> Path:
    return Path(root if root is not None else lake_root()) / CONVERTER_REPORT


def verification_status(table: str, root: Path | str | None = None) -> tuple[bool, str]:
    """(verified, reason). A table is verified when the last tools/sqlite_to_parquet.py run recorded a passing
    verification for it with no error, and no file in the table is newer than that report (so nothing was written
    to the table since)."""
    report_path = converter_report_path(root)
    try:
        entry = json.loads(report_path.read_text()).get("tables", {}).get(table)
    except (OSError, ValueError):
        return False, "no converter report; run tools/sqlite_to_parquet.py"
    if entry is None:
        return False, "the converter report has no entry for it"
    if "error" in entry:
        return False, "its last conversion failed"
    if "verify" not in entry:
        return False, "it was converted but not verified"
    if not entry["verify"].get("ok"):
        return False, "its verification failed"
    newest = max((f.stat().st_mtime for f in table_path(table, root).rglob("*.parquet")), default=0.0)
    if newest > report_path.stat().st_mtime:
        return False, "it was written to after it was verified"
    return True, ""


def prefer_lake(table: str, csv_exists: bool, root: Path | str | None = None) -> tuple[bool, str]:
    """Shared policy for tools that used to read a data/csv file: use the lake table only if it exists and is verified.
    Returns (use_lake, message to print). With no CSV to fall back to, an unverified lake is used with a warning."""
    if table not in tables(root):
        return False, ""
    ok, why = verification_status(table, root)
    if ok:
        return True, ""
    if not csv_exists:
        return True, f"warning: lake table {table!r} is not verified ({why}) and there is no CSV fallback; using the lake"
    return False, (f"note: not using lake table {table!r} ({why}); reading the CSV. "
                   "Run tools/sqlite_to_parquet.py to verify the lake")


def column_kind(arrow_type: pa.DataType) -> str:
    """``int`` | ``float`` | ``text``: how a lake column is summed and compared by the verification tools."""
    if pa.types.is_integer(arrow_type):
        return "int"
    if pa.types.is_floating(arrow_type):
        return "float"
    return "text"


def column_stats(dset: ds.Dataset, cols=None, group: int = 40) -> dict:
    """Stream per-column checksums of a lake dataset: ``{col: {kind, nn, empty, sum}}``.

    ``nn`` counts non-null values; ``empty`` counts empty strings (text only); ``sum`` is the column total (exact
    Python ``int`` for integer columns, a float for float columns) or, for text, the total character length.
    Columns are read ``group`` at a time so a 200-column table never sits in memory whole.
    """
    names = list(cols) if cols is not None else list(dset.schema.names)
    kinds = {n: column_kind(dset.schema.field(n).type) for n in names}
    out = {n: {"kind": kinds[n], "nn": 0, "empty": 0, "sum": 0.0 if kinds[n] == "float" else 0} for n in names}
    for i in range(0, len(names), group):
        part = names[i:i + group]
        for batch in dset.to_batches(columns=part):
            for n in part:
                arr = batch.column(batch.schema.get_field_index(n))
                st = out[n]
                st["nn"] += len(arr) - arr.null_count
                if st["kind"] == "text":
                    st["empty"] += pc.sum(pc.cast(pc.fill_null(pc.equal(arr, ""), False), pa.int64())).as_py() or 0
                    st["sum"] += pc.sum(pc.utf8_length(arr)).as_py() or 0
                else:
                    st["sum"] += pc.sum(arr).as_py() or 0
    return out


def duck(root: Path | str | None = None):
    """An in-memory DuckDB connection with one view per lake table, e.g.
    ``duck().execute("SELECT season_year, count(*) FROM play_by_play GROUP BY 1").df()``.

    DuckDB is imported here, not at module import, so the rest of the module (and CI) works without it.
    """
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - message only
        raise ImportError("duck() needs DuckDB: pip install -r requirements-bigdata.txt (or pip install duckdb)") from exc
    base = Path(root if root is not None else lake_root())
    con = duckdb.connect()
    for name in tables(base):
        glob = (base / name / "**" / "*.parquet").as_posix().replace("'", "''")
        hive = ", hive_partitioning = true, hive_types = {'season_year': VARCHAR}" if name in PARTITIONED else ""
        con.execute(f"CREATE VIEW \"{name}\" AS SELECT * FROM read_parquet('{glob}'{hive})")
    return con


def _java_available() -> bool:
    home = os.environ.get("JAVA_HOME")
    if home:  # Spark's launcher uses JAVA_HOME first, so a stale value breaks it even if java is on PATH
        return any((Path(home) / "bin" / name).is_file() for name in ("java", "java.exe"))
    return shutil.which("java") is not None


def _windows_hadoop_missing() -> bool:
    """True on native Windows when Hadoop's winutils.exe is not under $HADOOP_HOME/bin.

    Spark reads local files through Hadoop, and on Windows Hadoop needs winutils.exe + hadoop.dll; pip does not install
    them, and without them even a plain ``spark.read.parquet`` fails with an UnsatisfiedLinkError (NativeIO$Windows)."""
    if sys.platform != "win32":
        return False
    home = os.environ.get("HADOOP_HOME") or os.environ.get("hadoop.home.dir")
    return not (home and (Path(home) / "bin" / "winutils.exe").is_file())


def get_spark(app: str = "nba-lake", memory: str | None = None, cores: int | None = None):
    """A local-mode SparkSession for reading the lake (PySpark is optional: pip install -r requirements-bigdata.txt).

    Needs Java 17 or 21 on PATH or in JAVA_HOME. Memory and cores default to ``$NBA_SPARK_MEMORY`` (``4g``) and
    ``$NBA_SPARK_CORES`` (4) so a laptop is not starved. If a session is already running it is returned as is (Spark
    cannot resize it), with a warning when you asked for different memory or cores. READ ONLY by convention: all lake
    writes go through ``write_table`` (pyarrow), because Spark's local writes can need Hadoop binaries on Windows.
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:  # pragma: no cover - message only
        raise ImportError("get_spark() needs PySpark: pip install -r requirements-bigdata.txt") from exc
    if not _java_available():
        raise RuntimeError("get_spark() needs Java 17 or 21: install it and put java on PATH, or point JAVA_HOME at a "
                           "JDK that has bin/java (JAVA_HOME is " + repr(os.environ.get("JAVA_HOME")) + ")")
    if _windows_hadoop_missing():
        warnings.warn("Spark on native Windows needs Hadoop's winutils.exe and hadoop.dll (set HADOOP_HOME to a folder "
                      "whose bin\\ holds them); without them reading the lake fails with an UnsatisfiedLinkError. "
                      "DuckDB (datalake.duck()) needs none of this, or run Spark under WSL2. See data/README.md.",
                      stacklevel=2)
    asked = memory is not None or cores is not None
    memory = memory or os.environ.get("NBA_SPARK_MEMORY", "4g")
    cores = cores or int(os.environ.get("NBA_SPARK_CORES", "4"))
    active = SparkSession.getActiveSession()
    if active is not None:
        conf = active.sparkContext.getConf()
        if asked and (conf.get("spark.driver.memory", None) != memory
                      or active.sparkContext.master != f"local[{cores}]"):
            warnings.warn(f"a Spark session is already running ({active.sparkContext.master}, driver memory "
                          f"{conf.get('spark.driver.memory', 'default')}); requested memory={memory} cores={cores} "
                          "were NOT applied. Stop the session first to change them.", stacklevel=2)
        return active
    # workers must use the same interpreter as the driver, or a venv/Windows install picks a different Python
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    return (SparkSession.builder.master(f"local[{cores}]").appName(app)
            .config("spark.driver.memory", memory)
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", str(max(2, cores * 2)))
            .config("spark.sql.session.timeZone", "UTC")
            # keep partition values as text ("2019-20"), never let Spark guess a number
            .config("spark.sql.sources.partitionColumnTypeInference.enabled", "false")
            .getOrCreate())


def table_uri(table: str, root: Path | str | None = None) -> str:
    """``file:///...`` URI of a lake table directory; works for relative roots (``Path.as_uri`` alone does not)."""
    return table_path(table, root).resolve().as_uri()


def spark_read(spark, table: str, root: Path | str | None = None):
    """The lake table as a Spark DataFrame (``season_year`` is a string column for partitioned tables)."""
    dataset(table, root)  # same friendly FileNotFoundError as the other readers
    uri = table_uri(table, root)
    return spark.read.option("basePath", uri).parquet(uri)
