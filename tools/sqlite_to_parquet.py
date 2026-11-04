"""Convert the NBA SQLite databases to the partitioned Parquet lake (src/ingest/datalake.py), and prove the copy is exact.

    python tools/sqlite_to_parquet.py                          # convert + verify every table, then check seasons
    python tools/sqlite_to_parquet.py --tables game_odds       # just some tables
    python tools/sqlite_to_parquet.py --verify-only            # re-verify an existing lake against the SQLite files

Sources (under ``$NBA_DATA_DIR/sql``, default ``data/sql``; opened read-only, never modified):
    nba_stats.sqlite      games_index, games_schedule, game_odds, team_boxscores, player_boxscores
    nba_stats_pbp.sqlite  play_by_play

How it stays safe:
* Streams the source in chunks (``--chunk-rows``, default 500,000, further capped by ``--max-cells`` so a 200-column
  table never holds more than ~20M Python cells at once).
* Column types are fixed up front from a full scan of ``typeof()``: integer-only -> int64, any reals -> float64,
  text -> string. A column that mixes text and numbers aborts the conversion and names the column.
* ``game_id`` must be a 10-digit string everywhere (leading zeros are never recovered from integers).
* ``play_by_play`` has no season column: ``season_year`` is derived from ``game_id[3:5]`` as the text label
  (``0022000001`` -> ``2020-21``, pivot 96 -> 1996-97) and then checked against ``games_index``.
* Writes go to a hidden staging directory and replace the lake table only after every chunk is written, so a failure
  never leaves a half-converted table. An existing lake table is only replaced with ``--overwrite``.
* ``verify`` compares row counts, null counts, numeric column sums and text-length sums against the SQLite source.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sqlite3
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.ingest import datalake  # noqa: E402

MAIN_DB = "nba_stats.sqlite"
PBP_DB = "nba_stats_pbp.sqlite"
# small -> large, so a problem shows up before the 18M-row table
TABLE_SOURCES = {
    "games_schedule": MAIN_DB,
    "games_index": MAIN_DB,
    "game_odds": MAIN_DB,
    "team_boxscores": MAIN_DB,
    "player_boxscores": MAIN_DB,
    "play_by_play": PBP_DB,
}

SEASON_PIVOT = 96  # NBA game ids carry the season's start year as 2 digits; the data starts in 1996-97
GAME_ID_RE = r"^[0-9]{10}$"
SCAN_GROUP = 100  # columns per typeof() scan query
VERIFY_GROUP = 40  # columns per verification pass
ARROW_TYPES = {"int": pa.int64(), "float": pa.float64(), "text": pa.string()}


class ConversionError(RuntimeError):
    pass


def default_sql_dir() -> Path:
    return Path(os.environ.get("NBA_DATA_DIR", REPO / "data")) / "sql"


def season_from_game_id(game_id: str) -> str:
    """``0022000001`` -> ``2020-21``; ``0029900001`` -> ``1999-00``; ``0020000001`` -> ``2000-01``."""
    if len(game_id) != 10 or not game_id.isdigit():
        raise ValueError(f"not a 10-digit game id: {game_id!r}")
    yy = int(game_id[3:5])
    start = (1900 if yy >= SEASON_PIVOT else 2000) + yy
    return f"{start}-{(start + 1) % 100:02d}"


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _connect_ro(path: Path) -> sqlite3.Connection:
    path = Path(path).resolve()
    if not path.is_file():
        raise ConversionError(f"SQLite file not found: {path}")
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)


@dataclass
class Column:
    name: str
    declared: str
    kind: str  # int | float | text


def scan_columns(conn: sqlite3.Connection, table: str) -> list[Column]:
    """Decide each column's Arrow type from the values actually stored (one scan, ``SCAN_GROUP`` columns at a time)."""
    info = conn.execute(f"PRAGMA table_info({_q(table)})").fetchall()
    if not info:
        raise ConversionError(f"table {table!r} not found in the SQLite file")
    names = [r[1] for r in info]
    declared = {r[1]: (r[2] or "").upper() for r in info}
    out: list[Column] = []
    for i in range(0, len(names), SCAN_GROUP):
        group = names[i:i + SCAN_GROUP]
        sel = ", ".join(
            f"sum(typeof({_q(c)})='integer'), sum(typeof({_q(c)})='real'), "
            f"sum(typeof({_q(c)})='text'), sum(typeof({_q(c)})='blob')" for c in group)
        row = conn.execute(f"SELECT {sel} FROM {_q(table)}").fetchone()
        for j, c in enumerate(group):
            n_int, n_real, n_text, n_blob = (v or 0 for v in row[4 * j:4 * j + 4])
            if n_blob:
                raise ConversionError(f"{table}.{c}: holds {n_blob} blob values, which the lake does not store")
            if n_text and (n_int or n_real):
                raise ConversionError(
                    f"{table}.{c}: mixes text ({n_text}) with numbers ({n_int} integer, {n_real} real); "
                    "fix or exclude this column before converting")
            if n_text:
                kind = "text"
            elif n_real:
                kind = "float"
            elif n_int:
                kind = "int"
            else:  # all NULL: fall back to the declared type
                kind = "text" if any(t in declared[c] for t in ("TEXT", "CHAR", "CLOB")) else "float"
            if c == "game_id" and kind != "text" and (n_int or n_real):
                raise ConversionError(f"{table}.game_id is stored as a number; its leading zeros are already lost")
            out.append(Column(c, declared[c], kind))
    return out


def derive_season_array(game_ids: pa.Array) -> pa.Array:
    """Vectorised ``season_from_game_id`` (one Python call per distinct game id)."""
    uniq = pc.unique(game_ids)
    labels = pa.array([season_from_game_id(g) for g in uniq.to_pylist()], pa.string())
    return labels.take(pc.index_in(game_ids, value_set=uniq))


def _check_game_ids(table: str, ids: pa.Array) -> None:
    ok = pc.fill_null(pc.match_substring_regex(ids, GAME_ID_RE), False)
    if not pc.all(ok).as_py():
        bad = pc.filter(ids, pc.invert(ok)).to_pylist()[:5]
        raise ConversionError(f"{table}.game_id has values that are not 10-digit strings, e.g. {bad}")


def _chunk_to_arrow(table: str, rows: list[tuple], cols: list[Column]) -> pa.Table:
    data = list(zip(*rows))
    arrays = [pa.array(data[i], type=ARROW_TYPES[c.kind]) for i, c in enumerate(cols)]
    tbl = pa.Table.from_arrays(arrays, names=[c.name for c in cols])
    if "game_id" in tbl.schema.names:
        _check_game_ids(table, tbl.column("game_id").combine_chunks())
    if table in datalake.PARTITIONED and datalake.PARTITION_COL not in tbl.schema.names:
        if "game_id" not in tbl.schema.names:
            raise ConversionError(f"{table}: has neither {datalake.PARTITION_COL} nor game_id to derive it from")
        tbl = tbl.append_column(datalake.PARTITION_COL, derive_season_array(tbl.column("game_id").combine_chunks()))
    return tbl


def _dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


@dataclass
class ConvertResult:
    table: str
    rows: int
    columns: int
    chunks: int
    chunk_rows: int
    seconds: float
    lake_bytes: int


def convert(sqlite_path, table: str, root=None, chunk_rows: int = 500_000, max_cells: int = 20_000_000,
            overwrite: bool = False, log=print) -> ConvertResult:
    """Copy one SQLite table into the lake, in chunks. See the module docstring for the safety rules."""
    t0 = time.time()
    root = Path(root) if root is not None else datalake.lake_root()
    final = datalake.table_path(table, root)
    if final.exists() and not overwrite:
        raise ConversionError(f"lake table {final} already exists; pass overwrite=True (--overwrite) to replace it")
    conn = _connect_ro(Path(sqlite_path))
    try:
        cols = scan_columns(conn, table)
        total = conn.execute(f"SELECT count(*) FROM {_q(table)}").fetchone()[0]
        if total == 0:
            raise ConversionError(f"{table}: source table is empty")
        per_chunk = max(1, min(int(chunk_rows), int(max_cells) // len(cols)))
        staging = root / f".staging-{table}-{uuid.uuid4().hex[:8]}"
        written = chunks = 0
        try:
            cur = conn.execute(f"SELECT * FROM {_q(table)}")
            while True:
                rows = cur.fetchmany(per_chunk)
                if not rows:
                    break
                datalake.write_table(table, _chunk_to_arrow(table, rows, cols), mode="append", root=staging)
                written += len(rows)
                chunks += 1
                log(f"  {table}: {written:,}/{total:,} rows")
            if written != total:
                raise ConversionError(f"{table}: read {written} rows but count(*) said {total}")
            datalake.replace_dir(staging / table, final)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    finally:
        conn.close()
    return ConvertResult(table, written, len(cols), chunks, per_chunk, round(time.time() - t0, 1), _dir_bytes(final))


@dataclass
class VerifyReport:
    table: str
    ok: bool = True
    checks: int = 0
    failures: list[str] = field(default_factory=list)
    rows: int = 0

    def fail(self, msg: str) -> None:
        self.ok = False
        self.failures.append(msg)


def _close(a, b, rel_tol: float) -> bool:
    return math.isclose(float(a or 0), float(b or 0), rel_tol=rel_tol, abs_tol=1e-6)


def verify(sqlite_path, table: str, root=None, rel_tol: float = 1e-9, log=print) -> VerifyReport:
    """Compare the lake table with its SQLite source: columns, row count, null counts, numeric sums, text-length sums."""
    rep = VerifyReport(table)
    conn = _connect_ro(Path(sqlite_path))
    try:
        dset = datalake.dataset(table, root)
        src_cols = [r[1] for r in conn.execute(f"PRAGMA table_info({_q(table)})")]
        lake_cols = dset.schema.names
        expected_extra = {datalake.PARTITION_COL} if (table in datalake.PARTITIONED
                                                      and datalake.PARTITION_COL not in src_cols) else set()
        if set(src_cols) - set(lake_cols):
            rep.fail(f"columns missing from the lake: {sorted(set(src_cols) - set(lake_cols))}")
        if set(lake_cols) - set(src_cols) != expected_extra:
            rep.fail(f"unexpected lake columns: {sorted(set(lake_cols) - set(src_cols) - expected_extra)}")
        rep.checks += 2

        src_rows = conn.execute(f"SELECT count(*) FROM {_q(table)}").fetchone()[0]
        rep.rows = dset.count_rows()
        rep.checks += 1
        if rep.rows != src_rows:
            rep.fail(f"row count: sqlite={src_rows:,} lake={rep.rows:,}")

        common = [c for c in src_cols if c in lake_cols]
        lstats = datalake.column_stats(dset, common, group=VERIFY_GROUP)
        for i in range(0, len(common), VERIFY_GROUP):
            group = common[i:i + VERIFY_GROUP]
            exprs = []
            for c in group:
                agg = f"sum(length({_q(c)}))" if lstats[c]["kind"] == "text" else f"sum({_q(c)})"
                exprs.append(f"count({_q(c)}), {agg}")
            srow = conn.execute(f"SELECT {', '.join(exprs)} FROM {_q(table)}").fetchone()
            for j, c in enumerate(group):
                s_nn, s_sum = srow[2 * j], srow[2 * j + 1]
                lake = lstats[c]
                rep.checks += 2
                if lake["nn"] != s_nn:
                    rep.fail(f"{table}.{c}: non-null count sqlite={s_nn:,} lake={lake['nn']:,} "
                             f"(null count sqlite={src_rows - s_nn:,} lake={rep.rows - lake['nn']:,})")
                if lake["kind"] == "float":
                    same = _close(s_sum, lake["sum"], rel_tol)
                else:
                    same = int(s_sum or 0) == int(lake["sum"])
                if not same:
                    what = "text length sum" if lake["kind"] == "text" else "sum"
                    rep.fail(f"{table}.{c}: {what} sqlite={s_sum!r} lake={lake['sum']!r}")
            log(f"  verify {table}: {min(i + VERIFY_GROUP, len(common))}/{len(common)} columns")

        if "game_id" in lake_cols:
            rep.checks += 1
            bad = 0
            for batch in dset.to_batches(columns=["game_id"]):
                ok = pc.fill_null(pc.match_substring_regex(batch.column(0), GAME_ID_RE), False)
                bad += len(ok) - pc.sum(pc.cast(ok, pa.int64())).as_py()
            if bad:
                rep.fail(f"{table}.game_id: {bad:,} values are not 10-digit strings")
        if expected_extra:
            rep.checks += 1
            nulls = sum(b.column(0).null_count for b in dset.to_batches(columns=[datalake.PARTITION_COL]))
            if nulls:
                rep.fail(f"{table}.{datalake.PARTITION_COL}: {nulls:,} null values")
    finally:
        conn.close()
    return rep


@dataclass
class SeasonReport:
    games_index_rows: int = 0
    games_index_mismatches: int = 0
    games_index_examples: list = field(default_factory=list)
    tables: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.games_index_mismatches == 0 and all(t["mismatches"] == 0 for t in self.tables.values())


def check_seasons(games_index_sqlite, root=None, tables=("play_by_play",), log=print) -> SeasonReport:
    """Check ``season_from_game_id`` against ``games_index.season_year``, and the lake's ``season_year`` partitions
    against ``games_index`` too (a game filed under the wrong season is a mismatch; a game missing from the index is
    reported as ``unmatched`` but is not a mismatch)."""
    rep = SeasonReport()
    conn = _connect_ro(Path(games_index_sqlite))
    try:
        truth = dict(conn.execute("SELECT game_id, season_year FROM games_index").fetchall())
    finally:
        conn.close()
    rep.games_index_rows = len(truth)
    for gid, season in truth.items():
        if season_from_game_id(gid) != season:
            rep.games_index_mismatches += 1
            if len(rep.games_index_examples) < 5:
                rep.games_index_examples.append([gid, season, season_from_game_id(gid)])
    for table in tables:
        try:
            dset = datalake.dataset(table, root)
        except FileNotFoundError:
            continue
        pairs = set()
        for batch in dset.to_batches(columns=["game_id", datalake.PARTITION_COL]):
            pairs.update(batch.to_pandas().drop_duplicates().itertuples(index=False, name=None))
        mism = [(g, s, truth[g]) for g, s in pairs if g in truth and truth[g] != s]
        unmatched = {g for g, _ in pairs if g not in truth}
        rep.tables[table] = {"games": len({g for g, _ in pairs}), "mismatches": len(mism),
                             "unmatched": len(unmatched), "examples": [list(m) for m in mism[:5]]}
        log(f"  seasons {table}: {len({g for g, _ in pairs}):,} games, {len(mism)} mismatches, "
            f"{len(unmatched)} not in games_index")
    return rep


def lake_size(root=None) -> dict:
    root = Path(root) if root is not None else datalake.lake_root()
    sizes = {t: _dir_bytes(root / t) for t in datalake.tables(root)}
    sizes["TOTAL"] = sum(sizes.values())
    return sizes


def _fmt(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sql-dir", type=Path, default=None, help="folder with the .sqlite files (default $NBA_DATA_DIR/sql)")
    ap.add_argument("--lake-root", type=Path, default=None, help="default $NBA_DATA_DIR/lake")
    ap.add_argument("--tables", nargs="+", choices=list(TABLE_SOURCES), default=list(TABLE_SOURCES))
    ap.add_argument("--chunk-rows", type=int, default=500_000)
    ap.add_argument("--max-cells", type=int, default=20_000_000, help="cap on rows x columns held in memory per chunk")
    ap.add_argument("--overwrite", action="store_true", help="replace lake tables that already exist")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--report", type=Path, default=None, help="JSON report path (default <lake>/_reports/...)")
    a = ap.parse_args(argv)

    sql_dir = a.sql_dir or default_sql_dir()
    root = a.lake_root or datalake.lake_root()
    report = {"sql_dir": str(sql_dir), "lake_root": str(root), "tables": {}}
    failed = False
    for table in a.tables:
        src = sql_dir / TABLE_SOURCES[table]
        entry = report["tables"].setdefault(table, {})
        try:
            if not a.verify_only:
                print(f"converting {table} from {src.name} ...", flush=True)
                res = convert(src, table, root=root, chunk_rows=a.chunk_rows, max_cells=a.max_cells,
                              overwrite=a.overwrite)
                entry["convert"] = asdict(res)
                print(f"  -> {res.rows:,} rows, {res.chunks} chunks of <= {res.chunk_rows:,}, {_fmt(res.lake_bytes)}, "
                      f"{res.seconds}s", flush=True)
            if not a.skip_verify:
                print(f"verifying {table} ...", flush=True)
                v = verify(src, table, root=root)
                entry["verify"] = asdict(v)
                print(f"  {'OK' if v.ok else 'FAILED'}: {v.checks} checks, {v.rows:,} rows", flush=True)
                for msg in v.failures:
                    print(f"    - {msg}")
                failed |= not v.ok
        except (ConversionError, FileNotFoundError) as exc:
            failed = True
            entry["error"] = str(exc)
            print(f"  ERROR {table}: {exc}", file=sys.stderr)

    if not a.skip_verify:
        gi = sql_dir / MAIN_DB
        if gi.is_file() and "play_by_play" in datalake.tables(root):
            print("checking derived season_year against games_index ...", flush=True)
            s = check_seasons(gi, root=root)
            report["seasons"] = asdict(s) | {"ok": s.ok}
            print(f"  games_index self-check: {s.games_index_mismatches} mismatches in {s.games_index_rows:,} games")
            failed |= not s.ok

    sizes = lake_size(root)
    report["lake_bytes"] = sizes
    print("lake size: " + ", ".join(f"{k} {_fmt(v)}" for k, v in sizes.items()))
    report["ok"] = not failed
    path = a.report or root / "_reports" / "sqlite_to_parquet.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(f"report: {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
