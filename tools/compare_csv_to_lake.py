"""Prove that each data/csv/<table>.csv holds nothing the Parquet lake lacks, before anything stops reading the CSVs.

    python tools/compare_csv_to_lake.py [--csv-dir data/csv] [--tables player_boxscores game_odds ...]

For every CSV (matched to a lake table by file name) it streams the CSV once and compares it with the lake table:
columns, row count, distinct game ids, per-column non-null counts, numeric column sums and text-length sums.
A CSV is IDENTICAL only if all of those agree; anything else is DIFFERENT and the reasons are listed, so you can
decide what the CSV holds that the lake does not. Exit status is 0 only when every CSV is IDENTICAL.

Read-only: it never writes to the CSVs or the lake (the JSON report goes to <lake>/_reports/).

CSV quirks it allows for: empty fields are NULL (so '' in the lake counts as NULL on both sides), and a ``game_id``
that lost its leading zeros is zero-padded to 10 digits before comparing (and counted in the report).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.ingest import datalake  # noqa: E402


def default_csv_dir() -> Path:
    return Path(os.environ.get("NBA_DATA_DIR", REPO / "data")) / "csv"


@dataclass
class Comparison:
    csv: str
    table: str | None
    identical: bool = True
    csv_rows: int = 0
    lake_rows: int = 0
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def differ(self, msg: str) -> None:
        self.identical = False
        self.reasons.append(msg)


def _kind(t: pa.DataType) -> str:
    return "int" if pa.types.is_integer(t) else "float" if pa.types.is_floating(t) else "text"


def _lake_stats(dset, cols: list[str], kinds: dict[str, str]) -> tuple[dict, set]:
    stats = {c: {"nn": 0, "sum": 0.0} for c in cols}
    for batch in dset.to_batches(columns=cols):
        for c in cols:
            arr = batch.column(batch.schema.get_field_index(c))
            if kinds[c] == "text":
                empty = pc.sum(pc.cast(pc.fill_null(pc.equal(arr, ""), False), pa.int64())).as_py() or 0
                stats[c]["nn"] += len(arr) - arr.null_count - empty
                stats[c]["sum"] += pc.sum(pc.utf8_length(arr)).as_py() or 0
            else:
                stats[c]["nn"] += len(arr) - arr.null_count
                stats[c]["sum"] += pc.sum(arr).as_py() or 0
    ids: set = set()
    if "game_id" in dset.schema.names:
        for batch in dset.to_batches(columns=["game_id"]):
            ids.update(pc.unique(batch.column(0)).drop_null().to_pylist())
    return stats, ids


def _csv_stats(path: Path, cols: list[str], kinds: dict[str, str], per_chunk: int):
    stats = {c: {"nn": 0, "sum": 0.0, "bad": 0} for c in cols}
    ids: set = set()
    rows = padded = 0
    reader = pd.read_csv(path, usecols=cols, dtype=str, keep_default_na=False, na_values=[""], chunksize=per_chunk,
                         low_memory=False)
    for chunk in reader:
        rows += len(chunk)
        for c in cols:
            s = chunk[c]
            if c == "game_id":
                padded += int((s.notna() & (s.str.len() < 10)).sum())
                s = s.str.zfill(10)
                ids.update(s.dropna().unique())
            if kinds[c] == "text":
                stats[c]["nn"] += int(s.notna().sum())
                stats[c]["sum"] += int(s.str.len().sum())
            else:
                n = pd.to_numeric(s, errors="coerce")
                stats[c]["bad"] += int((s.notna() & n.isna()).sum())
                stats[c]["nn"] += int(n.notna().sum())
                stats[c]["sum"] += float(n.sum())
    return stats, ids, rows, padded


def compare_one(csv_path: Path, root=None, rel_tol: float = 1e-9, chunk_rows: int = 200_000,
                max_cells: int = 10_000_000, log=print) -> Comparison:
    table = csv_path.stem
    cmp = Comparison(csv=csv_path.name, table=table)
    if table not in datalake.tables(root):
        cmp.table = None
        cmp.differ(f"no lake table named {table!r}: this CSV's contents are not in the lake at all")
        return cmp
    dset = datalake.dataset(table, root)
    lake_cols = dset.schema.names
    csv_cols = list(pd.read_csv(csv_path, nrows=0).columns)
    cmp.lake_rows = dset.count_rows()

    extra = [c for c in csv_cols if c not in lake_cols]
    if extra:
        cmp.differ(f"CSV has columns the lake lacks: {extra}")
    lake_only = [c for c in lake_cols if c not in csv_cols]
    if lake_only:
        cmp.notes.append(f"lake has columns the CSV lacks (fine): {lake_only}")
    common = [c for c in csv_cols if c in lake_cols]
    kinds = {f.name: _kind(f.type) for f in dset.schema}

    per_chunk = max(1000, min(chunk_rows, max_cells // max(1, len(common))))
    log(f"  {csv_path.name}: streaming CSV ({len(common)} columns, {per_chunk:,} rows per chunk) ...")
    cstats, cids, cmp.csv_rows, padded = _csv_stats(csv_path, common, kinds, per_chunk)
    lstats, lids = _lake_stats(dset, common, kinds)

    if padded:
        cmp.notes.append(f"CSV game_id had lost its leading zeros on {padded:,} values (zero-padded to compare)")
    if cmp.csv_rows != cmp.lake_rows:
        cmp.differ(f"row count: csv={cmp.csv_rows:,} lake={cmp.lake_rows:,}"
                   + (" (CSV has rows the lake lacks)" if cmp.csv_rows > cmp.lake_rows
                      else " (CSV may be a subset, but only equal counts prove it)"))
    if cids != lids:
        cmp.differ(f"distinct game_id: csv-only={len(cids - lids):,} lake-only={len(lids - cids):,}")
    for c in common:
        a, b = cstats[c], lstats[c]
        if a.get("bad"):
            cmp.differ(f"{c}: {a['bad']:,} CSV values are not numbers but the lake holds numbers")
        if a["nn"] != b["nn"]:
            cmp.differ(f"{c}: non-null count csv={a['nn']:,} lake={b['nn']:,}")
        same = (math.isclose(a["sum"], float(b["sum"]), rel_tol=rel_tol, abs_tol=1e-6) if kinds[c] == "float"
                else abs(a["sum"] - float(b["sum"])) < 0.5)
        if not same:
            what = "text length sum" if kinds[c] == "text" else "sum"
            cmp.differ(f"{c}: {what} csv={a['sum']!r} lake={b['sum']!r}")
    return cmp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-dir", type=Path, default=None, help="default $NBA_DATA_DIR/csv")
    ap.add_argument("--lake-root", type=Path, default=None, help="default $NBA_DATA_DIR/lake")
    ap.add_argument("--tables", nargs="+", default=None, help="CSV stems to check (default: every CSV in the folder)")
    ap.add_argument("--rel-tol", type=float, default=1e-9)
    ap.add_argument("--chunk-rows", type=int, default=200_000)
    ap.add_argument("--report", type=Path, default=None)
    a = ap.parse_args(argv)

    csv_dir = a.csv_dir or default_csv_dir()
    root = a.lake_root or datalake.lake_root()
    files = sorted(csv_dir.glob("*.csv"))
    if a.tables:
        files = [f for f in files if f.stem in a.tables]
    if not files:
        print(f"no CSV files to compare in {csv_dir}", file=sys.stderr)
        return 2
    results = []
    for f in files:
        print(f"comparing {f.name} ...", flush=True)
        r = compare_one(f, root=root, rel_tol=a.rel_tol, chunk_rows=a.chunk_rows)
        results.append(r)
        print(f"  {'IDENTICAL' if r.identical else 'DIFFERENT'}: csv {r.csv_rows:,} rows, lake {r.lake_rows:,} rows")
        for msg in r.reasons:
            print(f"    - {msg}")
        for msg in r.notes:
            print(f"    . {msg}")
    ok = all(r.identical for r in results)
    path = a.report or root / "_reports" / "csv_vs_lake.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ok": ok, "csv_dir": str(csv_dir), "results": [asdict(r) for r in results]}, indent=2))
    print(("ALL IDENTICAL" if ok else "DIFFERENCES FOUND - the CSVs are NOT safely redundant") + f"; report: {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
