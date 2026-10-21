---
description: Diagnose or modify one NBA DataPrep ingestion, cache, schema, or feature-engineering slice with focused validation.
---

Work on the user's stated NBA DataPrep issue as one narrow slice.

1. Identify the owning step in `src/ingest/dataPrep/`, its caller in `pipeline.py`, and the relevant input/output artifact before editing.
2. Inspect `config.py` and preserve the existing global defaults and per-step override structure. Preserve the target step's current `run` signature and package import style unless the request requires a contract change.
3. For cache problems, establish whether the issue is a cache miss, incomplete reconstruction, stale schema, corrupt artifact, or unnecessary API call. Prefer read-through behavior and existing cache helpers over a new cache format.
4. For schema or dtype failures, normalize at the boundary and report the before/after columns and dtypes that matter. Do not hide missing required columns with silent defaults.
5. Make one focused edit. Do not combine performance work, feature redesign, and unrelated cleanup.
6. Run the cheapest discriminating check first: an import or compile check, a targeted fixture, or a narrowly scoped dry-run. Expand validation only when that check passes.
7. Report changed files, the verified behavior, remaining data/API assumptions, and any pre-existing failure separately.
