---
description: Review an NBA model or betting output with leakage, join quality, metric, and risk diagnostics before proposing one change.
---

Review the user's named NBA model, output, or experiment slice using current files and artifacts.

1. Locate the producer script, selected input files, version or modification date, and any join-audit or manifest output. Do not infer the latest artifact from a filename alone.
2. Check required columns and row counts before calculating diagnostics. Identify unmatched or fallback odds rows, duplicate game keys, missing dates, and side/sign conventions.
3. Separate the conclusions for walk-forward RMSE, calibration or probability quality, market-match quality, ROI, drawdown, bet coverage, Kelly sizing, and daily exposure. Explain when metrics disagree.
4. Check for leakage risks: postgame fields, outcome-derived columns, future rows in rolling features, global preprocessing, and odds or market data unavailable at prediction time.
5. Compare against a reproducible baseline using the same folds and candidate rows. Do not promote a change because of one favorable fold or one aggregate number.
6. Recommend at most one next experiment or code change. Preserve frozen experiment boundaries and write down the exact configuration and expected discriminating metric.
7. If implementation is requested, edit only the owning slice and run its narrowest test or diagnostic before broader runs.
