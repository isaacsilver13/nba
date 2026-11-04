# Results: lineup and tracking features vs the daily margin model

Run 2026-10-22 under `docs/lineup-tracking-ablation-preregistration.md` (frozen before any result existed; its SHA-256 and
mtime are in the result JSON). Full output: `outputs/ablation/lineup_tracking.json` and `.csv` (not committed).
Same 5,612 out-of-sample games, folds, models and seeds for all four variants.

## Home-margin RMSE (lower is better)
| Variant | Dev 2022-24 (3,149) | Holdout 2024-25 (1,230) | 2025-26 (1,230, descriptive) | Oct-Nov, dev+holdout |
|---|---|---|---|---|
| baseline | 14.035 | 14.218 | 14.504 | 13.362 |
| +lineup | 13.997 | 14.097 | 14.499 | 13.351 |
| +tracking | 13.993 | 14.294 | 14.519 | 13.341 |
| +both | 13.991 | 14.155 | 14.524 | 13.285 |

Holdout improvement over baseline (RMSE points, 90% date-block bootstrap interval):
+lineup **+0.121** [+0.054, +0.190]; +tracking **-0.076** [-0.154, +0.004]; +both **+0.063** [-0.032, +0.159].
Holdout Brier: baseline 0.2143, +lineup 0.2107, +tracking 0.2145, +both 0.2112.

## Verdicts under the registered rules
- **+both (the registered primary candidate): not supported.** Rule 1 fails: the point gain (0.063) clears 0.05 but the
  interval's lower end is below 0. Rules 2-4 pass. Keep the baseline.
- **+lineup (secondary): passes all four rules.** Caveats the rules do not capture: it is one of three secondary
  comparisons (no multiplicity correction was registered), the gain appears in one holdout season, the development
  gain is only 0.04, and the descriptive 2025-26 check shows none (14.499 vs 14.504).
- **+tracking (secondary): not supported.** Slightly worse on the holdout; tracking is also empty for 2025-26.

## Reading
A weak, unreplicated lead for the lineup features, nothing for tracking, and no reason to change the daily model.
The next test of +lineup has to be a new pre-registration on data not used here (2026-27 games as they arrive, which first
needs 2026-27 play-by-play in the stint step). Do not re-tune windows or shrinkage and re-run on this data.
