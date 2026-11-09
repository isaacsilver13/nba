# Pre-registration: does the lineup model add information beyond the closing spread?

Written 2026-10-22, **before this test was run**. Follows `docs/lineup-tracking-ablation-results.md`. The ablation
asked "does +lineup predict margins better than the baseline?"; this asks the question that matters for betting: "does the
extra information in +lineup predict what the **closing line misses**?" It uses the out-of-sample predictions the
ablation already saved (`outputs/ablation/lineup_tracking.csv`), so nothing is refit and nothing is tuned.
`tools/lineup_market_encompass.py` records this file's SHA-256 and refuses to run unless the marker below is present.

Frozen: yes

## Model
For each out-of-sample game with a closing spread, let `market = -spread_signed` (the line's expected home margin),
`E = pred_baseline - market` (the baseline's disagreement with the market) and `D = pred_variant - pred_baseline` (what the
variant adds). Regress the line's error on both:

    (home_margin - market) = a + b*E + g*D + error,    standard errors clustered by game date.

`g` is the primary quantity. `g > 0` means the variant's extra information points toward where the result lands
relative to the close; `g = 1` means its increment is worth full weight; `g = 0` means it adds nothing beyond the market.
The baseline's own `b` is reported as context (the earlier finding is that the model adds ~nothing beyond the close).

## Primary test
Variant **+lineup**, games from seasons **2022-2025** (the out-of-sample seasons through 2024-25) pooled.
**Supported** only if `g > 0` with one-sided p < 0.05 **and** the 2025-26 season (descriptive replication, same
regression) also has `g > 0`. Anything else is **not supported**: the lineup variant is not a betting signal.

## Always reported (not used for the decision)
`g`, its standard error and p for: each season separately; +tracking and +both (same regression with their own `D`);
the standard deviation of `D` (how much the variant moves the prediction at all); the number of games. A large standard
error relative to `g` is reported as low power, not as evidence of absence.

## Not allowed after seeing results
Changing the variant, seasons, controls, clustering or threshold and re-running on this data. Betting rules, EV
thresholds and bankroll are out of scope here.
