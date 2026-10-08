"""Data screen for outcome leakage in the model features (complements tools/pit_check.py).

For every feature, fit a small walk-forward LightGBM of abs(margin) on [abs(line), feature] and report the extra
out-of-fold R^2 over the line alone. A feature that knows how the game was played (blowout vs close) scores far
above the rest. Exit code 1 if any feature exceeds median + 4 MAD of the scores.

    python tools/leak_screen.py

It only detects leaks tied to the game's outcome. Same-game facts unrelated to the result, and look-ahead into the
future, are caught by tools/pit_check.py instead. Positive control: re-enable the old minutes_lost_last_game etc.
and they score 0.11 to 0.14 against a cutoff near 0.005.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import experiment as e

df = e.load_data()
feats = e.build_features(df)
for c in feats + ["home_margin", "spread_kaggle"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
d = df.dropna(subset=["spread_kaggle", "home_margin"]).sort_values("GAME_DATE", kind="stable").reset_index(drop=True)
d["absl"] = d["spread_kaggle"].abs()
d["absm"] = d["home_margin"].abs()

P = dict(n_estimators=150, learning_rate=0.05, max_depth=3, num_leaves=8, min_child_samples=60, subsample=0.8,
         subsample_freq=1, random_state=1, verbose=-1, n_jobs=1)
T, V = 2000, 400
folds = [(a, a + T, a + T + V) for a in range(0, len(d) - T - V + 1, V)]


def oof(cols):
    out = np.full(len(d), np.nan)
    for a, b, c in folds:
        t, v = d.iloc[a:b], d.iloc[b:c]
        if t[cols].notna().all(axis=1).sum() < 200:
            return None
        out[b:c] = lgb.LGBMRegressor(**P).fit(t[cols], t["absm"]).predict(v[cols])
    return out


def r2(pred, y):
    m = ~np.isnan(pred)
    return 1 - ((y[m] - pred[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()


base = r2(oof(["absl"]), d["absm"].values)


def score(f):
    p = oof(["absl", f])
    return None if p is None else (f, r2(p, d["absm"].values) - base)


rows = [r for r in Parallel(n_jobs=8)(delayed(score)(f) for f in feats) if r]
R = pd.DataFrame(rows, columns=["feature", "d_r2_absmargin"]).sort_values("d_r2_absmargin", ascending=False)
med = R["d_r2_absmargin"].median()
cut = med + 4 * 1.4826 * (R["d_r2_absmargin"] - med).abs().median()
print(R.head(15).round(4).to_string(index=False))
print(f"\nfeatures screened {len(R)}; noise cutoff {cut:.4f}")
flagged = R[R["d_r2_absmargin"] > cut]
if len(flagged):
    print("FLAGGED:", flagged["feature"].tolist())
    sys.exit(1)
print("PASS: no feature knows the game's margin beyond the line")
