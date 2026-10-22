"""predict_today's daily refit equals the walk-forward fold it stands in for when the slate date starts that fold
(same training window, same ensemble weights). Synthetic data, no local datasets needed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import experiment as ex  # noqa: E402
import predict_today as pt  # noqa: E402


def test_slate_prediction_equals_its_walk_forward_fold(monkeypatch):
    monkeypatch.setattr(ex, "TRAIN_SIZE", 120)
    monkeypatch.setattr(ex, "TEST_SIZE", 40)
    monkeypatch.setattr(ex, "FOLD_WORKERS", 1)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"GAME_DATE": np.repeat(pd.date_range("2024-01-01", periods=80), 3)})
    df["x1"], df["x2"] = rng.normal(size=(2, len(df)))
    df["home_margin"] = 5 * df["x1"] + rng.normal(0, 10, len(df))
    D = [t["GAME_DATE"].min() for _, t in ex.walk_forward_splits(df, "GAME_DATE", 120, 40)][2]

    _, pred = pt.walk_forward(df[df["GAME_DATE"] < D], ["x1", "x2"], df[df["GAME_DATE"] == D])
    rows = df[df["GAME_DATE"] <= D]
    oos, _ = pt.walk_forward(rows, ["x1", "x2"])
    np.testing.assert_array_equal(pred, oos.loc[rows.index[rows["GAME_DATE"] == D]].to_numpy())
