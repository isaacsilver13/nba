"""Curve math for tools/curve_calibration.py on synthetic data (no local data needed)."""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import curve_calibration as cc  # noqa: E402


def test_margin_pmf_sums_to_one_has_no_tie_and_matches_normal_rungs():
    line, sd = np.array([5.5, 0.0, -3.0]), np.array([12.0, 12.0, 12.0])
    xs, p = cc.pmf(line, sd, norm.cdf, no_tie=True)
    assert np.allclose(p.sum(1), 1)
    assert np.all(p[xs == 0] == 0)
    # home -5.5: the removed tie mass sits below the line, so P(M > 5.5) rises a little above 0.5;
    # pick'em P(M > 0) is exactly 0.5 after removing ties
    assert 0.5 < cc.rung_probs(xs, p, line, 0)[0] < 0.52
    assert abs(cc.rung_probs(xs, p, line, 0)[1] - 0.5) < 1e-9


def test_integer_rung_conditions_on_no_push():
    xs, p = cc.pmf(np.array([200.0]), np.array([18.0]), norm.cdf)
    over, push = p[0, xs[0] > 202].sum(), p[0, xs[0] == 202].sum()
    assert np.isclose(cc.rung_probs(xs, p, np.array([200.0]), 2)[0], over / (1 - push))


def test_scale_fit_recovers_width_that_grows_with_total():
    rng = np.random.default_rng(1)
    import pandas as pd
    df = pd.DataFrame({"line_t": rng.uniform(190, 240, 20000), "line_m": rng.uniform(-12, 12, 20000)})
    true = np.array([np.log(12.0), 0.05, 0.03])
    resid = rng.normal(0, np.exp(cc.design(df) @ true))
    assert np.allclose(cc.fit_scale(resid, cc.design(df)), true, atol=0.02)


def test_rps_is_zero_for_a_certain_correct_forecast():
    xs = np.arange(-2, 3)[None, :]
    p = np.array([[0, 0, 1.0, 0, 0]])
    assert cc.rps(xs, p, np.array([0.0]))[0] == 0
