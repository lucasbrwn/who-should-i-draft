"""Tier calibration: fixes a bend, never reorders players, doesn't cap the best projections."""

import numpy as np
import polars as pl

from src.models.weekly_calibration import WeeklyCalibration


def bent_history(n=4000, seed=0):
    """Projections that run low in the middle and high at the top (the pattern we found)."""
    rng = np.random.default_rng(seed)
    pred = rng.uniform(0, 25, n)
    truth = np.where(pred > 20, pred - 1.0, np.where(pred > 12, pred + 0.5, pred))
    return pl.DataFrame({"position": ["RB"] * n, "pred": pred, "actual": truth + rng.normal(0, 3, n)})


def test_corrects_the_bend():
    cal = WeeklyCalibration().fit(bent_history())
    out = cal.apply(np.array([16.0, 22.0]), np.array(["RB", "RB"]))
    assert out[0] > 16.0      # middle tier was under-projected -> raised
    assert out[1] < 22.0      # top tier was over-projected -> lowered


def test_never_reorders_players():
    cal = WeeklyCalibration().fit(bent_history())
    p = np.linspace(0, 30, 200)
    out = cal.apply(p, np.array(["RB"] * 200))
    assert np.all(np.diff(out) >= -1e-9)


def test_projections_above_training_range_keep_their_spread():
    cal = WeeklyCalibration().fit(bent_history())
    out = cal.apply(np.array([30.0, 40.0]), np.array(["RB", "RB"]))
    assert out[1] - out[0] == 10.0 or abs((out[1] - out[0]) - 10.0) < 1e-9   # shifted, not capped


def test_other_positions_untouched_when_not_fit():
    cal = WeeklyCalibration().fit(bent_history())
    out = cal.apply(np.array([10.0]), np.array(["K"]))
    assert out[0] == 10.0
