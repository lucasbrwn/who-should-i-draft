"""Weekly bust / ceiling ranges (SCRUM-52)."""

import numpy as np
import polars as pl
import pytest

from src.models.ranges import LIKELY, BucketRanges, QuantileRanges, add_volatility, pinball, score


@pytest.fixture
def backtest():
    """Fake backtest: actual = projection + noise (sd 6), floored at 0 like real scores; 2000 games per position."""
    rng = np.random.default_rng(0)
    n, positions = 2000, ["QB", "RB", "WR", "TE"]
    proj = rng.uniform(0, 25, n * len(positions))
    return pl.DataFrame({"position": np.repeat(positions, n), "proj": proj,
                         "actual": np.clip(proj + rng.normal(0, 6, proj.size), 0, None),
                         "volatility": rng.uniform(3, 9, proj.size)})


@pytest.mark.parametrize("method", [BucketRanges, QuantileRanges])
def test_ranges_cover_about_80_percent_and_bracket_the_projection(backtest, method):
    out = score(method().fit(backtest).predict(backtest))
    assert 0.75 < out["inside"].mean() < 0.85
    assert (out["bust"] >= 0).all()
    assert (out["bust"] <= out["proj"]).all() and (out["ceiling"] >= out["proj"]).all()


def test_top_bucket_ceiling_grows_with_the_projection(backtest):
    """The v1 bug: a fixed top-bucket ceiling gave a 26-point projection a 26.3 ceiling."""
    rows = pl.DataFrame({"position": ["RB"] * 3, "proj": [22.0, 30.0, 40.0]})
    out = BucketRanges().fit(backtest).predict(rows)
    ceilings = out["ceiling"].to_list()
    assert ceilings[0] < ceilings[1] < ceilings[2]
    assert all(c - p > 5 for c, p in zip(ceilings, rows["proj"]))


def test_volatility_uses_only_earlier_games_and_skips_left_injured():
    pw = pl.DataFrame({
        "player_id": ["a"] * 6, "season": [2024] * 6, "week": [1, 2, 3, 4, 5, 6],
        "fantasy_points_ppr": [10.0, 20.0, 10.0, 0.5, 20.0, 999.0],
        "left_injured": [False, False, False, True, False, False],
    })
    vol = add_volatility(pw).sort("week")
    assert vol["week"].to_list() == [1, 2, 3, 5, 6]          # the left-injured game is gone
    assert vol["volatility"][:3].is_null().all()              # fewer than 4 earlier games
    assert vol["volatility"][4] == pytest.approx(np.std([10, 20, 10, 20], ddof=1))  # week 6 ignores its own 999


def test_pinball_loss_charges_misses_by_side():
    assert pinball(np.array([10.0]), np.array([0.0]), 0.9)[0] == pytest.approx(9.0)   # ceiling too low: costly
    assert pinball(np.array([0.0]), np.array([10.0]), 0.9)[0] == pytest.approx(1.0)   # ceiling too high: cheap


def test_likely_range_holds_about_half_of_outcomes_and_is_narrower(backtest):
    wide = score(BucketRanges().fit(backtest).predict(backtest))
    likely = score(BucketRanges(LIKELY).fit(backtest).predict(backtest))
    # judged on projections >= 10 (like the real 2024 check): low projections pile up at 0 points,
    # and a 0 that equals a 0 bust counts as "inside", inflating coverage
    relevant = likely.filter(pl.col("proj") >= 10)
    assert 0.45 < relevant["inside"].mean() < 0.55
    assert likely["width"].mean() < 0.65 * wide["width"].mean()
