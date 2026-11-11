"""Gradient boosting model (SCRUM-49). Rules agreed 2026-11-10."""

import numpy as np
import polars as pl

from src.models.boosting import BoostingModel
from test_ridge import make_feats

FAST = {"max_iter": 30, "min_samples_leaf": 5}


def test_learns_a_simple_relationship():
    feats = make_feats()
    model = BoostingModel(params=FAST).fit(feats.filter(pl.col("season") < 2022))
    test = feats.filter((pl.col("season") == 2022) & (pl.col("week") > 1))
    pred = model.predict(test)["receptions"].to_numpy()
    assert np.corrcoef(pred, test["y_receptions"].to_numpy())[0, 1] > 0.9


def test_learns_an_interaction_ridge_cannot_add_up():
    # receptions = 8 only when BOTH flags are on: a pure "and" that a sum of weights can't express.
    rng = np.random.default_rng(1)
    feats = make_feats(seasons=(2020, 2021)).with_columns(
        a=pl.Series(rng.random(17 * 30 * 2) < 0.5).cast(pl.Float64),
        b=pl.Series(rng.random(17 * 30 * 2) < 0.5).cast(pl.Float64),
    ).with_columns(y_receptions=pl.col("a") * pl.col("b") * 8.0)
    model = BoostingModel(params=FAST).fit(feats.filter(pl.col("season") == 2020))
    test = feats.filter(pl.col("season") == 2021)
    pred = model.predict(test)["receptions"].to_numpy()
    both = (test["a"] * test["b"]).to_numpy() == 1
    assert pred[both].mean() > 6 and pred[~both].mean() < 2


def test_counts_use_poisson_and_yards_squared_error():
    feats = make_feats(seasons=(2020,)).with_columns(y_receiving_yards=pl.col("y_receptions") * 11)
    model = BoostingModel(params=FAST).fit(feats)
    assert model.models[("WR", "receptions")].loss == "poisson"
    assert model.models[("WR", "receiving_yards")].loss == "squared_error"


def test_refits_every_4_weeks_and_each_new_season():
    feats = make_feats(seasons=(2020, 2021))
    model = BoostingModel(params=FAST)
    history = lambda s, w: feats.filter((pl.col("season") < s) | ((pl.col("season") == s) & (pl.col("week") < w)))  # noqa: E731
    model.fit(history(2021, 2))
    first = model.models[("WR", "receptions")]
    model.fit(history(2021, 4))
    assert model.models[("WR", "receptions")] is first          # 2 weeks later: same model
    model.fit(history(2021, 6))
    assert model.models[("WR", "receptions")] is not first      # 4 weeks later: retrained


def test_unseen_injury_status_does_not_crash():
    feats = make_feats(seasons=(2020,))
    model = BoostingModel(params=FAST).fit(feats)
    new = feats.head(3).with_columns(report_status=pl.lit("Doubtful"))   # never seen in training
    assert model.predict(new).height == 3


def test_all_blank_column_for_a_position_is_dropped():
    feats = make_feats(seasons=(2020,)).with_columns(rb_only=pl.lit(None, dtype=pl.Float64))
    model = BoostingModel(params=FAST).fit(feats)
    assert "rb_only" not in model.columns["WR"]
