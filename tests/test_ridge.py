"""Ridge model (SCRUM-41). Rules agreed 2026-11-10."""

import numpy as np
import polars as pl
import pytest

from src.features.assemble import TARGETS
from src.models.ridge import DEFAULT_ALPHA, RidgeModel


def make_feats(seasons=(2020, 2021, 2022), weeks=range(1, 18), players=30, seed=0):
    """Synthetic WR data where receptions = 0.5 * targets_l3 + noise, plus a text column and blanks."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for w in weeks:
            for p in range(players):
                tl3 = rng.uniform(0, 10)
                row = {"player_id": f"p{p}", "player_display_name": f"p{p}", "position": "WR", "season": s, "week": w,
                       "game_id": f"{s}_{w}", "team": "T", "opponent_team": "O", "left_injured": False,
                       "targets_l3": tl3 if w > 1 else None,             # blank in week 1, like the real table
                       "report_status": "Questionable" if p % 5 == 0 else None}
                row.update({f"y_{t}": 0.0 for t in TARGETS})
                row["y_receptions"] = max(0.0, 0.5 * (tl3 if w > 1 else 5) + rng.normal(0, 0.3))
                rows.append(row)
    return pl.DataFrame(rows, schema_overrides={"report_status": pl.String, "targets_l3": pl.Float64})


def test_learns_a_simple_relationship():
    feats = make_feats()
    model = RidgeModel().fit(feats.filter(pl.col("season") < 2022))
    test = feats.filter((pl.col("season") == 2022) & (pl.col("week") > 1))
    pred = model.predict(test)["receptions"].to_numpy()
    assert np.corrcoef(pred, test["y_receptions"].to_numpy())[0, 1] > 0.9


def test_predictions_are_never_negative():
    feats = make_feats().with_columns(pl.lit(0.0).alias("y_receiving_tds"))
    model = RidgeModel().fit(feats.filter(pl.col("season") < 2022))
    preds = model.predict(feats.filter(pl.col("season") == 2022))
    assert (preds["receiving_tds"] >= 0).all() and (preds["receptions"] >= 0).all()


def test_alpha_tuned_on_earlier_seasons_only():
    feats = make_feats()
    model = RidgeModel().fit(feats.filter(pl.col("season") < 2022))   # validates on 2021, trains on 2020
    assert set(k[0] for k in model.alpha_cache) == {2021}


def test_default_alpha_when_nothing_earlier_to_tune_on():
    model = RidgeModel().fit(make_feats(seasons=(2020,)))
    assert model.alphas[("WR", "receptions")] == DEFAULT_ALPHA


def test_rare_stats_come_from_the_baseline():
    feats = make_feats().with_columns(pl.lit(1.0).alias("y_fumbles_lost_total"))
    model = RidgeModel().fit(feats.filter(pl.col("season") < 2022))
    preds = model.predict(feats.filter(pl.col("season") == 2022).head(5))
    assert preds["fumbles_lost_total"].to_list() == pytest.approx([1.0] * 5)


def test_left_injured_games_are_not_trained_on():
    feats = make_feats(seasons=(2020,)).with_columns(
        left_injured=pl.col("week") == 5,
        y_receptions=pl.when(pl.col("week") == 5).then(1000.0).otherwise(pl.col("y_receptions")),
    )
    model = RidgeModel().fit(feats)
    preds = model.predict(feats.filter(pl.col("week") == 6))
    assert preds["receptions"].max() < 20                             # the 1000-catch games were skipped


def test_coefficients_name_the_features():
    model = RidgeModel().fit(make_feats(seasons=(2020,)))
    coefs = model.coefficients().filter((pl.col("position") == "WR") & (pl.col("stat") == "receptions"))
    assert coefs["feature"][0] == "targets_l3"                       # the one real signal ranks first
