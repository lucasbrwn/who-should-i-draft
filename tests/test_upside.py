"""Season floor / ceiling / breakout (SCRUM-52 season half, SCRUM-94 #2)."""

import numpy as np
import polars as pl

from src.models.upside import UpsideModel, add_breakout_label, usable


def table(seasons=(2021, 2022, 2023), n=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for i in range(n):
            pos = ["QB", "RB", "WR", "TE"][i % 4]
            a = rng.uniform(2, 20)
            rows.append({"player_id": f"p{i}", "player_display_name": "x", "team": "T", "season": s, "position": pos,
                         "a_ppg": a, "a_exp_games": 15.0, "a_season": a * 15, "naive": a * 17,
                         "actual": max(0.0, a * 15 + rng.normal(0, 40))})
    return add_breakout_label(pl.DataFrame(rows))


def test_breakout_label_is_never_an_input():
    t = table()
    assert "breakout" not in usable(t) and "actual" not in usable(t)
    m = UpsideModel().fit(t.filter(pl.col("season") < 2023))
    assert "breakout" not in m.cols


def test_ranges_are_ordered_and_widening_only_adds():
    t = table()
    m = UpsideModel().fit(t.filter(pl.col("season") < 2023))
    p = m.predict(t.filter(pl.col("season") == 2023))
    assert (p["floor"] <= p["ceiling"]).all() and (p["floor"] >= 0).all()
    assert all(lo >= -1e9 for lo, _ in m.widen.values())
    assert p["breakout_p"].is_between(0, 1).all()


def test_one_training_season_means_no_conformal_widening():
    t = table(seasons=(2021, 2022))
    m = UpsideModel().fit(t.filter(pl.col("season") == 2021))
    assert m.widen == {}


def test_breakout_is_top_n_at_position_each_season():
    t = table()
    counts = t.filter(pl.col("breakout") == 1).group_by("season", "position").len()
    assert set(counts.filter(pl.col("position") == "QB")["len"].to_list()) == {12}
    assert set(counts.filter(pl.col("position") == "RB")["len"].to_list()) == {24}
