"""Decision accuracy vs consensus (SCRUM-55): pairing, scoring, bootstrap."""

import polars as pl
import pytest

from src.backtest.vs_consensus import bootstrap_rate, pairs, score_pairs
from src.ingest.consensus import normalize


def players(rows):
    return pl.DataFrame(rows, schema=["season", "player_id", "position", "rank", "season_b", "points"], orient="row")


def test_pairs_are_same_position_and_within_one_round():
    p = players([(2024, "a", "RB", 1.0, 0, 0), (2024, "b", "RB", 10.0, 0, 0), (2024, "c", "RB", 30.0, 0, 0),
                 (2024, "w", "WR", 2.0, 0, 0)])
    j = pairs(p, "rank")
    assert sorted(zip(j["player_id"], j["player_id_b"])) == [("a", "b")]  # c too far, w other position


def test_scoring_from_both_sides():
    # market ranks a ahead of b. a scored more -> market right. We projected b higher -> we're wrong, and we disagreed.
    p = players([(2024, "a", "WR", 1.0, 100.0, 250.0), (2024, "b", "WR", 5.0, 200.0, 150.0),
                 (2024, "c", "WR", 9.0, 50.0, 150.0)])  # b and c tie on points -> that pair is dropped
    s = score_pairs(pairs(p, "rank"), "points").sort("player_id", "player_id_b")
    assert s.select("player_id", "player_id_b").rows() == [("a", "b"), ("a", "c")]
    assert s["market_right"].to_list() == [True, True]
    assert s["we_right"].to_list() == [False, True]
    assert s["disagree"].to_list() == [True, False]


def test_bootstrap_brackets_the_mean():
    rows = [(2024, f"p{i}", "WR", float(i), 0.0, 0.0) for i in range(30)]
    s = score_pairs(pairs(players(rows).with_columns(points=pl.col("rank") * -1.0, season_b=pl.col("rank")), "rank"), "points")
    mean, lo, hi = bootstrap_rate(s, "we_right")
    assert mean == 0.0 and lo == 0.0 and hi == 0.0  # we're always wrong here: projections reversed
    s2 = s.with_columns(we_right=pl.int_range(pl.len()) % 2 == 0)
    mean, lo, hi = bootstrap_rate(s2, "we_right")
    assert lo <= mean <= hi and 0.3 < mean < 0.7


@pytest.mark.parametrize("raw,clean", [("D.J. Moore", "dj moore"), ("DJ Moore", "dj moore"),
                                       ("Marvin Harrison Jr.", "marvin harrison"), ("Hollywood Brown", "marquise brown"),
                                       ("Amon-Ra St. Brown", "amon-ra st brown"), ("Michael Pittman Jr.", "michael pittman")])
def test_name_normalization(raw, clean):
    assert pl.select(normalize(pl.lit(raw))).item() == clean


def test_weekly_snapshot_maps_to_the_week_it_sits_inside():
    from datetime import date as d
    from src.backtest.weekly_vs_consensus import snapshot_weeks
    sched = pl.DataFrame({"season": [2023] * 4, "week": [2, 2, 3, 3], "game_type": ["REG"] * 4,
                          "gameday": ["2023-09-14", "2023-09-18", "2023-09-21", "2023-09-25"]})
    w = snapshot_weeks(sched).sort("week")
    friday = d(2023, 9, 15)  # after week 2's Thursday game, before its Sunday/Monday games
    hit = w.filter(pl.lit(friday) >= pl.col("first"), pl.lit(friday) <= pl.col("last"))
    assert hit["week"].to_list() == [2]


def test_weekly_pairs_stay_inside_one_week_and_position():
    from src.backtest.weekly_vs_consensus import pairs_weekly
    p = pl.DataFrame({"season": [2023] * 4, "week": [5, 5, 5, 6], "position": ["WR", "WR", "RB", "WR"],
                      "player_id": ["a", "b", "c", "d"], "rank": [10.0, 14.0, 11.0, 12.0]})
    j = pairs_weekly(p)
    assert list(zip(j["player_id"], j["player_id_b"])) == [("a", "b")]
