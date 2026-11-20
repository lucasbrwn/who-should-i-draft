"""Rank remap: the ranker decides the ORDER, our projection supplies the POINTS."""

import polars as pl

from src.backtest.ranker_test import rank_remap


def frame(scores, preds, pos="WR", week=1):
    n = len(scores)
    return pl.DataFrame({"player_id": [f"{pos}{week}{i}" for i in range(n)], "season": [2024] * n,
                         "week": [week] * n, "position": [pos] * n, "score": scores, "pred": preds})


def test_ranker_order_gets_our_point_values():
    # ranker likes player 2 best, then 0, then 1; our points are 20 / 10 / 5
    out = rank_remap(frame([0.5, -1.0, 2.0], [20.0, 10.0, 5.0]), "score", "pred", "remap")
    assert out["remap"].to_list() == [10.0, 5.0, 20.0]


def test_same_points_just_reshuffled():
    f = frame([3.0, 1.0, 2.0, 0.0], [7.0, 18.0, 2.0, 11.0])
    out = rank_remap(f, "score", "pred", "remap")
    assert sorted(out["remap"].to_list()) == sorted(f["pred"].to_list())


def test_when_ranker_agrees_nothing_changes():
    f = frame([3.0, 2.0, 1.0], [15.0, 9.0, 4.0])
    assert rank_remap(f, "score", "pred", "remap")["remap"].to_list() == [15.0, 9.0, 4.0]


def test_groups_never_mix():
    # a QB must never receive a WR's points, and week 1 never week 2's
    f = pl.concat([frame([1.0, 2.0], [30.0, 25.0], pos="QB"), frame([1.0, 2.0], [8.0, 6.0], pos="WR"),
                   frame([2.0, 1.0], [12.0, 3.0], pos="WR", week=2)])
    out = rank_remap(f, "score", "pred", "remap")
    assert out["remap"].to_list() == [25.0, 30.0, 6.0, 8.0, 12.0, 3.0]
