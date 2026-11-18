"""Chance he plays: depth-chart rate x injury rate, applied to live projections."""

import polars as pl
import pytest

from src.models.play_chance import PlayChance, apply


def pool():
    rows = []
    # healthy QB1s play 9/10, QB2s 3/10
    rows += [("QB", 1, None, None, i < 9) for i in range(10)]
    rows += [("QB", 2, None, None, i < 3) for i in range(10)]
    # Questionable + did not practice: 4 of 10 played; Out: none
    rows += [("RB", 1, "Questionable", "DNP", i < 4) for i in range(10)]
    rows += [("RB", 1, "Out", "DNP", False) for _ in range(10)]
    rows += [("RB", 1, None, None, True) for _ in range(10)]
    return pl.DataFrame(rows, schema=["position", "depth_rank", "report_status", "practice_code", "played"], orient="row")


@pytest.fixture
def chance(monkeypatch):
    monkeypatch.setattr("src.models.play_chance.MIN_N", 5)
    return PlayChance().fit(pool())


def test_rates_multiply_depth_and_injury(chance):
    rows = pl.DataFrame({"position": ["QB", "QB", "RB", "RB"], "depth_rank": [1, 2, 1, 1],
                         "report_status": [None, None, "Questionable", "Out"], "practice_code": [None, None, "DNP", "DNP"]})
    p = chance.rates(rows)["p_play"].to_list()
    assert p == pytest.approx([0.9, 0.3, 1.0 * 0.4, 0.0])


def test_unknown_depth_falls_back_to_position_rate(chance):
    rows = pl.DataFrame({"position": ["QB"], "depth_rank": [None], "report_status": [None], "practice_code": [None]},
                        schema_overrides={"depth_rank": pl.Int64, "report_status": pl.String, "practice_code": pl.String})
    assert chance.rates(rows)["p_play"][0] == pytest.approx(0.6)  # 12 of 20 healthy QBs played


def test_apply_keeps_points_if_he_plays_and_zeroes_the_low_end_when_he_often_sits(chance):
    preds = pl.DataFrame({"position": ["QB", "QB"], "depth_rank": [1, 2], "report_status": [None, None],
                          "practice_code": [None, None], "proj": [20.0, 12.0],
                          "likely_low": [15.0, 7.0], "likely_high": [25.0, 17.0]},
                         schema_overrides={"report_status": pl.String, "practice_code": pl.String})
    out = apply(preds, chance)
    assert out["proj_if_plays"].to_list() == [20.0, 12.0]
    assert out["proj"].to_list() == pytest.approx([18.0, 3.6])
    assert out["likely_low"].to_list() == [15.0, 0.0]   # backup QB sits 70% of the time
