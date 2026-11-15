"""Offseason vacated opportunity (SCRUM-94)."""

import polars as pl
import pytest

from src.features.offseason import build_offseason_features


def pw_rows():
    # 2023: DET's WR1 (wr1) had 60 of 100 targets, wr2 40; rb1 all 50 carries. In 2024 wr1 has left for KC.
    rows = []
    for wk in (1, 2):
        rows += [("wr1", "WR", 2023, wk, "DET", 30, 0), ("wr2", "WR", 2023, wk, "DET", 20, 0),
                 ("rb1", "RB", 2023, wk, "DET", 0, 25), ("kc1", "WR", 2023, wk, "KC", 50, 0)]
    rows += [("wr2", "WR", 2024, 1, "DET", 8, 0), ("rb1", "RB", 2024, 1, "DET", 0, 20),
             ("wr1", "WR", 2024, 1, "KC", 9, 0), ("kc1", "WR", 2024, 1, "KC", 7, 0)]
    return pl.DataFrame(rows, schema=["player_id", "position", "season", "week", "team", "targets", "carries"], orient="row")


def week1():
    return pl.DataFrame({"season": [2023] * 4 + [2024] * 4, "team": ["DET", "DET", "DET", "KC", "DET", "DET", "KC", "KC"],
                         "player_id": ["wr1", "wr2", "rb1", "kc1", "wr2", "rb1", "wr1", "kc1"]})


def test_departures_and_arrivals():
    f = build_offseason_features(pw_rows(), week1()).filter(pl.col("season") == 2024)
    det = f.filter(pl.col("player_id") == "wr2").row(0, named=True)
    kc = f.filter(pl.col("player_id") == "kc1").row(0, named=True)
    assert det["offseason_vacated_tgt_wr"] == pytest.approx(0.6)     # wr1's 60% of DET targets left
    assert det["offseason_vacated_carry"] == pytest.approx(0.0)      # rb1 stayed
    assert kc["offseason_arrived_tgt"] == pytest.approx(0.6)         # KC brought in a 60%-share receiver
    assert kc["offseason_vacated_tgt_total"] == pytest.approx(0.0)


def test_first_season_is_blank():
    f = build_offseason_features(pw_rows(), week1()).filter(pl.col("season") == 2023)
    assert f["offseason_vacated_tgt_total"].null_count() == f.height
