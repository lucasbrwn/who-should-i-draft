"""Expert blend: FP rank -> points never uses its own season; blend averages whichever sources exist."""

import polars as pl
import pytest

from src.backtest.expert_blend import add_blends, fp_points


def frame():
    rows = []
    for season, top_pts in [(2022, 20.0), (2023, 20.0), (2024, 999.0)]:   # 2024's outcomes are absurd
        for rank, pts in [(1, top_pts), (2, 10.0)]:
            rows.append({"player_id": f"p{rank}", "season": season, "week": 1, "position": "WR",
                         "fp_rank": float(rank), "actual": pts})
    return pl.DataFrame(rows)


def test_rank_map_ignores_its_own_season():
    f = fp_points(frame())
    top_2024 = f.filter((pl.col("season") == 2024) & (pl.col("fp_rank") == 1))["fp_pts"][0]
    assert top_2024 == pytest.approx(20.0)          # learned from 2022-23 only, not 2024's 999


def test_blend_averages_available_sources():
    f = pl.DataFrame({"expected": [10.0, 10.0], "espn_proj": [14.0, None], "fp_pts": [12.0, 16.0]})
    b = add_blends(f)
    assert b["blend"].to_list() == pytest.approx([12.0, 13.0])
    assert b["experts only"].to_list() == pytest.approx([13.0, 16.0])
