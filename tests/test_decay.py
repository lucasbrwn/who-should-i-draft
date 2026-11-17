"""Exponentially weighted recent form: half-life math, no peeking, skips injured exits, crosses seasons."""

import polars as pl
import pytest

from src.features.decay import HALF_LIFE, build_decay_features
from src.features.usage import MEAN_STATS


def player_weeks(points, seasons=None, weeks=None, left_injured=None):
    n = len(points)
    df = pl.DataFrame({
        "player_id": ["a"] * n, "player_display_name": ["A"] * n, "position": ["WR"] * n,
        "season": seasons or [2024] * n, "week": weeks or list(range(1, n + 1)),
        "game_id": [f"g{i}" for i in range(n)], "team": ["X"] * n,
        "left_injured": left_injured or [False] * n,
        "fantasy_points_ppr": points,
    })
    stats = [c for c in MEAN_STATS + ["rushing_epa", "receiving_epa", "passing_epa", "sacks_suffered"]
             if c not in df.columns]
    return df.with_columns([pl.lit(1.0).alias(c) for c in stats])


def test_first_game_has_no_history_and_second_sees_only_the_first():
    f = build_decay_features(player_weeks([10.0, 30.0, 999.0])).sort("week")
    assert f["fantasy_points_ppr_ewm"][0] is None
    assert f["fantasy_points_ppr_ewm"][1] == pytest.approx(10.0)


def test_half_life_weights_recent_games_more():
    # games 10 then 20: weights are 1/2 and 1 (half-life 3 -> one game back = 0.5 ** (1/3))
    f = build_decay_features(player_weeks([10.0, 20.0, 0.0])).sort("week")
    w = 0.5 ** (1 / HALF_LIFE)
    assert f["fantasy_points_ppr_ewm"][2] == pytest.approx((w * 10 + 20) / (w + 1))
    assert f["fantasy_points_ppr_ewm"][2] > 15  # closer to the recent 20 than a plain average


def test_left_injured_games_are_skipped():
    f = build_decay_features(player_weeks([10.0, 0.5, 0.0], left_injured=[False, True, False])).sort("week")
    assert f["fantasy_points_ppr_ewm"][2] == pytest.approx(10.0)


def test_carries_across_seasons_into_week_one():
    f = build_decay_features(player_weeks([12.0, 14.0, 0.0], seasons=[2023, 2023, 2024], weeks=[16, 17, 1]))
    week1 = f.filter(pl.col("season") == 2024)["fantasy_points_ppr_ewm"][0]
    assert 12.0 < week1 < 14.0


def test_upcoming_blank_row_gets_history():
    pw = player_weeks([10.0, 20.0, None])
    f = build_decay_features(pw).sort("week")
    assert f["fantasy_points_ppr_ewm"][2] is not None


def test_looks_back_at_most_to_last_season():
    f = build_decay_features(player_weeks([50.0, 10.0, 0.0], seasons=[2022, 2023, 2024], weeks=[17, 17, 1]))
    assert f.filter(pl.col("season") == 2024)["fantasy_points_ppr_ewm"][0] == pytest.approx(10.0)
