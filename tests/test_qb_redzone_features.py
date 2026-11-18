"""QB red-zone passing: counts, share, trips, Vegas link, no peeking, QBs only."""

import polars as pl
import pytest

from src.features.qb_redzone import build_qb_redzone_features


def player_weeks(n=3, position="QB"):
    return pl.DataFrame({
        "player_id": ["q"] * n, "position": [position] * n, "season": [2024] * n,
        "week": list(range(1, n + 1)), "game_id": [f"g{i}" for i in range(1, n + 1)], "team": ["X"] * n,
        "left_injured": [False] * n, "fantasy_points_ppr": [10.0] * n,
    })


def play(game, play_type="pass", passer="q", yardline=15, drive=1, two_pt=0):
    return {"game_id": game, "season_type": "REG", "posteam": "X", "play_type": play_type,
            "yardline_100": yardline, "passer_player_id": passer, "fixed_drive": drive,
            "qb_spike": 0, "two_point_attempt": two_pt}


def pbp(plays):
    return pl.DataFrame(plays, schema_overrides={"passer_player_id": pl.Utf8})


def schedules(n=3, total=49.0):
    return pl.DataFrame({
        "game_id": [f"g{i}" for i in range(1, n + 1)], "season": [2024] * n, "week": list(range(1, n + 1)),
        "home_team": ["X"] * n, "away_team": ["Y"] * n, "total_line": [total] * n, "spread_line": [0.0] * n,
        "div_game": [0] * n, "roof": ["outdoors"] * n, "temp": [70] * n, "wind": [0] * n,
        "home_rest": [7] * n, "away_rest": [7] * n, "home_score": [None] * n, "away_score": [None] * n,
    })


def test_counts_share_trips_and_vegas_link():
    plays = [play("g1", yardline=15, drive=1), play("g1", yardline=8, drive=1),         # 2 rz passes, 1 at <=10
             play("g1", "run", passer=None, yardline=3, drive=1),                       # team rz play
             play("g1", "run", passer=None, yardline=18, drive=2),                      # 2nd trip
             play("g1", yardline=40, drive=3),                                          # not red zone
             play("g1", yardline=2, drive=2, two_pt=1)]                                 # ignored
    f = build_qb_redzone_features(player_weeks(2), pbp(plays), schedules(2)).sort("week")
    assert f["qb_rz_passes_ewm"][0] is None
    w2 = f.row(1, named=True)
    assert w2["qb_rz_passes_ewm"] == pytest.approx(2)
    assert w2["qb_gl10_passes_ewm"] == pytest.approx(1)
    assert w2["qb_rz_pass_share_ewm"] == pytest.approx(2 / 4)
    assert w2["qb_team_rz_trips_ewm"] == pytest.approx(2)
    assert w2["qb_rz_pass_td_exp"] == pytest.approx(24.5 / 7 * 0.5)   # even spread: implied total 24.5


def test_future_games_do_not_leak():
    a = build_qb_redzone_features(player_weeks(3), pbp([play("g1"), play("g2")]), schedules())
    b = build_qb_redzone_features(player_weeks(3), pbp([play("g1"), play("g2")] + [play("g3")] * 20), schedules())
    assert a.sort("week")["qb_rz_passes_ewm"].to_list() == b.sort("week")["qb_rz_passes_ewm"].to_list()


def test_non_qbs_get_nulls():
    f = build_qb_redzone_features(player_weeks(2, position="WR"), pbp([play("g1")]), schedules(2))
    assert f["qb_rz_passes_ewm"].is_null().all()
