"""QB rushing split: designed vs scramble counting, no peeking, QBs only, kneels/2-pt ignored."""

import polars as pl
import pytest

from src.features.qb_rush import build_qb_rush_features


def player_weeks(n=3, position="QB"):
    return pl.DataFrame({
        "player_id": ["q"] * n, "position": [position] * n, "season": [2024] * n,
        "week": list(range(1, n + 1)), "game_id": [f"g{i}" for i in range(1, n + 1)],
        "left_injured": [False] * n, "fantasy_points_ppr": [10.0] * n,
    })


def play(game, play_type="run", rusher="q", passer=None, scramble=0, dropback=0, yardline=50, yards=5, two_pt=0):
    return {"game_id": game, "season_type": "REG", "play_type": play_type, "yardline_100": yardline,
            "rusher_player_id": rusher, "passer_player_id": passer, "qb_scramble": scramble,
            "qb_dropback": dropback, "qb_spike": 0, "two_point_attempt": two_pt, "rushing_yards": yards}


def pbp(plays):
    return pl.DataFrame(plays, schema_overrides={"passer_player_id": pl.Utf8, "rusher_player_id": pl.Utf8})


def test_designed_runs_and_scrambles_are_split_and_week_one_has_no_history():
    plays = [play("g1", yards=7), play("g1", yards=3, yardline=4),            # 2 designed, one at the 4
             play("g1", scramble=1, dropback=1, yards=12),                     # 1 scramble
             play("g1", "pass", rusher=None, passer="q", dropback=1),          # 3 more dropbacks
             play("g1", "pass", rusher=None, passer="q", dropback=1),
             play("g1", "pass", rusher=None, passer="q", dropback=1),
             play("g1", "qb_kneel", yards=-1),                                 # ignored
             play("g1", yards=2, two_pt=1)]                                    # ignored
    f = build_qb_rush_features(player_weeks(2), pbp(plays)).sort("week")
    assert f["qb_designed_runs_ewm"][0] is None
    w2 = f.row(1, named=True)
    assert w2["qb_designed_runs_ewm"] == pytest.approx(2)
    assert w2["qb_scrambles_ewm"] == pytest.approx(1)
    assert w2["qb_scramble_rate_ewm"] == pytest.approx(1 / 4)
    assert w2["qb_designed_rush_yds_ewm"] == pytest.approx(10)
    assert w2["qb_rz_designed_runs_ewm"] == pytest.approx(1)
    assert w2["qb_gl_runs_ewm"] == pytest.approx(1)


def test_future_games_do_not_leak():
    base = [play("g1")]
    a = build_qb_rush_features(player_weeks(3), pbp(base + [play("g2")])).sort("week")
    b = build_qb_rush_features(player_weeks(3), pbp(base + [play("g2")] + [play("g3")] * 20)).sort("week")
    assert a["qb_designed_runs_ewm"].to_list() == b["qb_designed_runs_ewm"].to_list()


def test_game_without_runs_counts_as_zero():
    f = build_qb_rush_features(player_weeks(3), pbp([play("g1"), play("g1", "pass", rusher=None, passer="q", dropback=1)
                                                     ])).sort("week")
    assert 0 < f["qb_designed_runs_ewm"][2] < 1


def test_non_qbs_get_nulls():
    f = build_qb_rush_features(player_weeks(2, position="RB"), pbp([play("g1")]))
    assert f["qb_designed_runs_ewm"].is_null().all()
