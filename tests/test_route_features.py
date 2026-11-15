"""Routes-run features (SCRUM-87)."""

import polars as pl
import pytest

from src.features.routes import add_route_columns, build_route_features, routes_per_game


def pbp(rows):
    return pl.DataFrame(rows, schema=["game_id", "play_id", "season_type", "posteam", "qb_dropback", "qb_spike"], orient="row")


def part(rows):
    return pl.DataFrame(rows, schema=["game_id", "play_id", "offense_players"], orient="row")


def test_routes_are_dropbacks_on_the_field_and_spikes_dont_count():
    plays = pbp([("g1", 1, "REG", "DET", 1, 0), ("g1", 2, "REG", "DET", 1, 0), ("g1", 3, "REG", "DET", 0, 0),
                 ("g1", 4, "REG", "DET", 1, 1), ("g1", 5, "POST", "DET", 1, 0)])
    on_field = part([("g1", 1, "wr1;te1"), ("g1", 2, "wr1"), ("g1", 3, "wr1;te1;rb1"), ("g1", 4, "wr1;te1"),
                     ("g1", 5, "wr1;te1")])
    g = routes_per_game(on_field, plays).sort("player_id")
    # Only plays 1 and 2 count: 3 is a run, 4 a spike, 5 a playoff game.
    assert g.select("player_id", "routes", "team_dropbacks").rows() == [("te1", 1, 2), ("wr1", 2, 2)]


def test_no_participation_data_means_blank_not_zero():
    pw = pl.DataFrame({"player_id": ["wr1", "rb9", "wr1"], "game_id": ["g1", "g1", "g2"], "team": ["DET"] * 3})
    per_game = pl.DataFrame({"game_id": ["g1"], "team": ["DET"], "team_dropbacks": [30], "player_id": ["wr1"], "routes": [28]})
    out = add_route_columns(pw, per_game).sort("game_id", "player_id")
    # rb9 was in g1 (which has data) but never on the field for a dropback -> 0. g2 has no data -> blank.
    assert out.select("player_id", "game_id", "routes").rows() == [("rb9", "g1", 0), ("wr1", "g1", 28), ("wr1", "g2", None)]


def test_windows_use_earlier_games_only():
    pw = pl.DataFrame({
        "player_id": ["wr1"] * 3, "player_display_name": ["W"] * 3, "position": ["WR"] * 3, "season": [2024] * 3,
        "week": [1, 2, 3], "game_id": ["g1", "g2", "g3"], "team": ["DET"] * 3,
        "targets": [10, 2, 6], "receiving_yards": [100.0, 20.0, 60.0], "left_injured": [False] * 3,
    })
    plays, on = [], []
    for g, n_db, n_routes in [("g1", 40, 40), ("g2", 40, 20), ("g3", 40, 30)]:
        for i in range(n_db):
            plays.append((g, i, "REG", "DET", 1, 0))
            on.append((g, i, "wr1" if i < n_routes else "wr2"))
    f = build_route_features(pw, part(on), pbp(plays)).sort("week")
    assert f["routes_std"].to_list()[0] is None                              # week 1: nothing before it
    assert f["route_participation_std"][1] == pytest.approx(1.0)            # week 2 sees only week 1: 40/40
    assert f["route_participation_std"][2] == pytest.approx(60 / 80)        # week 3: weeks 1-2, never week 3
    assert f["targets_per_route_std"][2] == pytest.approx(12 / 60)
    assert f["yards_per_route_std"][2] == pytest.approx(120 / 60)
