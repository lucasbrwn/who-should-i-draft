"""Team context features (SCRUM-35). Each test is one rule we agreed on 2026-11-10."""

import polars as pl
import pytest

from src.features.team import build_team_features, check_team_features, join_to_player_weeks, team_games


def sched(week, home="DET", away="GB", season=2025, spread=3.0, total=48.0, roof="outdoors", temp=50, wind=10,
          home_score=24, away_score=20):
    return {"game_id": f"{season}_{week:02d}_{away}_{home}", "season": season, "week": week, "game_type": "REG",
            "home_team": home, "away_team": away, "spread_line": spread, "total_line": total,
            "home_rest": 7, "away_rest": 6, "home_score": home_score, "away_score": away_score,
            "div_game": 1, "roof": roof, "temp": temp, "wind": wind}


def plays(game_id, team, n, proe=0.0):
    return {"game_id": game_id, "team": team, "plays": n, "proe": proe}


def team_row(feats, team, week, season=2025):
    return feats.filter((pl.col("team") == team) & (pl.col("week") == week) & (pl.col("season") == season)).row(0, named=True)


def test_implied_totals_split_the_game_total_by_the_spread():
    games = team_games(pl.DataFrame([sched(1, spread=3.0, total=48.0)])).sort("is_home")
    away, home = games.row(0, named=True), games.row(1, named=True)
    assert home["team_spread"] == 3.0 and away["team_spread"] == -3.0      # positive = home favored
    assert home["implied_total"] == 25.5 and away["implied_total"] == 22.5
    assert home["implied_total"] + away["implied_total"] == 48.0


def test_dome_gets_neutral_weather():
    games = team_games(pl.DataFrame([sched(1, roof="dome", temp=None, wind=None)]))
    assert games["is_dome"].all()
    assert games["temp"].to_list() == [70.0, 70.0] and games["wind"].to_list() == [0.0, 0.0]


def test_outdoor_missing_weather_stays_missing():
    games = team_games(pl.DataFrame([sched(1, temp=None, wind=None)]))
    assert games["temp"].null_count() == 2


def test_tendencies_use_only_earlier_games():
    s = pl.DataFrame([sched(1, home_score=10), sched(2, home_score=40)])
    p = pl.DataFrame([plays(s["game_id"][0], "DET", 60), plays(s["game_id"][1], "DET", 70)])
    feats = build_team_features(s, p)
    assert team_row(feats, "DET", 1)["plays_std"] is None      # nothing before week 1
    assert team_row(feats, "DET", 2)["plays_std"] == 60.0      # week 2 sees only week 1
    assert team_row(feats, "DET", 2)["points_std"] == 10.0     # not its own 40


def test_bye_week_is_skipped():
    s = pl.DataFrame([sched(1), sched(3)])                       # DET has a week-2 bye
    p = pl.DataFrame([plays(s["game_id"][0], "DET", 66), plays(s["game_id"][1], "DET", 50)])
    assert team_row(build_team_features(s, p), "DET", 3)["plays_l5"] == 66.0


def test_previous_season_carries_into_week_1():
    s = pl.DataFrame([sched(17, season=2024, home_score=30), sched(1, season=2025)])
    p = pl.DataFrame([plays(s["game_id"][0], "DET", 64)])
    week1 = team_row(build_team_features(s, p), "DET", 1)
    assert week1["points_prev"] == 30.0 and week1["plays_prev"] == 64.0
    assert week1["points_l5"] is None


def test_each_team_gets_its_own_trends_not_its_opponents():
    s = pl.DataFrame([sched(1, home_score=35, away_score=7), sched(2)])
    p = pl.DataFrame([plays(s["game_id"][0], "DET", 70), plays(s["game_id"][0], "GB", 50)])
    feats = build_team_features(s, p)
    assert team_row(feats, "DET", 2)["points_std"] == 35.0
    assert team_row(feats, "GB", 2)["points_std"] == 7.0
    assert team_row(feats, "GB", 2)["plays_std"] == 50.0


def test_every_player_week_gets_its_teams_context():
    s = pl.DataFrame([sched(1)])
    feats_team = build_team_features(s, pl.DataFrame([plays(s["game_id"][0], "DET", 60)]))
    pw = pl.DataFrame([{"player_id": "a", "season": 2025, "week": 1, "game_id": s["game_id"][0], "team": "GB"}])
    joined = join_to_player_weeks(pw, feats_team)
    assert joined.row(0, named=True)["implied_total"] == pytest.approx(22.5)
    assert check_team_features(joined, pw) == []


def test_check_flags_player_week_with_no_scheduled_game():
    s = pl.DataFrame([sched(1)])
    feats_team = build_team_features(s, pl.DataFrame([plays(s["game_id"][0], "DET", 60)]))
    pw = pl.DataFrame([{"player_id": "a", "season": 2025, "week": 1, "game_id": "nope", "team": "GB"}])
    assert any("didn't match" in p for p in check_team_features(join_to_player_weeks(pw, feats_team), pw))
