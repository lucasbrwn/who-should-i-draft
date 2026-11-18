"""Live week predictions (SCRUM-73/74/76): who gets a row, what the model trains on, bust/ceiling."""

import numpy as np
import polars as pl
import pytest

from src.predict.week import train_rows, upcoming_rows


def roster(rows):
    return pl.DataFrame(rows, schema=["season", "week", "status", "position", "gsis_id", "full_name", "team"], orient="row")


SCHEDULE = pl.DataFrame({
    "season": [2026], "week": [11], "game_id": ["2026_11_IND_HOU"], "gameday": ["2026-11-19"],
    "home_team": ["HOU"], "away_team": ["IND"],
})
NO_INJURIES = pl.DataFrame(schema={"season": pl.Int32, "week": pl.Int32, "game_type": pl.String, "gsis_id": pl.String,
                                   "team": pl.String, "report_status": pl.String, "practice_code": pl.String,
                                   "report_primary_injury": pl.String})


def test_upcoming_rows_only_active_skill_players_whose_team_plays():
    r = roster([
        (2026, 11, "ACT", "WR", "a", "Active WR", "IND"),
        (2026, 11, "RES", "QB", "b", "On IR", "IND"),        # injured reserve: no row
        (2026, 11, "ACT", "OL", "c", "Lineman", "HOU"),      # not a skill position
        (2026, 11, "ACT", "RB", "d", "Bye Week RB", "MIA"),  # team not playing
        (2026, 10, "ACT", "TE", "e", "Last Week Only", "HOU"),
        (2026, 11, "ACT", "TE", "f", "Home TE", "HOU"),
    ])
    rows = upcoming_rows(r, SCHEDULE, NO_INJURIES, 2026, 11).sort("player_id")
    assert rows["player_id"].to_list() == ["a", "f"]
    assert rows["opponent_team"].to_list() == ["HOU", "IND"]
    assert rows["is_home"].to_list() == [False, True]
    assert not rows["left_injured"].any() and not rows["on_injury_report"].any()


def test_upcoming_rows_pick_up_the_injury_report_when_it_is_out():
    r = roster([(2026, 11, "ACT", "WR", "a", "Questionable WR", "IND")])
    inj = pl.DataFrame({"season": [2026], "week": [11], "game_type": ["REG"], "gsis_id": ["a"], "team": ["IND"],
                        "report_status": ["Questionable"], "practice_code": ["LP"], "report_primary_injury": ["Ankle"]},
                       schema_overrides=NO_INJURIES.schema)
    rows = upcoming_rows(r, SCHEDULE, inj, 2026, 11)
    assert rows["report_status"].to_list() == ["Questionable"]


def test_train_rows_never_include_this_week_or_the_future():
    # 2025 is used for training since the final exam ran (2026-11-18, v2)
    feats = pl.DataFrame({"season": [2024, 2025, 2026, 2026, 2026], "week": [5, 5, 10, 11, 12],
                          "player_id": ["a"] * 5})
    kept = train_rows(feats, 2026, 11)
    assert kept.select("season", "week").rows() == [(2024, 5), (2025, 5), (2026, 10)]


def test_scenario_blend_weights_by_chance_starters_play():
    from src.predict.week import scenario_blend
    out = pl.DataFrame({"player_id": ["walker", "johnson", "other"], "team": ["KC", "KC", "BUF"],
                        "proj": [20.0, 6.0, 10.0]})
    without = pl.DataFrame({"player_id": ["johnson", "other"], "proj_without": [18.0, 10.0]})
    absent = pl.DataFrame({"player_id": ["walker"], "team": ["KC"], "p_play": [0.45]})
    r = scenario_blend(out, without, absent).sort("player_id")
    johnson = r.filter(pl.col("player_id") == "johnson").row(0, named=True)
    assert johnson["proj"] == pytest.approx(0.45 * 6 + 0.55 * 18)     # blend
    assert johnson["proj_without"] == 18.0 and johnson["absent_chance"] == pytest.approx(0.55)
    walker = r.filter(pl.col("player_id") == "walker").row(0, named=True)
    assert walker["proj"] == 20.0 and walker["absent_chance"] == 0.0   # the absent player himself isn't blended
    assert r.filter(pl.col("player_id") == "other")["proj"][0] == 10.0  # other teams untouched


def test_next_week_is_the_week_of_the_next_game():
    from datetime import date
    from src.predict.next_week import next_week
    sched = pl.DataFrame({"season": [2026] * 3, "week": [10, 11, 11], "game_type": ["REG"] * 3,
                          "gameday": ["2026-11-16", "2026-11-19", "2026-11-22"]})
    assert next_week(date(2026, 11, 19), sched) == (2026, 11)   # Thursday night game day
    assert next_week(date(2026, 11, 21), sched) == (2026, 11)   # Saturday: Sunday games ahead
    assert next_week(date(2026, 11, 17), sched) == (2026, 11)   # Tuesday after week 10
