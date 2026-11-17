"""Live week predictions (SCRUM-73/74/76): who gets a row, what the model trains on, bust/ceiling."""

import numpy as np
import polars as pl
import pytest

from src.predict.week import apply_ranges, fit_ranges, train_rows, upcoming_rows


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


def test_train_rows_never_include_this_week_the_future_or_the_final_exam():
    feats = pl.DataFrame({"season": [2024, 2025, 2026, 2026, 2026], "week": [5, 5, 10, 11, 12]})
    kept = train_rows(feats, 2026, 11)
    assert kept.rows() == [(2024, 5), (2026, 10)]


@pytest.fixture
def backtest():
    """Fake backtest: actual = projection + noise (sd 6), floored at 0 like real scores; 2000 games per position."""
    rng = np.random.default_rng(0)
    n, positions = 2000, ["QB", "RB", "WR", "TE"]
    pred = rng.uniform(0, 25, n * len(positions))
    return pl.DataFrame({"position": np.repeat(positions, n), "pred": pred,
                         "actual": np.clip(pred + rng.normal(0, 6, pred.size), 0, None)})


def test_ranges_cover_about_80_percent_and_bracket_the_projection(backtest):
    ranges = fit_ranges(backtest)
    out = apply_ranges(backtest.rename({"pred": "proj"}), ranges)
    inside = ((out["actual"] >= out["bust"]) & (out["actual"] <= out["ceiling"])).mean()
    assert 0.75 < inside < 0.85
    assert (out["bust"] >= 0).all()                        # no negative busts
    assert (out["bust"] <= out["proj"]).all() and (out["ceiling"] >= out["proj"]).all()


def test_higher_projections_get_higher_ceilings(backtest):
    out = apply_ranges(pl.DataFrame({"position": ["WR", "WR"], "proj": [3.0, 22.0]}), fit_ranges(backtest))
    assert out["ceiling"][1] > out["ceiling"][0]
