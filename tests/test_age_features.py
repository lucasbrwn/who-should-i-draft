"""Age + years of experience (SCRUM-90)."""

from datetime import date

import polars as pl
import pytest

from src.features.age import build_age_features, players_from_rosters


def test_age_on_game_day_and_years_of_experience():
    rosters = pl.DataFrame({
        "gsis_id": ["vet", "vet", "rook", None],
        "birth_date": [None, date(1996, 9, 15), date(2002, 1, 1), date(2000, 1, 1)],  # first file missing the date
        "entry_year": [2018, 2018, 2024, 2020],
    })
    pw = pl.DataFrame({"player_id": ["vet", "rook", "unknown"], "season": [2024] * 3, "week": [1] * 3,
                       "gameday": ["2024-09-15", "2024-09-15", "2024-09-15"]})
    f = build_age_features(pw, players_from_rosters(rosters)).sort("player_id")
    assert f["player_id"].to_list() == ["rook", "unknown", "vet"]
    assert f["age"][2] == pytest.approx(28.0, abs=0.01)  # 28th birthday that day
    assert f["years_exp"].to_list() == [0.0, None, 6.0]  # rookie = 0; unknown player stays blank
    assert f["age"][1] is None
