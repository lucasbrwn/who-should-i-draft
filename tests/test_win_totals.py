"""Win totals paste parser + validation (SCRUM-88)."""

import polars as pl
import pytest

from src.ingest.win_totals import PASTE_PATH, TEAM_CODES, parse_paste, validate

NFLVERSE_TEAMS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND",
    "JAX", "KC", "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF",
    "TB", "TEN", "WAS",
}


def season_df(season, totals):
    """A full 32-team season with the given totals (one value repeated, or a list of 32)."""
    totals = totals if isinstance(totals, list) else [totals] * 32
    return pl.DataFrame({"season": [season] * 32, "team": sorted(NFLVERSE_TEAMS), "win_total": totals})


def test_parses_both_header_styles_and_keeps_only_win_total():
    text = """26: Tm,Super Bowl Odds,W/L O-U,Record
Los Angeles Rams,+500,11.5,5-4

21:
Tm,Super Bowl Odds,W/L O-U,Record
Washington Football Team,+4000,8.5,7-10 (under)
"""
    df = parse_paste(text)
    assert df.columns == ["season", "team", "win_total"]  # record + Super Bowl odds dropped
    assert df.rows() == [(2026, "LA", 11.5), (2021, "WAS", 8.5)]


def test_unknown_team_name_fails_loudly():
    with pytest.raises(ValueError, match="unknown team name"):
        parse_paste("25: Tm,Super Bowl Odds,W/L O-U,Record\nSt. Louis Rams,+500,9.5,1-0\n")


def test_missing_win_total_names_the_team():
    with pytest.raises(ValueError, match="line 2: Buffalo Bills has no usable win total"):
        parse_paste("25: Tm,Super Bowl Odds,W/L O-U,Record\nBuffalo Bills,+700,,12-5 (under)\n")


def test_row_before_season_header_fails():
    with pytest.raises(ValueError, match="before any season header"):
        parse_paste("Los Angeles Rams,+500,11.5,5-4\n")


def test_team_codes_are_nflverse_codes():
    assert set(TEAM_CODES.values()) == NFLVERSE_TEAMS


def test_validate_accepts_a_balanced_season():
    validate(season_df(2024, 8.5))  # 32 * 8.5 = 272 = 16 * 17 games


def test_validate_catches_missing_team():
    with pytest.raises(ValueError, match="expected 32"):
        validate(season_df(2024, 8.5).head(31))


def test_validate_catches_typo_via_sum():
    totals = [8.5] * 31 + [85.0]  # someone dropped the decimal point
    with pytest.raises(ValueError, match="win total is outside|sum to"):
        validate(season_df(2024, totals))


def test_validate_uses_16_games_for_2020():
    validate(season_df(2020, 8.0))  # 256 = 16 * 16 games: fine for 2020
    with pytest.raises(ValueError, match="sum to"):
        validate(season_df(2020, 8.5))  # 272 would be a 17-game season


def test_real_paste_is_complete_and_valid():
    df = parse_paste(PASTE_PATH.read_text(encoding="utf-8"))
    validate(df)
    assert sorted(df["season"].unique()) == list(range(2020, 2027))
    assert df.height == 7 * 32
