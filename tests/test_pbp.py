import polars as pl
import pytest

from src.ingest.pbp import check_pbp, fill_rates, load_if_available


def play(**overrides):
    row = {"game_id": "2024_01_AAA_BBB", "play_id": 1, "season": 2024, "play_type": "pass",
           "posteam": "AAA", "defteam": "BBB", "epa": 0.5}
    row.update(overrides)
    return row


def test_clean_season_has_no_issues():
    df = pl.DataFrame([play(play_id=1), play(play_id=2, play_type="run")])
    assert check_pbp(df, 2024, expected_games={"2024_01_AAA_BBB"}) == []


def test_duplicate_plays_are_flagged():
    df = pl.DataFrame([play(play_id=1), play(play_id=1)])
    assert any("duplicate" in i for i in check_pbp(df, 2024))


def test_wrong_season_is_flagged():
    df = pl.DataFrame([play(season=2023)])
    assert any("season other than 2024" in i for i in check_pbp(df, 2024))


def test_missing_game_is_flagged():
    df = pl.DataFrame([play()])
    issues = check_pbp(df, 2024, expected_games={"2024_01_AAA_BBB", "2024_01_CCC_DDD"})
    assert any("1 completed games have no plays" in i and "CCC_DDD" in i for i in issues)


def test_null_defteam_on_scrimmage_play_is_flagged():
    df = pl.DataFrame([play(play_id=1), play(play_id=2, defteam=None)])
    assert any("defteam is null on 50.0%" in i for i in check_pbp(df, 2024))


def test_nulls_on_non_scrimmage_plays_are_ignored():
    # Timeouts and end-of-quarter rows legitimately have no defteam or epa.
    df = pl.DataFrame([play(play_id=1), play(play_id=2, play_type=None, defteam=None, epa=None)])
    assert check_pbp(df, 2024) == []


def test_load_if_available_returns_none_for_uncovered_season():
    def loader(seasons):
        raise ValueError("Season must be between 2022 and 2026")
    assert load_if_available(loader, 2020) is None


def test_load_if_available_does_not_hide_other_errors():
    # A network failure is a real problem, not "no data for this season" -- it must still crash.
    def loader(seasons):
        raise ConnectionError("offline")
    with pytest.raises(ConnectionError):
        load_if_available(loader, 2024)


def test_fill_rates():
    df = pl.DataFrame({"route": ["GO", None, "SLANT", None], "was_pressure": [True, False, True, False]})
    assert fill_rates(df, ["route", "was_pressure"]) == {"route": 0.5, "was_pressure": 1.0}


def test_fill_rates_on_empty_frame_is_zero_not_a_crash():
    df = pl.DataFrame({"route": []}, schema={"route": pl.String})
    assert fill_rates(df, ["route"]) == {"route": 0.0}
