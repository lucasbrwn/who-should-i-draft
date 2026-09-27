import pytest

from src.scoring.config import (
    PRESETS, LeagueConfig, Roster, list_leagues, load_league, save_league, slugify,
)


@pytest.mark.parametrize("key", PRESETS)
def test_presets_are_valid(key):
    assert PRESETS[key].validate() == []


def test_summaries_describe_the_league():
    assert PRESETS["standard-ppr"].summary() == "12-team · PPR · Redraft"
    assert PRESETS["superflex-half-ppr"].summary() == "12-team · Half-PPR · Superflex · Redraft"
    assert PRESETS["bestball-half-ppr"].summary() == "12-team · Half-PPR · Best Ball"


def test_roster_counts():
    roster = Roster(QB=1, RB=2, WR=3, TE=1, FLEX=2, SUPERFLEX=1, K=0, DST=0, BENCH=6)
    assert roster.starters == 10
    assert roster.size == 16


def test_round_trip_through_dict_and_file(tmp_path):
    config = LeagueConfig.from_dict(PRESETS["superflex-half-ppr"].to_dict())
    config.league.name = "Home League!"
    path = save_league(config, tmp_path)
    assert path.name == "home-league.json"
    assert load_league(path) == config
    assert list(list_leagues(tmp_path)) == ["home-league"]


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="unknown scoring setting"):
        LeagueConfig.from_dict({"scoring": {"recepshuns": 1}})


def test_validation_messages():
    config = LeagueConfig.from_dict({
        "league": {"name": " ", "type": "dynasty", "teams": 40},
        "roster": {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0, "K": 0, "DST": 0, "IDP": 2},
        "scoring": {"receptions": 3.0},
    })
    errors = config.validate()
    assert "League name is required." in errors
    assert "Dynasty leagues are planned but not supported yet." in errors
    assert "Teams must be between 4 and 32." in errors
    assert "IDP slots are planned for a later version." in errors
    assert "Points per reception must be between 0 and 2." in errors


def test_slugify():
    assert slugify("  My  League #2 ") == "my-league-2"
    assert slugify("!!!") == "league"
