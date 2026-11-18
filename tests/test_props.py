"""Prop-line saving: flatten the odds API response and match player names to ids."""

import polars as pl

from src.ingest.props import flatten, name_key

EVENT = {"bookmakers": [{"key": "draftkings", "markets": [
    {"key": "player_anytime_td", "last_update": "2026-11-21T18:00:00Z",
     "outcomes": [{"name": "Yes", "description": "Jahmyr Gibbs", "price": -160}]},
    {"key": "player_rush_yds", "last_update": "2026-11-21T18:00:00Z",
     "outcomes": [{"name": "Over", "description": "Jahmyr Gibbs", "price": -115, "point": 78.5},
                  {"name": "Under", "description": "Jahmyr Gibbs", "price": -105, "point": 78.5}]},
]}]}


def test_flatten_one_row_per_outcome():
    rows = flatten(EVENT, "2026_11_TB_DET", "2026-11-21T19:00:00Z")
    assert len(rows) == 3
    over = [r for r in rows if r["outcome"] == "Over"][0]
    assert (over["market"], over["line"], over["price"], over["game_id"]) == ("player_rush_yds", 78.5, -115, "2026_11_TB_DET")
    assert [r for r in rows if r["market"] == "player_anytime_td"][0]["line"] is None


def test_flatten_empty_event():
    assert flatten({}, "g", "t") == []


def test_name_key_ignores_suffixes_and_punctuation():
    names = pl.DataFrame({"n": ["Kenneth Walker III", "kenneth walker", "D'Andre Swift", "Marvin Harrison Jr."]})
    keys = names.select(name_key(pl.col("n")))["n"].to_list()
    assert keys == ["kenneth walker", "kenneth walker", "dandre swift", "marvin harrison"]
