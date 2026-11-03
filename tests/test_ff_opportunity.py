"""Expected fantasy points loader (SCRUM-46)."""

import polars as pl

from src.ingest.ff_opportunity import check_ff_opportunity, clean


def opp(**overrides):
    row = {"season": "2024", "week": 1.0, "game_id": "2024_01_ARI_BUF", "player_id": "p1",
           "position": "WR", "total_fantasy_points_exp": 8.4}
    row.update(overrides)
    return row


def test_clean_fixes_key_types_and_drops_rows_without_a_player():
    df = clean(pl.DataFrame([opp(), opp(player_id=None)]))
    assert df.schema["season"] == pl.Int32 and df.schema["week"] == pl.Int32
    assert df.height == 1


def test_kneel_downs_are_not_an_error():
    # backup QB kneeling: about -0.3 expected points is real
    assert check_ff_opportunity(clean(pl.DataFrame([opp(position="QB", total_fantasy_points_exp=-0.3)]))) == []


def test_broken_values_and_duplicates_are_reported():
    df = clean(pl.DataFrame([opp(total_fantasy_points_exp=-9.0), opp(total_fantasy_points_exp=-9.0)]))
    issues = check_ff_opportunity(df)
    assert any("duplicate" in i for i in issues)
    assert any("below" in i for i in issues)


def test_every_target_or_carry_needs_expected_points():
    pw = pl.DataFrame({"player_id": ["p1", "p2", "p3"], "game_id": ["2024_01_ARI_BUF"] * 3,
                       "targets": [3, 0, 0], "carries": [0, 5, 0]})
    df = clean(pl.DataFrame([opp()]))           # only p1 has expected points
    issues = check_ff_opportunity(df, pw)
    assert any("1 of 2" in i for i in issues)   # p2 carried the ball but is missing; p3 never touched it
