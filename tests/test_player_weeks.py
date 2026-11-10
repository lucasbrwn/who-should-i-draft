"""Player-week master table (SCRUM-22). Each test is one rule we decided on 2026-11-03."""

import polars as pl

from src.ingest.player_weeks import build_player_weeks, check_player_weeks

GAME = "2023_11_NYJ_BUF"


def stat_row(player_id, name="P", position="WR", game_id=GAME, season_type="REG", week=11, team="BUF",
             opponent="NYJ", **stats):
    row = {"player_id": player_id, "player_name": name, "player_display_name": name, "position": position,
           "position_group": position, "headshot_url": None, "season": 2023, "week": week,
           "season_type": season_type, "game_id": game_id, "team": team, "opponent_team": opponent,
           "targets": 0, "receptions": 0, "receiving_yards": 0, "racr": None, "target_share": None,
           "def_tackles_solo": 0}
    row.update(stats)
    return row


def snap_row(gsis_id, name="P", position="WR", game_id=GAME, game_type="REG", week=11, team="BUF",
             opponent="NYJ", offense_snaps=50, offense_pct=0.8):
    return {"game_id": game_id, "game_type": game_type, "season": 2023, "week": week, "player": name,
            "position": position, "team": team, "opponent": opponent, "offense_snaps": offense_snaps,
            "offense_pct": offense_pct, "st_snaps": 0, "st_pct": 0.0, "gsis_id": gsis_id}


SCHEDULES = pl.DataFrame([
    {"game_id": GAME, "game_type": "REG", "gameday": "2023-11-19", "home_team": "BUF"},
    {"game_id": "2023_22_BUF_KC", "game_type": "POST", "gameday": "2024-01-21", "home_team": "KC"},
])


SNAP_SCHEMA = pl.DataFrame([snap_row("x")]).schema


def build(stats, snaps):
    stats_df = pl.DataFrame(stats, schema_overrides={"racr": pl.Float64, "target_share": pl.Float64})
    return build_player_weeks(stats_df, pl.DataFrame(snaps, schema=SNAP_SCHEMA), SCHEDULES)


def test_played_but_no_stats_gets_a_zero_row():
    # Gabriel Davis 2023 wk 11: 88% of snaps, 0 catches, no nflverse stats row
    df = build([stat_row("other", name="Someone")], [snap_row("davis", "Gabriel Davis", offense_pct=0.88)])
    davis = df.filter(pl.col("player_id") == "davis").to_dicts()[0]
    assert davis["snaps_only"] is True
    assert davis["receptions"] == 0 and davis["targets"] == 0
    assert davis["offense_pct"] == 0.88


def test_ratio_stats_stay_null_when_there_were_no_plays():
    df = build([stat_row("other")], [snap_row("davis")])
    assert df.filter(pl.col("player_id") == "davis")["racr"].item() is None


def test_stats_without_a_snap_row_are_kept_with_zero_snaps():
    df = build([stat_row("trick", receptions=1, targets=1)], [])
    row = df.to_dicts()[0]
    assert row["receptions"] == 1 and row["offense_snaps"] == 0 and row["snaps_only"] is False


def test_no_snaps_and_no_stats_means_no_row():
    # bye, inactive, or special-teams only: never a zero row
    df = build([stat_row("a")], [snap_row("st_only", offense_snaps=0, offense_pct=0.0)])
    assert "st_only" not in df["player_id"].to_list()


def test_playoff_games_are_left_out():
    df = build([stat_row("a"), stat_row("a", game_id="2023_22_BUF_KC", season_type="POST", week=22, opponent="KC")],
               [snap_row("b", game_id="2023_22_BUF_KC", game_type="POST", week=22, opponent="KC")])
    assert df["game_id"].to_list() == [GAME]


def test_non_offensive_positions_are_left_out():
    df = build([stat_row("lb", position="LB")], [snap_row("k", position="K")])
    assert df.height == 0


def test_one_player_one_position():
    # Taysom Hill-style: listed TE twice, QB once -> TE every week
    stats = [stat_row("hill", position="TE", week=w, game_id=f"g{w}") for w in (1, 2)]
    stats.append(stat_row("hill", position="QB", week=3, game_id="g3"))
    df = build(stats, [])
    assert set(df["position"]) == {"TE"}


def test_home_away_and_opponent():
    df = build([stat_row("bills", team="BUF", opponent="NYJ"), stat_row("jets", team="NYJ", opponent="BUF")], [])
    by_team = {r["team"]: r for r in df.to_dicts()}
    assert by_team["BUF"]["is_home"] is True and by_team["NYJ"]["is_home"] is False
    assert by_team["NYJ"]["opponent_team"] == "BUF"


def test_expected_points_attach_and_snap_only_games_expect_zero():
    opportunity = pl.DataFrame({"player_id": ["a"], "game_id": [GAME], "rec_touchdown_exp": [0.4],
                                "rec_touchdown_diff": [0.6], "rec_attempt_team": [35.0]})
    df = build_player_weeks(
        pl.DataFrame([stat_row("a", targets=5)], schema_overrides={"racr": pl.Float64, "target_share": pl.Float64}),
        pl.DataFrame([snap_row("davis")], schema=SNAP_SCHEMA), SCHEDULES, opportunity=opportunity,
    )
    by_id = {r["player_id"]: r for r in df.to_dicts()}
    assert by_id["a"]["rec_touchdown_exp"] == 0.4
    assert by_id["davis"]["rec_touchdown_exp"] == 0.0           # no opportunities -> nothing expected
    assert "rec_touchdown_diff" not in df.columns and "rec_attempt_team" not in df.columns


def test_check_finds_duplicates_and_missing_opponents():
    df = build([stat_row("a")], [])
    bad = pl.concat([df, df.with_columns(pl.lit(None, pl.String).alias("opponent_team"))])
    issues = check_player_weeks(bad)
    assert any("duplicate" in i for i in issues)
    assert any("opponent_team" in i for i in issues)


def test_clean_table_passes_checks():
    assert check_player_weeks(build([stat_row("a")], [snap_row("b")])) == []
