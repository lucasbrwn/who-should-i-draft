import polars as pl

from src.ingest.checks import check_player_stats, standard_fantasy_points
from src.ingest.crosscheck import STATS, compare, stats_from_pbp
from src.ingest.ids import normalize_name

ZERO_STATS = {c: 0 for c in [
    "passing_yards", "passing_tds", "passing_interceptions", "rushing_yards", "rushing_tds",
    "receiving_yards", "receiving_tds", "special_teams_tds", "passing_2pt_conversions",
    "rushing_2pt_conversions", "receiving_2pt_conversions", "sack_fumbles_lost",
    "rushing_fumbles_lost", "receiving_fumbles_lost", "receptions", "targets",
    "completions", "attempts",
]}


def player_row(**overrides):
    row = {
        "player_id": "P1", "game_id": "2024_01_AAA_BBB", "season_type": "REG", "week": 1,
        "team": "AAA", "opponent_team": "BBB", **ZERO_STATS,
        "fantasy_points": 0.0, "fantasy_points_ppr": 0.0,
    }
    row.update(overrides)
    return row


def test_standard_fantasy_points_ppr_and_standard():
    df = pl.DataFrame([player_row(receptions=5, targets=7, receiving_yards=80, receiving_tds=1, rushing_fumbles_lost=1)])
    # 80 * 0.1 + 6 - 2 = 12 standard; +5 receptions in PPR
    assert standard_fantasy_points(df, ppr=0.0).item() == 12.0
    assert standard_fantasy_points(df, ppr=1.0).item() == 17.0


def test_clean_data_has_no_issues():
    df = pl.DataFrame([player_row(receptions=2, targets=3, receiving_yards=20, fantasy_points=2.0, fantasy_points_ppr=4.0)])
    assert check_player_stats(df, 2024) == []


def test_checks_flag_bad_rows():
    df = pl.DataFrame([
        player_row(receptions=4, targets=3, receiving_yards=0, fantasy_points=0.0, fantasy_points_ppr=4.0),
        player_row(receptions=4, targets=3, receiving_yards=0, fantasy_points=0.0, fantasy_points_ppr=4.0),
        player_row(player_id="P2", opponent_team=None, week=19, fantasy_points=99.0),
    ])
    issues = " | ".join(check_player_stats(df, 2024))
    assert "duplicate" in issues
    assert "missing opponent_team" in issues
    assert "exceeds 18-week" in issues
    assert "receptions > targets" in issues
    assert "fantasy_points doesn't match" in issues


def pbp_play(**overrides):
    play = {
        "game_id": "G1", "play_type": "pass", "two_point_attempt": 0, "pass_attempt": 1,
        "complete_pass": 0, "sack": 0, "interception": 0, "rush_attempt": 0,
        "passer_player_id": "QB", "receiver_player_id": "WR", "rusher_player_id": None,
        "passing_yards": None, "receiving_yards": None, "rushing_yards": None,
        "pass_touchdown": 0, "rush_touchdown": 0, "td_player_id": None,
    }
    play.update(overrides)
    return play


def test_stats_from_pbp_counts_plays_and_skips_two_point_tries():
    pbp = pl.DataFrame([
        pbp_play(complete_pass=1, passing_yards=25, receiving_yards=25, pass_touchdown=1, td_player_id="WR"),
        pbp_play(),  # incompletion: attempt + target
        pbp_play(sack=1, receiver_player_id=None),  # sack: not an attempt
        pbp_play(play_type="run", pass_attempt=0, rush_attempt=1, passer_player_id=None,
                 receiver_player_id=None, rusher_player_id="RB", rushing_yards=7),
        pbp_play(two_point_attempt=1, complete_pass=1),  # ignored
    ])
    out = {r["player_id"]: r for r in stats_from_pbp(pbp).to_dicts()}
    assert out["QB"]["attempts"] == 2 and out["QB"]["completions"] == 1
    assert out["QB"]["passing_yards"] == 25 and out["QB"]["passing_tds"] == 1
    assert out["WR"]["targets"] == 2 and out["WR"]["receptions"] == 1 and out["WR"]["receiving_tds"] == 1
    assert out["RB"]["carries"] == 1 and out["RB"]["rushing_yards"] == 7


def test_compare_reports_mismatches():
    weekly = pl.DataFrame([{"game_id": "G1", "player_id": "WR", "player_display_name": "W", "position": "WR",
                            **{s: 0 for s in STATS}, "targets": 2, "receptions": 1, "receiving_yards": 30}])
    rebuilt = pl.DataFrame([{"game_id": "G1", "player_id": "WR", **{s: 0 for s in STATS},
                             "targets": 2, "receptions": 1, "receiving_yards": 25}])
    summary, mismatches = compare(weekly, rebuilt)
    by_stat = {r["stat"]: r for r in summary.to_dicts()}
    assert by_stat["receiving_yards"]["mismatches"] == 1
    assert by_stat["targets"]["mismatches"] == 0
    assert mismatches.height == 1


def test_normalize_name_handles_suffixes_and_punctuation():
    names = pl.Series(["Brian Thomas Jr.", "C.J. Stroud", "Ja'Marr Chase", "Amon-Ra St. Brown"])
    assert pl.select(normalize_name(pl.lit(names))).to_series().to_list() == [
        "brian thomas", "cj stroud", "jamarr chase", "amonra st brown",
    ]
