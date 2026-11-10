"""Usage / recent-form features (SCRUM-34). Each test is one rule we agreed on 2026-11-10."""

import polars as pl
import pytest

from src.features.usage import build_usage_features, check_usage_features

STATS = ["offense_pct", "target_share", "air_yards_share", "wopr", "targets", "carries", "receptions",
         "receiving_yards", "rushing_yards", "attempts", "passing_yards", "sacks_suffered",
         "total_fantasy_points_exp", "total_touchdown_exp", "fantasy_points_ppr",
         "rushing_epa", "receiving_epa", "passing_epa"]


def game(week, season=2023, player="A", team="BUF", left_injured=False, **stats):
    row = {"player_id": player, "player_display_name": player, "position": "WR", "season": season,
           "week": week, "game_id": f"{season}_{week:02d}_{team}", "team": team, "left_injured": left_injured}
    row.update({s: 0.0 for s in STATS})
    row.update(stats)
    return row


def build(rows):
    return build_usage_features(pl.DataFrame(rows))


def row(feats, week, season=2023, player="A"):
    return feats.filter((pl.col("week") == week) & (pl.col("season") == season)
                        & (pl.col("player_id") == player)).row(0, named=True)


def test_first_game_of_season_has_no_current_season_features():
    feats = build([game(1, targets=8), game(2, targets=4)])
    first = row(feats, 1)
    assert first["games_std"] == 0
    assert first["targets_l3"] is None and first["targets_std"] is None


def test_features_use_only_earlier_games():
    # The week-3 row must not see week 3's 20 targets (that's the game being predicted).
    feats = build([game(1, targets=2), game(2, targets=4), game(3, targets=20)])
    assert row(feats, 3)["targets_l3"] == 3.0
    assert row(feats, 3)["games_std"] == 2


def test_changing_the_current_game_does_not_change_its_features():
    # Leakage test: rewrite week 4 completely; week 4's features must stay identical.
    base = [game(w, targets=w, carries=1, rushing_yards=5) for w in range(1, 5)]
    changed = base[:3] + [game(4, targets=99, carries=30, rushing_yards=300)]
    assert row(build(base), 4) == row(build(changed), 4)


def test_last_3_window_drops_older_games():
    feats = build([game(w, targets=t) for w, t in [(1, 30), (2, 3), (3, 3), (4, 3), (5, 0)]])
    assert row(feats, 5)["targets_l3"] == 3.0        # weeks 2-4 only
    assert row(feats, 5)["targets_std"] == 9.75      # (30+3+3+3)/4


def test_bye_week_is_skipped_not_a_zero():
    # Week 3 is a bye (no row). The week-4 average is over 2 games, not 3.
    feats = build([game(1, targets=6), game(2, targets=6), game(4, targets=0)])
    assert row(feats, 4)["targets_l3"] == 6.0
    assert row(feats, 4)["games_std"] == 2


def test_left_injured_game_is_left_out_of_averages():
    feats = build([game(1, targets=10), game(2, targets=1, left_injured=True), game(3, targets=0)])
    assert row(feats, 3)["targets_std"] == 10.0
    assert row(feats, 3)["games_std"] == 1
    assert row(feats, 2)["targets_std"] == 10.0      # the injured game still gets its own features row


def test_previous_season_follows_into_next_season():
    feats = build([game(16, season=2022, targets=4), game(17, season=2022, targets=8, left_injured=True),
                   game(1, season=2023, targets=0)])
    week1 = row(feats, 1, season=2023)
    assert week1["targets_prev"] == 4.0               # left-injured week 17 excluded
    assert week1["games_prev"] == 1
    assert week1["targets_l3"] is None                # season windows don't carry over


def test_rookie_has_no_previous_season():
    week1 = row(build([game(1, targets=5)]), 1)
    assert week1["targets_prev"] is None and week1["games_prev"] == 0


def test_carry_share_is_share_of_team_carries():
    feats = build([game(1, player="A", carries=15), game(1, player="B", carries=5),
                   game(2, player="A"), game(2, player="B")])
    assert row(feats, 2, player="A")["carry_share_l3"] == 0.75
    assert row(feats, 2, player="B")["carry_share_l3"] == 0.25


def test_efficiency_is_total_over_total_not_average_of_ratios():
    # 1 carry for 10 yds, then 19 carries for 19 yds: 29/20 = 1.45, not (10 + 1)/2 = 5.5.
    feats = build([game(1, carries=1, rushing_yards=10), game(2, carries=19, rushing_yards=19), game(3)])
    assert row(feats, 3)["yards_per_touch_l3"] == pytest.approx(1.45)


def test_efficiency_is_null_with_no_touches():
    feats = build([game(1, offense_pct=0.1), game(2)])
    assert row(feats, 2)["yards_per_touch_l3"] is None


def test_check_flags_duplicate_player_weeks():
    pw = pl.DataFrame([game(1), game(1)])
    feats = build_usage_features(pw)
    assert any("duplicate" in p for p in check_usage_features(feats, pw))


def test_check_passes_on_clean_data():
    pw = pl.DataFrame([game(w, targets=w, target_share=0.2, offense_pct=0.8) for w in range(1, 6)])
    assert check_usage_features(build_usage_features(pw), pw) == []
