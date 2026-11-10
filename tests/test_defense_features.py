"""Defensive ratings (SCRUM-36). Each test is one rule we agreed on 2026-11-10."""

import polars as pl
import pytest

from src.features.defense import build_defense_features, defense_plays, fantasy_over_expected_allowed


def sched(week, home="DET", away="GB", season=2023):
    return {"game_id": f"{season}_{week:02d}_{away}_{home}", "season": season, "week": week,
            "home_team": home, "away_team": away}


def play(game_id, defteam="DET", play_type="pass", dropback=1, epa=0.0, wp=0.5, sack=0, qb_hit=0,
         pass_length="short"):
    return {"game_id": game_id, "season_type": "REG", "defteam": defteam, "play_type": play_type,
            "qb_dropback": dropback, "rush_attempt": int(play_type == "run"),
            "pass_attempt": int(play_type == "pass" and not sack), "pass_length": pass_length if play_type == "pass" else None,
            "epa": epa, "sack": sack, "qb_hit": qb_hit, "wp": wp}


def pw_row(game_id, position, actual, expected, defense="DET"):
    return {"game_id": game_id, "opponent_team": defense, "position": position,
            "fantasy_points_ppr": actual, "total_fantasy_points_exp": expected}


NO_FPOE = pl.DataFrame(schema={"game_id": pl.String, "defteam": pl.String, "fpoe_qb": pl.Float64,
                              "fpoe_rb": pl.Float64, "fpoe_wr": pl.Float64, "fpoe_te": pl.Float64})


def build(schedule_rows, plays, fpoe=NO_FPOE):
    return build_defense_features(pl.DataFrame(schedule_rows), defense_plays(pl.DataFrame(plays)), fpoe)


def det(feats, week, season=2023):
    return feats.filter((pl.col("defteam") == "DET") & (pl.col("week") == week) & (pl.col("season") == season)).row(0, named=True)


def test_ratings_use_only_earlier_games():
    s = [sched(1), sched(2)]
    feats = build(s, [play(s[0]["game_id"], epa=0.5), play(s[1]["game_id"], epa=-3.0)])
    assert det(feats, 1)["epa_per_dropback_std"] is None
    assert det(feats, 2)["epa_per_dropback_std"] == 0.5        # week 2's own -3.0 not included


def test_garbage_time_is_left_out():
    s = [sched(1), sched(2)]
    plays = [play(s[0]["game_id"], epa=0.2, wp=0.50),
             play(s[0]["game_id"], epa=5.0, wp=0.02),          # down big: garbage time
             play(s[0]["game_id"], epa=5.0, wp=0.97)]          # up big: garbage time
    assert det(build(s, plays), 2)["epa_per_dropback_std"] == pytest.approx(0.2)


def test_rates_are_total_over_total():
    # Game 1: 1 sack on 10 dropbacks; game 2: 0 on 30. Rate = 1/40, not the average of 10% and 0%.
    s = [sched(1), sched(2), sched(3)]
    plays = ([play(s[0]["game_id"], sack=1)] + [play(s[0]["game_id"])] * 9 + [play(s[1]["game_id"])] * 30)
    assert det(build(s, plays), 3)["sack_rate_std"] == pytest.approx(1 / 40)


def test_designed_runs_and_scrambles_are_separated():
    s = [sched(1), sched(2)]
    plays = [play(s[0]["game_id"], play_type="run", dropback=0, epa=-0.4),
             play(s[0]["game_id"], play_type="run", dropback=1, epa=2.0)]   # QB scramble = a dropback
    week2 = det(build(s, plays), 2)
    assert week2["epa_per_rush_std"] == pytest.approx(-0.4)
    assert week2["epa_per_dropback_std"] == pytest.approx(2.0)


def test_short_and_deep_passes_are_rated_separately():
    s = [sched(1), sched(2)]
    plays = [play(s[0]["game_id"], epa=0.1, pass_length="short"), play(s[0]["game_id"], epa=1.5, pass_length="deep")]
    week2 = det(build(s, plays), 2)
    assert week2["epa_short_std"] == pytest.approx(0.1) and week2["epa_deep_std"] == pytest.approx(1.5)


def test_fantasy_points_above_expected_by_position():
    s = [sched(1), sched(2)]
    pw = pl.DataFrame([pw_row(s[0]["game_id"], "WR", 20.0, 12.0), pw_row(s[0]["game_id"], "WR", 3.0, 5.0),
                       pw_row(s[0]["game_id"], "RB", 10.0, 10.0)])
    week2 = det(build(s, [play(s[0]["game_id"])], fantasy_over_expected_allowed(pw)), 2)
    assert week2["fpoe_wr_std"] == pytest.approx(6.0)          # (20-12) + (3-5)
    assert week2["fpoe_rb_std"] == pytest.approx(0.0)


def test_upcoming_games_still_get_a_rating():
    s = [sched(1), sched(2)]                                     # week 2 not played yet: no plays
    week2 = det(build(s, [play(s[0]["game_id"], epa=0.3)]), 2)
    assert week2["epa_per_dropback_std"] == pytest.approx(0.3)


def test_previous_season_carries_into_week_1():
    s = [sched(17, season=2022), sched(1, season=2023)]
    week1 = det(build(s, [play(s[0]["game_id"], epa=-0.25)]), 1)
    assert week1["epa_per_dropback_prev"] == pytest.approx(-0.25)
    assert week1["epa_per_dropback_l5"] is None


def test_each_defense_gets_its_own_rating():
    s = [sched(1), sched(2)]
    plays = [play(s[0]["game_id"], defteam="DET", epa=-1.0), play(s[0]["game_id"], defteam="GB", epa=1.0)]
    feats = build(s, plays)
    gb = feats.filter((pl.col("defteam") == "GB") & (pl.col("week") == 2)).row(0, named=True)
    assert det(feats, 2)["epa_per_dropback_std"] == -1.0 and gb["epa_per_dropback_std"] == 1.0
