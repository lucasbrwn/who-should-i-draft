"""Scoring engine (SCRUM-30/32) and optimal lineup (SCRUM-31)."""

import itertools
import random
from dataclasses import fields, replace

import polars as pl
import pytest

from src.scoring.config import PRESETS, Roster, Scoring
from src.scoring.engine import add_fantasy_points, points_expr, unhandled_settings
from src.scoring.lineup import Player, lineup_points, optimal_lineup

PPR = PRESETS["standard-ppr"].scoring


def score(scoring: Scoring = PPR, **stats) -> float:
    return add_fantasy_points(pl.DataFrame([stats]), scoring)["points"].item()


# --- Scoring engine -------------------------------------------------------------------------

def test_every_scoring_setting_is_handled():
    assert unhandled_settings() == set()


@pytest.mark.parametrize("stat, value, expected", [
    ("passing_yards", 250, 10.0),         # 0.04 per yard
    ("passing_tds", 2, 8.0),
    ("passing_interceptions", 1, -2.0),
    ("passing_2pt_conversions", 1, 2.0),
    ("rushing_yards", 85, 8.5),
    ("rushing_tds", 1, 6.0),
    ("rushing_2pt_conversions", 1, 2.0),
    ("receptions", 6, 6.0),
    ("receiving_yards", 73, 7.3),
    ("receiving_tds", 2, 12.0),
    ("receiving_2pt_conversions", 1, 2.0),
    ("special_teams_tds", 1, 6.0),
    ("sack_fumbles_lost", 1, -2.0),
    ("rushing_fumbles_lost", 1, -2.0),
    ("receiving_fumbles_lost", 1, -2.0),
])
def test_each_stat_under_default_ppr(stat, value, expected):
    assert score(**{stat: value}) == pytest.approx(expected)


@pytest.mark.parametrize("preset, expected", [
    ("standard-ppr", 6 + 8.0 + 6),     # 6 rec, 80 yds, 1 TD
    ("half-ppr", 3 + 8.0 + 6),
    ("standard", 0 + 8.0 + 6),
])
def test_ppr_variants(preset, expected):
    assert score(PRESETS[preset].scoring, receptions=6, receiving_yards=80, receiving_tds=1) == pytest.approx(expected)


def test_six_point_passing_tds():
    six = replace(PPR, passing_tds=6.0)
    assert score(six, passing_tds=3) == 18.0


def test_te_premium_only_applies_to_tes():
    tep = replace(PPR, te_reception_bonus=0.5)
    assert score(tep, position="TE", receptions=4) == 6.0
    assert score(tep, position="WR", receptions=4) == 4.0


def test_te_premium_without_position_column_is_an_error():
    with pytest.raises(ValueError, match="position"):
        score(replace(PPR, te_reception_bonus=0.5), receptions=4)


@pytest.mark.parametrize("setting, stat, threshold", [
    ("bonus_pass_300", "passing_yards", 300),
    ("bonus_rush_100", "rushing_yards", 100),
    ("bonus_rec_100", "receiving_yards", 100),
])
def test_bonuses_trigger_at_threshold_not_below(setting, stat, threshold):
    bonus = replace(Scoring(), **{setting: 3.0})
    base = replace(Scoring())
    at = score(bonus, **{stat: threshold}) - score(base, **{stat: threshold})
    below = score(bonus, **{stat: threshold - 1}) - score(base, **{stat: threshold - 1})
    assert at == pytest.approx(3.0)
    assert below == pytest.approx(0.0)


def test_fumbles_lost_total_column_is_used_instead_of_parts():
    # A predicted-stats table might carry one total; it must not double count with the parts.
    assert score(fumbles_lost=1, rushing_fumbles_lost=1) == -2.0


def test_return_fumbles_count_as_lost():
    # Skyy Moore 2022 wk 3: muffed punt, recovered by IND. nflverse puts it only in the total.
    # ESPN shows "1 FUM 1 LOST", so the -2 must apply even though every scrimmage part is 0.
    moore = dict(punt_returns=2, punt_return_yards=12, sack_fumbles_lost=0, rushing_fumbles_lost=0,
                 receiving_fumbles_lost=0, fumbles_lost_total=1, fumbles_total=1)
    assert score(**moore) == -2.0


def test_return_yards_score_only_when_the_league_turns_them_on():
    # Britain Covey 2025 wk 14: 54 punt + 103 kick return yards, 0 fantasy points by default
    covey = dict(punt_return_yards=54, kickoff_return_yards=103)
    assert score(**covey) == 0.0
    returns = replace(PPR, punt_return_yards=0.04, kickoff_return_yards=0.04)   # 1 pt per 25 yds
    assert score(returns, **covey) == pytest.approx(157 * 0.04)


def test_platform_presets():
    assert PRESETS["espn-ppr"].scoring == PRESETS["standard-ppr"].scoring
    sleeper = PRESETS["sleeper-ppr"].scoring
    assert sleeper.passing_interceptions == -1.0
    assert replace(sleeper, passing_interceptions=-2.0) == PRESETS["standard-ppr"].scoring


def test_recovered_fumbles_cost_nothing_by_default():
    # 2 fumbles, offense recovered 1: only the lost one counts (Lucas's leagues)
    assert score(rushing_fumbles=2, rushing_fumbles_lost=1) == -2.0


def test_any_fumble_setting_stacks_with_fumble_lost():
    # Sleeper-style: -1 per fumble plus -2 per lost fumble. Josh Allen 2022 wk 3: 3 fumbles, 1 lost.
    sleeper_like = replace(PPR, fumbles=-1.0)
    assert score(sleeper_like, sack_fumbles=2, sack_fumbles_lost=1, rushing_fumbles=1) == -5.0


def test_missing_columns_count_as_zero():
    # e.g. a model that only predicts rushing
    assert score(rushing_yards=100, rushing_tds=1) == 16.0


def test_nulls_count_as_zero():
    df = pl.DataFrame({"rushing_yards": [None, 50.0], "rushing_tds": [1, None]})
    assert add_fantasy_points(df, PPR)["points"].to_list() == [6.0, 5.0]


def test_fractional_predicted_stats():
    # M4 predicts expected values like 0.6 TDs
    assert score(receiving_tds=0.6, receptions=4.5, receiving_yards=52.3) == pytest.approx(3.6 + 4.5 + 5.23)


def test_empty_table():
    df = pl.DataFrame(schema={"passing_yards": pl.Float64})
    assert add_fantasy_points(df, PPR).height == 0


def test_points_expr_works_inside_a_lazy_query():
    lf = pl.LazyFrame({"rushing_yards": [100.0]})
    assert lf.select(points_expr(PPR, lf.collect_schema().names())).collect().item() == 10.0


# --- Optimal lineup -------------------------------------------------------------------------

def brute_force_best(players: list[Player], roster: Roster) -> float:
    """Try every assignment of players to slots. Only for tiny rosters."""
    from src.scoring.config import SLOT_ELIGIBILITY
    from src.scoring.lineup import FILL_ORDER

    slots = [s for s in FILL_ORDER for _ in range(getattr(roster, s))]
    best = 0.0
    for size in range(1, min(len(players), len(slots)) + 1):   # some players may not fit at all
        for order in itertools.permutations(players, size):
            for chosen_slots in itertools.combinations(range(len(slots)), size):
                if all(p.position in SLOT_ELIGIBILITY[slots[i]] for p, i in zip(order, chosen_slots)):
                    best = max(best, sum(p.points for p in order))
    return best


def test_standard_lineup():
    players = [
        Player("qb1", "QB", 22), Player("qb2", "QB", 18),
        Player("rb1", "RB", 15), Player("rb2", "RB", 12), Player("rb3", "RB", 11),
        Player("wr1", "WR", 14), Player("wr2", "WR", 9), Player("wr3", "WR", 10.5),
        Player("te1", "TE", 8), Player("k1", "K", 7), Player("dst1", "DST", 6),
    ]
    lineup = optimal_lineup(players, PRESETS["standard-ppr"].roster)
    ids = {slot: [p.id for p in ps] for slot, ps in lineup.items()}
    assert ids["QB"] == ["qb1"]
    assert ids["RB"] == ["rb1", "rb2"]
    assert ids["WR"] == ["wr1", "wr3"]
    assert ids["FLEX"] == ["rb3"]          # 11 beats wr2's 9
    assert ids["SUPERFLEX"] == []


def test_superflex_takes_a_second_qb():
    players = [Player("qb1", "QB", 25), Player("qb2", "QB", 20), Player("rb1", "RB", 10),
               Player("rb2", "RB", 9), Player("rb3", "RB", 5)]
    roster = Roster(QB=1, RB=2, WR=0, TE=0, FLEX=0, SUPERFLEX=1, K=0, DST=0)
    assert [p.id for p in optimal_lineup(players, roster)["SUPERFLEX"]] == ["qb2"]


def test_two_flex():
    players = [Player("rb1", "RB", 10), Player("rb2", "RB", 8), Player("wr1", "WR", 9),
               Player("te1", "TE", 7), Player("te2", "TE", 6)]
    roster = Roster(QB=0, RB=1, WR=1, TE=1, FLEX=2, SUPERFLEX=0, K=0, DST=0)
    assert lineup_points(players, roster) == 10 + 9 + 7 + 8 + 6


def test_thin_roster_leaves_slots_empty():
    roster = PRESETS["standard-ppr"].roster
    lineup = optimal_lineup([Player("rb1", "RB", 10)], roster)
    assert [p.id for p in lineup["RB"]] == ["rb1"]
    assert lineup["FLEX"] == [] and lineup["QB"] == []


def test_positions_without_a_slot_stay_on_the_bench():
    roster = Roster(QB=1, RB=0, WR=0, TE=0, FLEX=0, K=0, DST=0)
    assert lineup_points([Player("k1", "K", 9), Player("qb1", "QB", 3)], roster) == 3


def test_tied_points_fill_every_slot():
    roster = Roster(QB=0, RB=2, WR=0, TE=0, FLEX=1, K=0, DST=0)
    players = [Player(f"rb{i}", "RB", 10) for i in range(4)]
    lineup = optimal_lineup(players, roster)
    used = [p.id for ps in lineup.values() for p in ps]
    assert len(used) == 3 and len(set(used)) == 3   # nobody starts twice


def test_idp_is_rejected():
    with pytest.raises(ValueError, match="IDP"):
        optimal_lineup([], Roster(IDP=1))


@pytest.mark.parametrize("seed", range(40))
def test_greedy_matches_brute_force(seed):
    rng = random.Random(seed)
    roster = Roster(QB=rng.randint(0, 1), RB=rng.randint(0, 2), WR=rng.randint(0, 1),
                    TE=rng.randint(0, 1), FLEX=rng.randint(0, 1), SUPERFLEX=rng.randint(0, 1),
                    K=0, DST=0)
    players = [Player(f"p{i}", rng.choice(["QB", "RB", "WR", "TE"]), round(rng.uniform(0, 30), 1))
               for i in range(rng.randint(1, 6))]
    assert lineup_points(players, roster) == pytest.approx(brute_force_best(players, roster))
