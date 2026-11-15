"""Availability model (SCRUM-51): what counts as an injury miss, who's covered, model safety."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from src.models.availability import Poisson, PositionAverage, injury_weeks, season_table, walk_forward


def schedules(season, weeks, team="DET", opp="GB"):
    return pl.DataFrame([{"season": season, "week": w, "game_type": "REG", "home_team": team, "away_team": opp}
                         for w in weeks])


def roster(player, season, week, status="ACT", code=None, team="DET", position="WR"):
    return {"season": season, "week": week, "game_type": "REG", "team": team, "position": position,
            "gsis_id": player, "status": status, "status_description_abbr": code, "birth_date": date(1998, 1, 1)}


def report(player, season, week, status):
    return {"season": season, "week": week, "game_type": "REG", "gsis_id": player, "report_status": status}


def played(player, season, week, ppr=10.0):
    return {"player_id": player, "season": season, "week": week, "fantasy_points_ppr": ppr}


def weeks_for(rosters, reports, games, plays):
    inj = pl.DataFrame(reports, schema={"season": pl.Int64, "week": pl.Int64, "game_type": pl.String,
                                        "gsis_id": pl.String, "report_status": pl.String})
    pw = pl.DataFrame(plays, schema={"player_id": pl.String, "season": pl.Int64, "week": pl.Int64,
                                     "fantasy_points_ppr": pl.Float64})
    return injury_weeks(pl.DataFrame(rosters), inj, games, pw)


def miss(weeks, player, week):
    return weeks.filter(pl.col("player_id") == player, pl.col("week") == week)["injury_miss"].item()


def test_what_counts_as_an_injury_miss():
    rosters = [
        roster("ir", 2024, 1, "RES", "R01"),        # injured reserve, didn't play -> miss
        roster("susp", 2024, 1, "RES", "R40"),      # suspended -> not an injury
        roster("covid", 2024, 1, "RES", "R59"),     # COVID list -> not an injury
        roster("out", 2024, 1, "INA"),              # inactive and reported Out -> miss
        roster("benched", 2024, 1, "ACT"),          # active, didn't play, no report -> benching, not a miss
        roster("played", 2024, 1, "ACT"),           # Doubtful but played -> not a miss
    ]
    reports = [report("out", 2024, 1, "Out"), report("played", 2024, 1, "Doubtful")]
    w = weeks_for(rosters, reports, schedules(2024, [1]), [played("played", 2024, 1)])
    assert [miss(w, p, 1) for p in ["ir", "susp", "covid", "out", "benched", "played"]] == \
           [True, False, False, True, False, False]


def test_bye_weeks_are_not_counted():
    rosters = [roster("ir", 2024, wk, "RES", "R01") for wk in (1, 2, 3)]
    w = weeks_for(rosters, [], schedules(2024, [1, 3]), [])  # week 2 is the bye
    assert sorted(w["week"].to_list()) == [1, 3]


def test_population_needs_a_relevant_last_season_and_a_week_1_roster_spot():
    games = pl.concat([schedules(2023, range(1, 6)), schedules(2024, range(1, 6))])
    # 2023: all three play 5 games. star and gone score 15/game (relevant), scrub 3/game (not).
    rosters = [roster(p, 2023, wk) for p in ("star", "scrub", "gone") for wk in range(1, 6)]
    plays = [played(p, 2023, wk, ppr) for p, ppr in [("star", 15), ("scrub", 3), ("gone", 15)] for wk in range(1, 6)]
    # 2024: gone is off every roster. scrub is rostered. star plays weeks 1-3, then goes on IR.
    rosters += [roster("scrub", 2024, wk) for wk in range(1, 6)]
    rosters += [roster("star", 2024, wk) for wk in (1, 2, 3)] + [roster("star", 2024, wk, "RES", "R01") for wk in (4, 5)]
    plays += [played("star", 2024, wk, 15) for wk in (1, 2, 3)]

    t = season_table(weeks_for(rosters, [], games, plays), pl.DataFrame(plays)).filter(pl.col("season") == 2024)
    assert t["player_id"].to_list() == ["star"]  # scrub wasn't fantasy-relevant; gone has no 2024 roster spot
    assert t["injury_missed"].item() == 2 and t["prev_missed"].item() == 0
    assert 26 < t["age"].item() < 27  # born 1998-01-01, measured on 2024-09-01


def table(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "player_id": [f"p{i}" for i in range(n)] * 2, "season": [2021] * n + [2022] * n,
        "position": (["QB", "RB", "WR", "TE"] * n)[: 2 * n],
        "injury_missed": rng.poisson(2.5, 2 * n).astype(float), "prev_missed": rng.poisson(2, 2 * n).astype(float),
        "prev2_missed": [None] * n + list(rng.poisson(2, n).astype(float)),
        "has_prev2": [False] * n + [True] * n, "age": rng.uniform(22, 32, 2 * n),
    })


def test_poisson_ignores_a_column_with_no_spread_in_training():
    # Training on 2021 only: prev2 is the same filler for everyone. 2022's real prev2 values
    # must not blow predictions up (the bug that once predicted 17 missed games).
    t = table()
    model = Poisson().fit(t.filter(pl.col("season") == 2021))
    test = t.filter(pl.col("season") == 2022).with_columns(prev2_missed=pl.lit(15.0))
    assert model.predict(test).max() < 8


def test_position_average_is_per_position():
    t = pl.DataFrame({"position": ["QB", "QB", "RB"], "injury_missed": [1.0, 3.0, 5.0]})
    assert PositionAverage().fit(t).predict(pl.DataFrame({"position": ["RB", "QB", "K"]})).tolist() == [5.0, 2.0, 0.0]


def test_2025_is_off_limits():
    with pytest.raises(ValueError, match="final exam"):
        walk_forward(table(), seasons=[2025])
