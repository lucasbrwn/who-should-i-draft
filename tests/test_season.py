"""Season projections (SCRUM-53)."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.models.season import (
    FEATURES_PATH, SeasonModel, add_placeholder_rows, load_a, neutralize, preseason_rows, season_dataset, week1_players,
)
from src.ingest.in_game_injuries import rosters_path
from src.features.team import read_schedules

needs_data = pytest.mark.skipif(not FEATURES_PATH.exists(), reason="needs the real data (run python -m src.features)")


def test_placeholder_rows_only_for_players_without_a_week1_game():
    pw = pl.DataFrame({"player_id": ["a"], "season": [2024], "week": [1], "team": ["DET"], "game_id": ["g1"],
                       "gameday": ["2024-09-08"], "opponent_team": ["LA"], "is_home": [True],
                       "left_injured": [False], "targets": [7.0]})
    players = pl.DataFrame({"player_id": ["a", "b", "c"], "player_display_name": ["A", "B", "C"],
                            "position": ["WR"] * 3, "team": ["DET", "DET", "KC"], "draft_number": [1.0, 2.0, 3.0]})
    sched = pl.DataFrame({"season": [2024], "week": [1], "game_type": ["REG"], "game_id": ["g1"],
                          "gameday": ["2024-09-08"], "home_team": ["DET"], "away_team": ["LA"]})
    out = add_placeholder_rows(pw, players, sched, 2024).sort("player_id")
    assert out["player_id"].to_list() == ["a", "b"]  # c's team (KC) has a Week 1 bye here: no row
    assert out.filter(pl.col("player_id") == "b")["targets"].item() is None  # blank stats, never invented
    assert out.filter(pl.col("player_id") == "b")["game_id"].item() == "g1"


def test_neutralize_copies_a_typical_week1_row():
    train = pl.DataFrame({"week": [1, 1, 5], "opp_pass_epa_l5": [None, None, 0.3], "opp_pass_epa_prev": [0.1, 0.3, 9.0],
                          "is_home": [True, False, True], "targets_l3": [None, None, 5.0]})
    rows = pl.DataFrame({"player_id": ["x"], "season": [2024], "week": [1], "opp_pass_epa_l5": [None],
                         "opp_pass_epa_prev": [0.9], "is_home": [True], "targets_l3": [None]}, schema_overrides={"opp_pass_epa_l5": pl.Float64, "targets_l3": pl.Float64})
    out = neutralize(rows, train)
    assert out["opp_pass_epa_l5"].item() is None             # blank in Week 1 training rows -> stays blank
    assert out["opp_pass_epa_prev"].item() == pytest.approx(0.2)  # Week 1 average only (not the week-5 9.0)
    assert out["is_home"].item() == pytest.approx(0.5)       # a number, not True
    assert out["targets_l3"].item() is None                   # player inputs untouched


def test_season_target_is_zero_when_he_never_played_and_lags_line_up():
    a_rows = pl.DataFrame({"player_id": ["star", "backup"], "player_display_name": ["S", "B"], "position": ["QB", "QB"],
                           "team": ["BUF", "BUF"], "season": [2024, 2024], "draft_number": [7.0, None],
                           "a_ppg": [20.0, 15.0], "a_exp_games": [15.0, 15.0], "a_season": [300.0, 225.0]})
    from src.models.season import ROW_INPUTS
    for c in ROW_INPUTS:
        a_rows = a_rows.with_columns(pl.lit(0.0).alias(c))
    pw = pl.DataFrame({"player_id": ["star"] * 3 + ["backup"], "season": [2023, 2023, 2024, 2023],
                       "week": [1, 2, 1, 3], "fantasy_points_ppr": [20.0, 30.0, 22.0, 4.0]})
    sched = pl.DataFrame({"season": [2024], "week": [1], "game_type": ["REG"], "home_qb_id": ["star"], "away_qb_id": ["x"]})
    d = season_dataset(a_rows, pw, sched).sort("player_id")
    assert d["actual"].to_list() == [0.0, 22.0]       # backup never played in 2024 -> 0, not dropped
    assert d["ppg_last"].to_list() == [4.0, 25.0] and d["games_last"].to_list() == [1, 2]
    assert d["week1_starter"].to_list() == [0.0, 1.0]
    assert d["naive"].to_list() == [68.0, 425.0]


def test_season_model_learns_and_never_goes_negative():
    rng = np.random.default_rng(0)
    n = 400
    df = pl.DataFrame({"position": rng.choice(["QB", "RB", "WR", "TE"], n), "a_ppg": rng.uniform(0, 20, n),
                       "games_last": rng.integers(0, 18, n).astype(float)})
    df = df.with_columns(actual=pl.col("a_ppg") * pl.col("games_last"), player_id=pl.lit("p"), season=pl.lit(2022))
    pred = SeasonModel().fit(df).predict(df)
    assert pred.min() >= 0
    assert np.corrcoef(pred, df["actual"].to_numpy())[0, 1] > 0.9


def test_2025_is_off_limits():
    with pytest.raises(ValueError, match="final exam"):
        load_a(2025)


@needs_data
def test_preseason_rows_match_the_real_week1_rows():
    players = week1_players(pl.read_parquet(rosters_path(2023)), 2023)
    rows = preseason_rows(2023, players)
    real = pl.read_parquet(FEATURES_PATH).filter(pl.col("season") == 2023, pl.col("week") == 1)
    both = rows.join(real, on="player_id", suffix="_real")
    assert both.height > 300
    for c in [c for c in real.columns if c not in ("player_id",) and not c.startswith("y_")]:
        a, b = both[c], both[f"{c}_real"]
        same = (a == b) | (a.is_null() & b.is_null())
        if a.dtype.is_float():
            same = same | ((a - b).abs() < 1e-9)
        assert same.all(), c
    assert rows.height > both.height  # and players with no Week 1 game got rows too


@pytest.mark.skipif(not Path("data/reference/adp/fantasypros_adp_2024.csv").exists(), reason="needs ADP files")
def test_adp_inputs_rank_within_position_from_the_pre_draft_file():
    from src.models.season import adp_inputs
    a = adp_inputs([2024])
    assert a.height > 300 and a["season"].unique().to_list() == [2024]
    top = a.sort("adp").head(1)
    assert top["adp_pos_rank"].item() == 1          # the overall #1 pick is #1 at his position
    assert a["adp_pos_rank"].min() == 1 and a["player_id"].is_unique().all()
