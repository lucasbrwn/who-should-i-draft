"""fp_ecr must be pre-game: never present for a game played on or before the snapshot day."""

from pathlib import Path

import polars as pl
import pytest

from src.backtest.weekly_vs_consensus import game_days, weekly_ecr
from src.features.expert import build_expert_features

pytestmark = pytest.mark.skipif(not Path("data/raw/ff_rankings_all.parquet").exists(), reason="needs real data")


def test_no_rank_for_games_on_or_before_the_snapshot():
    f = build_expert_features([2023])
    snaps = weekly_ecr([2023]).select("season", "week", "d").unique(["season", "week"])
    j = f.join(game_days([2023]), on=["player_id", "season", "week"]).join(snaps, on=["season", "week"])
    assert j.height > 5000
    assert (j["gameday"] > j["d"]).all()
    # Thursday games exist that week but have no rank
    thursday = game_days([2023]).filter(pl.col("gameday").dt.weekday() == 4)
    assert thursday.join(f, on=["player_id", "season", "week"], how="semi").height == 0


def test_one_rank_per_player_week():
    f = build_expert_features([2023])
    assert f.group_by("player_id", "season", "week").len()["len"].max() == 1
