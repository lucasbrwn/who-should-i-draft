"""End-to-end leakage test (SCRUM-39): no feature for week N may use data from week N or later.

Take two real seasons (2022 for "last season", 2023 to test), scramble every RESULT from week N
onward (stats, points, plays, EPA, who left injured...), rebuild every feature table, and require
week N's features to be identical. If any feature peeked at week N or later, it changes.

Not scrambled: pre-game information (Vegas lines, injury reports, who is active), which is known
before kickoff, and the row itself (whether the player plays).
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from src.features.assemble import PBP_COLS, build_all, feature_columns
from src.features.team import read_schedules
from src.ingest.pbp import read_pbp

SEASONS = [2022, 2023]
TEST_SEASON, TEST_WEEKS = 2023, [2, 9, 15]
PRE_GAME = {"season", "week", "is_home", "on_injury_report", "report_status", "practice_code", "injury"}
PBP_RESULTS = ["epa", "pass_oe", "wp", "sack", "qb_hit", "qb_dropback", "rush_attempt", "pass_attempt"]

pytestmark = pytest.mark.skipif(
    not Path("data/processed/player_weeks.parquet").exists(), reason="needs the real data (run python -m src.ingest)"
)


@pytest.fixture(scope="module")
def real_data():
    pw = pl.read_parquet("data/processed/player_weeks.parquet").filter(pl.col("season").is_in(SEASONS))
    schedules = read_schedules().filter(pl.col("season").is_in(SEASONS))
    pbp = read_pbp(SEASONS, columns=PBP_COLS).join(
        schedules.select("game_id", "season", "week"), on="game_id", how="left"
    )
    return pw, schedules, pbp


def scramble_from(pw, schedules, pbp, week, seed=0):
    rng = np.random.default_rng(seed)
    future = lambda df: (pl.col("season") == TEST_SEASON) & (pl.col("week") >= week)  # noqa: E731

    def noise(df, cols):
        mask = df.select(future(df)).to_series()
        return df.with_columns([
            pl.when(mask).then(pl.Series(rng.uniform(0, 50, df.height)).cast(df.schema[c])).otherwise(pl.col(c)).alias(c)
            for c in cols
        ])

    pw_results = [c for c, t in pw.schema.items() if t.is_numeric() and c not in PRE_GAME]
    pw2 = noise(pw, pw_results).with_columns(
        left_injured=pl.when(future(pw)).then(pl.Series(rng.random(pw.height) < 0.5)).otherwise(pl.col("left_injured"))
    )
    sched2 = noise(schedules, ["home_score", "away_score"])
    pbp2 = noise(pbp, PBP_RESULTS).with_columns(
        pass_length=pl.when(future(pbp)).then(pl.lit("deep")).otherwise(pl.col("pass_length")),
        play_type=pl.when(future(pbp)).then(pl.lit("pass")).otherwise(pl.col("play_type")),
    )
    return pw2, sched2, pbp2.drop("season", "week")


@pytest.mark.parametrize("week", TEST_WEEKS)
def test_week_n_features_ignore_week_n_and_later(real_data, week):
    pw, schedules, pbp = real_data
    clean = build_all(pw, schedules, pbp.drop("season", "week"))
    dirty = build_all(*scramble_from(pw, schedules, pbp, week))

    cols = ["player_id", "season", "week"] + feature_columns(clean)
    this_week = lambda f: f.filter((pl.col("season") == TEST_SEASON) & (pl.col("week") == week)).select(cols)  # noqa: E731
    assert this_week(clean).height > 100
    assert_frame_equal(this_week(clean), this_week(dirty))


def test_the_scramble_really_changes_the_future(real_data):
    # Guard against a test that passes because nothing was scrambled: week 16 features
    # must change when results from week 9 on are scrambled.
    pw, schedules, pbp = real_data
    clean = build_all(pw, schedules, pbp.drop("season", "week"))
    dirty = build_all(*scramble_from(pw, schedules, pbp, 9))
    week16 = lambda f: f.filter((pl.col("season") == TEST_SEASON) & (pl.col("week") == 16)).select("targets_std")  # noqa: E731
    assert not week16(clean).equals(week16(dirty))
