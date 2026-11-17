"""Live predictions for an upcoming week (SCRUM-73/74/76).

Steps:
    1. upcoming_rows   one blank player-week row per ACTIVE skill player on this week's roster
                       (no stats yet), with that week's opponent, home/away, and injury report
                       if it's out. Players on IR / practice squad get no row.
    2. build_all       the normal feature pipeline over history + those rows. Every feature only
                       looks at earlier games (tests/test_leakage.py), so blank rows are safe.
    3. BoostingModel   tuned weekly model, trained on every finished game EXCEPT 2025, which
                       stays untouched until the final exam (SCRUM-66).
    4. Ranges          likely range = 25th-75th percentile outcome (src/models/ranges.py, BucketRanges
                       with LIKELY), fit on the 2022-24 walk-forward backtest. Inside ~half the time.
"""

from pathlib import Path

import polars as pl

from src.backtest.harness import FINAL_EXAM_SEASON, points
from src.features.assemble import PBP_COLS, build_all
from src.features.team import read_schedules
from src.ingest.pbp import read_pbp
from src.ingest.player_weeks import INJURIES_DIR, attach_injury_report, read_dir
from src.models.boosting import TUNED_PATH, BoostingModel
from src.models.ranges import LIKELY, BucketRanges, load_backtest

PW_PATH = Path("data/processed/player_weeks.parquet")
ROSTERS_DIR = Path("data/raw/rosters")
DEMO_PATH = Path("data/reference/demo_players.csv")
POSITIONS = ["QB", "RB", "WR", "TE"]


def week_games(schedules: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """One row per team playing this week: game_id, opponent, home/away, kickoff."""
    games = schedules.filter((pl.col("season") == season) & (pl.col("week") == week))
    cols = ["game_id", "gameday"]
    home = games.select(*cols, team=pl.col("home_team"), opponent_team=pl.col("away_team"), is_home=pl.lit(True))
    away = games.select(*cols, team=pl.col("away_team"), opponent_team=pl.col("home_team"), is_home=pl.lit(False))
    return pl.concat([home, away])


def upcoming_rows(rosters: pl.DataFrame, schedules: pl.DataFrame, injuries: pl.DataFrame,
                  season: int, week: int) -> pl.DataFrame:
    """Blank player-week rows for active skill players whose team plays this week."""
    active = rosters.filter(
        (pl.col("season") == season) & (pl.col("week") == week) & (pl.col("status") == "ACT")
        & pl.col("position").is_in(POSITIONS) & pl.col("gsis_id").is_not_null()
    ).select(
        pl.col("gsis_id").alias("player_id"), pl.col("full_name").alias("player_display_name"),
        "position", "team",
    ).unique("player_id")
    rows = active.join(week_games(schedules, season, week), on="team", how="inner").with_columns(
        season=pl.lit(season, pl.Int32), week=pl.lit(week, pl.Int32),
        snaps_only=pl.lit(False), left_injured=pl.lit(False),
    )
    return attach_injury_report(rows, injuries)


def train_rows(feats: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """Every finished game before this week, minus the final-exam season."""
    before = (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week))
    # sorted: boosting bins features from a row sample, so row order must be fixed for repeatable picks
    return feats.filter(before & (pl.col("season") != FINAL_EXAM_SEASON)).sort("season", "week", "player_id")


def project_week(season: int, week: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """All active players' projections for the week, plus that week's games with Vegas lines."""
    pw = pl.read_parquet(PW_PATH)
    seasons = sorted(pw["season"].unique().to_list())
    schedules = read_schedules().filter(pl.col("season").is_in(seasons))
    pbp = read_pbp(seasons, columns=PBP_COLS).join(
        schedules.select("game_id", "season", "week"), on="game_id", how="left")
    rosters = read_dir(ROSTERS_DIR, "rosters", [season])
    injuries = read_dir(INJURIES_DIR, "injuries", [season])

    pw = pw.filter((pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week)))
    upcoming = upcoming_rows(rosters, schedules, injuries, season, week)
    feats = build_all(pl.concat([pw, upcoming], how="diagonal_relaxed"), schedules, pbp)

    model = BoostingModel(tuned=TUNED_PATH, name="boosting").fit(train_rows(feats, season, week))
    rows = feats.filter((pl.col("season") == season) & (pl.col("week") == week)).sort("player_id")
    stats = model.predict(rows)
    pred_stats = rows.select("player_id", "position").join(stats, on="player_id", how="left")
    preds = rows.select("player_id", "player_display_name", "position", "team", "opponent_team", "game_id",
                        "report_status").with_columns(proj=points(pred_stats.drop("player_id")).round(1))
    # likely range = 25th-75th percentile outcome (he lands inside about half the time)
    preds = BucketRanges(LIKELY).fit(load_backtest()).predict(preds).select(
        pl.exclude("bust", "ceiling"), likely_low=pl.col("bust").round(1), likely_high=pl.col("ceiling").round(1))

    games = schedules.filter((pl.col("season") == season) & (pl.col("week") == week)).select(
        "game_id", "gameday", "gametime", "away_team", "home_team", "spread_line", "total_line").sort("gameday", "gametime")
    return preds.sort("proj", descending=True), games


def demo_table(preds: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """The 10 demo players. Anyone without a projection gets the reason (roster status, e.g. RES = IR)."""
    demo = pl.read_csv(DEMO_PATH)
    rosters = read_dir(ROSTERS_DIR, "rosters", [season]).filter(pl.col("week") == week)
    status = rosters.select(pl.col("gsis_id").alias("player_id"), pl.col("team").alias("roster_team"),
                            pl.col("status").alias("roster_status")).unique("player_id")
    out = demo.join(preds.drop("position", "team", "player_display_name"), on="player_id", how="left")
    out = out.join(status, on="player_id", how="left")
    team_changed = out.filter(pl.col("roster_team").is_not_null() & (pl.col("roster_team") != pl.col("team")))
    if team_changed.height:
        raise ValueError(f"demo_players.csv team is out of date: {team_changed.select('name', 'team', 'roster_team').rows()}")
    return out.drop("roster_team")
