"""Chance he plays (live predictions only).

The weekly model is trained on games players actually PLAYED, so its projection means "points if he
plays". In live predictions every active player gets a row, including backups who usually sit and
starters who may be ruled out. Week 11 2026: Kyler Murray (MIN QB2) was projected 12.6.

    expected points = projection if he plays x chance he plays
    chance he plays = depth-chart rate x injury rate

    depth-chart rate  share of HEALTHY active players at that position and pre-game depth rank who
                      played (QB2 ~32%, RB2 ~94%, WR1 ~99%)
    injury rate       share of players with that final injury status + last practice level who played
                      (Questionable + did not practice ~44%, Doubtful ~1%, Out 0). 1 if not on the report.

Rates come from earlier seasons only (fit(seasons)); "played" = a row in player_weeks, the same
definition the model trains on. Cells with fewer than MIN_N examples fall back to the broader group.
"""

from pathlib import Path

import polars as pl

from src.ingest.depth_charts import read_weekly_depth
from src.ingest.player_weeks import INJURIES_DIR, read_dir

ROSTERS_DIR = Path("data/raw/rosters")
PW_PATH = Path("data/processed/player_weeks.parquet")
POSITIONS = ["QB", "RB", "WR", "TE"]
# Weekly rosters are game-day snapshots: players ruled inactive show as INA. Before kickoff they were
# on the active roster, so count both (using ACT alone hides exactly the players who sat).
PRE_GAME_ACTIVE = ["ACT", "INA"]
DESIGNATIONS = ["Questionable", "Doubtful", "Out"]
MIN_N = 30


def depth_bucket(rank: pl.Expr) -> pl.Expr:
    return (pl.when(rank.is_null()).then(pl.lit("off chart"))
            .when(rank >= 3).then(pl.lit("3+")).otherwise(rank.cast(pl.Int32).cast(pl.String)))


def game_day_pool(seasons: list[int], schedules: pl.DataFrame) -> pl.DataFrame:
    """Every active QB/RB/WR/TE on a team that played that week, with depth rank, injury report, and
    whether he played."""
    rosters = read_dir(ROSTERS_DIR, "rosters", seasons).filter(
        pl.col("status").is_in(PRE_GAME_ACTIVE) & pl.col("position").is_in(POSITIONS) & (pl.col("game_type") == "REG")
    ).select("season", "week", pl.col("gsis_id").alias("player_id"), "position", "team").drop_nulls().unique(
        ["season", "week", "player_id"])
    games = schedules.filter(pl.col("game_type") == "REG")
    played_teams = pl.concat([games.select("season", "week", team=pl.col("home_team")),
                              games.select("season", "week", team=pl.col("away_team"))])
    injuries = read_dir(INJURIES_DIR, "injuries", seasons).filter(pl.col("game_type") == "REG").select(
        "season", "week", pl.col("gsis_id").alias("player_id"), "report_status", "practice_code"
    ).unique(["season", "week", "player_id"])
    played = pl.read_parquet(PW_PATH, columns=["player_id", "season", "week"]).with_columns(played=pl.lit(True))
    key = ["season", "week", "player_id"]
    return (rosters.join(played_teams, on=["season", "week", "team"])
            .join(read_weekly_depth(seasons, schedules), on=key, how="left")
            .join(injuries, on=key, how="left")
            .join(played, on=key, how="left")
            .with_columns(pl.col("played").fill_null(False), depth=depth_bucket(pl.col("depth_rank"))))


class PlayChance:
    def fit(self, pool: pl.DataFrame) -> "PlayChance":
        """pool: position, depth_rank, report_status, practice_code, played (see game_day_pool)."""
        pool = pool.with_columns(depth=depth_bucket(pl.col("depth_rank")))
        healthy = pool.filter(~pl.col("report_status").is_in(DESIGNATIONS) | pl.col("report_status").is_null())
        self.depth = healthy.group_by("position", "depth").agg(depth_rate=pl.col("played").mean(), n=pl.len())
        self.position = healthy.group_by("position").agg(position_rate=pl.col("played").mean())
        hurt = pool.filter(pl.col("report_status").is_in(DESIGNATIONS))
        self.injury = hurt.group_by("report_status", "practice_code").agg(injury_rate=pl.col("played").mean(), n=pl.len())
        self.status = hurt.group_by("report_status").agg(status_rate=pl.col("played").mean())
        return self

    def rates(self, rows: pl.DataFrame) -> pl.DataFrame:
        """rows need position, depth_rank, report_status, practice_code. Adds depth_rate, injury_rate, p_play."""
        depth = self.depth.filter(pl.col("n") >= MIN_N).drop("n")
        injury = self.injury.filter(pl.col("n") >= MIN_N).drop("n")
        out = (rows.with_columns(depth=depth_bucket(pl.col("depth_rank")))
               .join(depth, on=["position", "depth"], how="left")
               .join(self.position, on="position", how="left")
               .join(injury, on=["report_status", "practice_code"], how="left", nulls_equal=False)
               .join(self.status, on="report_status", how="left"))
        designated = pl.col("report_status").is_in(DESIGNATIONS).fill_null(False)
        return out.with_columns(
            depth_rate=pl.coalesce("depth_rate", "position_rate"),
            injury_rate=pl.when(designated).then(pl.coalesce("injury_rate", "status_rate")).otherwise(1.0),
        ).with_columns(p_play=(pl.col("depth_rate") * pl.col("injury_rate")).round(3)).drop(
            "depth", "position_rate", "status_rate")


def apply(preds: pl.DataFrame, chance: PlayChance) -> pl.DataFrame:
    """proj -> proj_if_plays; proj becomes expected points. If he sits more than 25% of the time, the
    25th-percentile outcome is a zero, so the likely range starts at 0."""
    out = chance.rates(preds).rename({"proj": "proj_if_plays"})
    out = out.with_columns(proj=(pl.col("proj_if_plays") * pl.col("p_play")).round(1))
    if "likely_low" in out.columns:
        out = out.with_columns(likely_low=pl.when(pl.col("p_play") < 0.75).then(0.0).otherwise(pl.col("likely_low")))
    return out
