"""Download team depth charts from nflverse (QB/RB/WR/TE only) and turn them into one row per
player per game week: where he sat on his team's depth chart BEFORE that game.

nflverse changed format in 2025:
    2020-2024  one chart per team per week: depth_team 1/2/3 within each spot
               (three receiver spots, so a team has three WR "1"s)
    2025+      dated snapshots (several per week): pos_rank numbers ALL receivers together
               (1-6), with pos_slot telling which receiver spot. We re-rank within each spot so
               both eras mean the same thing: 1 = starter at that spot, 2 = his backup, ...
               and use the latest snapshot taken BEFORE game day (pre-game information only).

Usage:
    python -m src.ingest.depth_charts                 # 2020 through the current season
    python -m src.ingest.depth_charts --seasons 2026
"""

import argparse
from pathlib import Path

import nflreadpy
import polars as pl

from src.ingest.player_stats import default_seasons

RAW_DIR = Path("data/raw/depth_charts")
POSITIONS = ["QB", "RB", "WR", "TE"]
NEW_FORMAT_FROM = 2025


def season_path(season: int) -> Path:
    return RAW_DIR / f"depth_charts_{season}.parquet"


def load_season(season: int) -> pl.DataFrame:
    """Raw chart rows for skill positions (the full file has every defender and kicker too)."""
    df = nflreadpy.load_depth_charts([season])
    if season >= NEW_FORMAT_FROM:
        return df.filter(pl.col("pos_abb").is_in(POSITIONS))
    return df.filter(pl.col("position").is_in(POSITIONS + ["HB"]) & (pl.col("game_type") == "REG"))


def weekly_old_format(raw: pl.DataFrame) -> pl.DataFrame:
    return raw.filter(pl.col("gsis_id").is_not_null()).select(
        "season", "week", pl.col("gsis_id").alias("player_id"),
        pl.col("depth_team").str.strip_chars().cast(pl.Int32, strict=False).alias("depth_rank"),
    ).drop_nulls("depth_rank")


def weekly_new_format(raw: pl.DataFrame, schedules: pl.DataFrame, season: int) -> pl.DataFrame:
    """Latest snapshot strictly before each team's game day, re-ranked within receiver spot."""
    games = schedules.filter((pl.col("season") == season) & (pl.col("game_type") == "REG"))
    team_games = pl.concat([
        games.select("week", "gameday", team=pl.col("home_team")),
        games.select("week", "gameday", team=pl.col("away_team")),
    ])
    snaps = raw.filter(pl.col("gsis_id").is_not_null()).with_columns(day=pl.col("dt").str.slice(0, 10))
    # rank within (snapshot, team, position, spot): 1 = the starter at that spot
    snaps = snaps.with_columns(
        spot_rank=pl.col("pos_rank").rank("dense").over("dt", "team", "pos_abb", "pos_slot").cast(pl.Int32))
    stamps = snaps.select("team", "dt", "day").unique()
    chosen = (team_games.join(stamps, on="team")
              .filter(pl.col("day") < pl.col("gameday"))
              .group_by("team", "week").agg(pl.col("dt").max()))
    rows = snaps.join(chosen, on=["team", "dt"])
    return rows.group_by("week", "gsis_id").agg(pl.col("spot_rank").min().alias("depth_rank")).select(
        pl.lit(season, pl.Int32).alias("season"), pl.col("week").cast(pl.Int32),
        pl.col("gsis_id").alias("player_id"), "depth_rank")


def weekly_depth(season: int, schedules: pl.DataFrame) -> pl.DataFrame:
    """One row per player-week: his best (lowest) rank across the spots he's listed at."""
    raw = pl.read_parquet(season_path(season))
    rows = (weekly_new_format(raw, schedules, season) if season >= NEW_FORMAT_FROM
            else weekly_old_format(raw))
    return rows.group_by("season", "week", "player_id").agg(pl.col("depth_rank").min()).sort("season", "week")


def read_weekly_depth(seasons: list[int], schedules: pl.DataFrame) -> pl.DataFrame:
    return pl.concat([weekly_depth(s, schedules) for s in seasons if season_path(s).exists()])


def ingest(seasons: list[int]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        df = load_season(season)
        df.write_parquet(season_path(season))
        print(f"{season}: {df.height:,} skill-position chart rows -> {season_path(season)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=None)
    args = parser.parse_args()
    ingest(args.seasons or default_seasons())


if __name__ == "__main__":
    main()
