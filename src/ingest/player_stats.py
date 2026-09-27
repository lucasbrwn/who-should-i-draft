"""Download weekly player stats from nflverse and save one Parquet file per season.

Usage:
    python -m src.ingest.player_stats                 # 2020 through the current season
    python -m src.ingest.player_stats --seasons 2023 2024
"""

import argparse
from pathlib import Path

import nflreadpy
import polars as pl

from src.ingest.checks import check_player_stats

FIRST_SEASON = 2020
RAW_DIR = Path("data/raw/player_stats")


def default_seasons() -> list[int]:
    return list(range(FIRST_SEASON, nflreadpy.get_current_season() + 1))


def load_season(season: int) -> pl.DataFrame:
    """Weekly stats (regular + postseason) for every player who recorded a stat."""
    return nflreadpy.load_player_stats([season], summary_level="week")


def season_path(season: int) -> Path:
    return RAW_DIR / f"player_stats_{season}.parquet"


def ingest(seasons: list[int]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        df = load_season(season)
        df.write_parquet(season_path(season))

        issues = check_player_stats(df, season)
        reg = df.filter(pl.col("season_type") == "REG")
        print(
            f"{season}: {df.height:,} rows "
            f"({reg.height:,} REG, weeks {reg['week'].min()}-{reg['week'].max()}) "
            f"-> {season_path(season)}"
        )
        for issue in issues:
            print(f"  ! {issue}")


def read_player_stats(seasons: list[int] | None = None) -> pl.DataFrame:
    """Read saved seasons back as one DataFrame."""
    seasons = seasons or default_seasons()
    return pl.concat([pl.read_parquet(season_path(s)) for s in seasons], how="diagonal_relaxed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=None)
    args = parser.parse_args()
    ingest(args.seasons or default_seasons())


if __name__ == "__main__":
    main()
