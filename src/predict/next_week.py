"""Print the season and week of the next NFL week with a game today or later ("2026 11").

Used by scripts/weekly_picks.ps1 so the scheduled task always predicts the upcoming week.
On a Thursday-night game day it returns that week; on Saturday, the same week (Sunday games ahead).

    python -m src.predict.next_week
"""

from datetime import date

import polars as pl

from src.features.team import read_schedules


def next_week(today: date, schedules: pl.DataFrame) -> tuple[int, int]:
    upcoming = (schedules.filter(pl.col("game_type") == "REG")
                .with_columns(day=pl.col("gameday").str.to_date())
                .filter(pl.col("day") >= today).sort("day"))
    if upcoming.is_empty():
        raise SystemExit("no upcoming regular-season games")
    first = upcoming.row(0, named=True)
    return first["season"], first["week"]


if __name__ == "__main__":
    season, week = next_week(date.today(), read_schedules())
    print(season, week)
