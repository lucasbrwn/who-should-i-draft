"""Expected fantasy points per player-week (nflverse ff_opportunity), SCRUM-46.

For every target, carry, and pass, nflverse's model estimates what an average player would have
gained from that opportunity (based on where on the field it happened, air yards, down and
distance). `*_exp` columns are those expectations, `*_diff` = actual minus expected.

Why it matters: the literature review found expected TDs predict future TDs better than actual
TDs or raw red-zone counts. A player who scored 3 TDs on 0.8 expected TDs got lucky.

Usage:
    python -m src.ingest.ff_opportunity
"""

import argparse
from pathlib import Path

import nflreadpy
import polars as pl

from src.ingest.player_stats import default_seasons

RAW_DIR = Path("data/raw/ff_opportunity")
KEY = ["player_id", "game_id"]
MIN_EXPECTED_POINTS = -2.0


def season_path(season: int) -> Path:
    return RAW_DIR / f"ff_opportunity_{season}.parquet"


def clean(raw: pl.DataFrame) -> pl.DataFrame:
    """nflverse ships season as text and week as a float; match the other tables' Int32 keys."""
    return raw.with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32)).filter(
        pl.col("player_id").is_not_null()
    )


def check_ff_opportunity(df: pl.DataFrame, player_weeks: pl.DataFrame | None = None) -> list[str]:
    issues = []
    dupes = df.height - df.unique(KEY).height
    if dupes:
        issues.append(f"{dupes} duplicate player-games")
    # Slightly negative is real: backup QBs kneeling out the clock (-3 yds -> about -0.3 pts).
    # Anything far below that would mean a broken row.
    too_low = df.filter(pl.col("total_fantasy_points_exp") < MIN_EXPECTED_POINTS).height
    if too_low:
        issues.append(f"{too_low} rows with expected fantasy points below {MIN_EXPECTED_POINTS}")
    if player_weeks is not None:
        # every player-game with a target or carry should have expected points
        used = player_weeks.filter((pl.col("targets") > 0) | (pl.col("carries") > 0))
        missing = used.join(df.select(KEY), on=KEY, how="anti").height
        if missing / max(used.height, 1) > 0.01:
            issues.append(f"{missing} of {used.height:,} player-games with a target/carry have no expected points")
    return issues


def ingest(seasons: list[int]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        try:
            df = clean(nflreadpy.load_ff_opportunity([season], stat_type="weekly"))
        except Exception as exc:   # current season can lag a few days behind
            print(f"{season}: not available ({exc.__class__.__name__})")
            continue
        df.write_parquet(season_path(season))
        print(f"{season}: {df.height:,} player-games, weeks {df['week'].min()}-{df['week'].max()} -> {season_path(season)}")
        for issue in check_ff_opportunity(df):
            print(f"  ! {issue}")


def read_ff_opportunity(seasons: list[int]) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(season_path(s)) for s in seasons if season_path(s).exists()],
                     how="diagonal_relaxed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=None)
    args = parser.parse_args()
    ingest(args.seasons or default_seasons())


if __name__ == "__main__":
    main()
