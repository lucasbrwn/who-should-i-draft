"""Download nflverse play-by-play and save one Parquet file per season.

One row per play (~49k per season, ~370 columns). This is the raw material for defensive ratings
in M3: EPA, pass_location, air_yards, sacks, QB hits, all tagged with the defending team.

Usage:
    python -m src.ingest.pbp                 # 2020 through the current season
    python -m src.ingest.pbp --seasons 2023 2024
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import nflreadpy
import polars as pl

from src.ingest.player_stats import default_seasons

RAW_DIR = Path("data/raw/pbp")
KEY = ["game_id", "play_id"]

# Per-play add-ons that only exist for some seasons (nflreadpy raises ValueError outside them).
# participation: NGS coverage (man/zone), pressure, pass rushers, each player's route.
# ftn: FTN charting -- blitzers, play action, drops. Starts in 2022.
EXTRAS: dict[str, Callable[[list[int]], pl.DataFrame]] = {
    "participation": nflreadpy.load_participation,
    "ftn": nflreadpy.load_ftn_charting,
}

# Plays where the offense actually ran a snap (not kickoffs, punts, penalties, timeouts).
SCRIMMAGE = pl.col("play_type").is_in(["pass", "run"])
# Columns M3 needs on every scrimmage play; a gap here means a hole in the defensive ratings.
REQUIRED_ON_SCRIMMAGE = ["posteam", "defteam", "epa"]
MAX_NULL_SHARE = 0.01


def season_path(season: int) -> Path:
    return RAW_DIR / f"pbp_{season}.parquet"


def extra_path(name: str, season: int) -> Path:
    return Path("data/raw") / name / f"{name}_{season}.parquet"


def load_if_available(loader: Callable[[list[int]], pl.DataFrame], season: int) -> pl.DataFrame | None:
    """None when the source doesn't cover this season, instead of crashing the whole ingest."""
    try:
        return loader([season])
    except ValueError:
        return None


def fill_rates(df: pl.DataFrame, columns: list[str]) -> dict[str, float]:
    """Share of rows (0-1) where each column has a value. Empty frame -> 0.0 for every column."""
    if df.height == 0:
        return {c: 0.0 for c in columns}
    return {c: 1 - df[c].null_count() / df.height for c in columns}


def check_pbp(df: pl.DataFrame, season: int, expected_games: set[str] | None = None) -> list[str]:
    """Human-readable issues; an empty list means the season passed."""
    issues = []

    dupes = df.select(KEY).is_duplicated().sum()
    if dupes:
        issues.append(f"{dupes} duplicate (game_id, play_id) rows")

    wrong_season = df.filter(pl.col("season") != season).height
    if wrong_season:
        issues.append(f"{wrong_season} plays tagged with a season other than {season}")

    if expected_games is not None:
        missing = expected_games - set(df["game_id"].unique())
        if missing:
            issues.append(f"{len(missing)} completed games have no plays, e.g. {sorted(missing)[:3]}")

    scrimmage = df.filter(SCRIMMAGE)
    for col in REQUIRED_ON_SCRIMMAGE:
        share = scrimmage[col].null_count() / max(scrimmage.height, 1)
        if share > MAX_NULL_SHARE:
            issues.append(f"{col} is null on {share:.1%} of scrimmage plays")
    return issues


def ingest(seasons: list[int]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        df = nflreadpy.load_pbp([season])
        df.write_parquet(season_path(season))
        print(f"{season}: {df.height:,} plays, {df['game_id'].n_unique()} games -> {season_path(season)}")
        for issue in check_pbp(df, season):
            print(f"  ! {issue}")

        for name, loader in EXTRAS.items():
            extra = load_if_available(loader, season)
            if extra is None:
                print(f"  {name}: not available for {season}")
                continue
            extra_path(name, season).parent.mkdir(parents=True, exist_ok=True)
            extra.write_parquet(extra_path(name, season))
            print(f"  {name}: {extra.height:,} plays -> {extra_path(name, season)}")


def read_pbp(seasons: list[int] | None = None, columns: list[str] | None = None) -> pl.DataFrame:
    """Read saved seasons back as one DataFrame. Pass `columns` to load only what you need --
    370 columns x 7 seasons is a lot of memory for a job that uses 10 of them."""
    seasons = seasons or default_seasons()
    return pl.concat(
        [pl.read_parquet(season_path(s), columns=columns) for s in seasons], how="diagonal_relaxed"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=None)
    args = parser.parse_args()
    ingest(args.seasons or default_seasons())


if __name__ == "__main__":
    main()
