"""Download NFL schedules (results, Vegas lines, weather) and snapshot lines for upcoming games.

Completed games have one final line and kickoff weather, which is what the model trains on.
Lines for upcoming games move daily, so each run appends a timestamped snapshot instead of
overwriting. Projections use the latest snapshot; the history shows line movement.

spread_line > 0 means the home team is favored (nflverse convention, matches `result`).

Usage:
    python -m src.ingest.schedules                  # 2020 through current season + snapshot
    python -m src.ingest.schedules --seasons 2024
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy
import polars as pl

from src.ingest.player_stats import default_seasons

RAW_DIR = Path("data/raw/schedules")
SNAPSHOT_PATH = RAW_DIR / "line_snapshots.parquet"

# Games with no line or weather that are legitimately absent, e.g. the cancelled BUF-CIN game.
EXPECTED_REG_GAMES = {2020: 256, 2022: 271}
DEFAULT_REG_GAMES = 272

SNAPSHOT_COLS = [
    "game_id", "season", "week", "gameday", "gametime", "home_team", "away_team",
    "spread_line", "total_line", "home_moneyline", "away_moneyline",
    "home_spread_odds", "away_spread_odds", "over_odds", "under_odds",
    "roof", "temp", "wind",
]


def season_path(season: int) -> Path:
    return RAW_DIR / f"schedules_{season}.parquet"


def is_completed() -> pl.Expr:
    return pl.col("result").is_not_null()


def check_schedules(df: pl.DataFrame, season: int) -> tuple[list[str], list[str]]:
    """Returns (issues, notes). Issues are data errors; notes are known gaps worth knowing about."""
    issues, notes = [], []
    reg = df.filter(pl.col("game_type") == "REG")
    done = df.filter(is_completed())

    expected = EXPECTED_REG_GAMES.get(season, DEFAULT_REG_GAMES)
    if reg.height != expected:
        issues.append(f"{reg.height} REG games, expected {expected}")

    dupes = df.height - df["game_id"].n_unique()
    if dupes:
        issues.append(f"{dupes} duplicate game_ids")

    for col in ("spread_line", "total_line"):
        missing = done[col].null_count()
        if missing:
            issues.append(f"{missing} completed games missing {col}")

    outdoor = done.filter(pl.col("roof").is_in(["outdoors", "open"]))
    no_weather = outdoor.filter(pl.col("temp").is_null() | pl.col("wind").is_null()).height
    if no_weather:
        notes.append(f"{no_weather}/{outdoor.height} completed outdoor games missing temp/wind")

    unknown_roof = done["roof"].null_count()
    if unknown_roof:
        notes.append(f"{unknown_roof} completed games with unknown roof status")

    return issues, notes


def snapshot_upcoming(schedules: pl.DataFrame, fetched_at: datetime) -> pl.DataFrame:
    """Current lines for games not yet played, stamped with when we saw them."""
    return (
        schedules.filter(~is_completed() & pl.col("spread_line").is_not_null())
        .select(SNAPSHOT_COLS)
        .with_columns(pl.lit(fetched_at).alias("fetched_at"))
    )


def append_snapshot(snap: pl.DataFrame, path: Path = SNAPSHOT_PATH) -> int:
    """Append a snapshot, skipping games whose line hasn't changed since the last snapshot."""
    if snap.is_empty():
        return 0
    if path.exists():
        history = pl.read_parquet(path)
        watched = ["spread_line", "total_line", "home_moneyline", "away_moneyline"]
        latest = history.sort("fetched_at").group_by("game_id").last().select("game_id", *watched)
        changed = snap.join(latest, on="game_id", how="left", suffix="_prev").filter(
            pl.any_horizontal([
                pl.col(c).ne_missing(pl.col(f"{c}_prev")) for c in watched
            ])
        ).select(snap.columns)
        combined = pl.concat([history, changed], how="diagonal_relaxed")
        new_rows = changed.height
    else:
        combined, new_rows = snap, snap.height
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(path)
    return new_rows


def latest_lines(path: Path = SNAPSHOT_PATH) -> pl.DataFrame:
    """Most recent line we have for each upcoming game."""
    return pl.read_parquet(path).sort("fetched_at").group_by("game_id").last()


def ingest(seasons: list[int]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fetched_at = datetime.now(timezone.utc)
    schedules = nflreadpy.load_schedules(seasons)

    for season in seasons:
        df = schedules.filter(pl.col("season") == season)
        df.write_parquet(season_path(season))
        issues, notes = check_schedules(df, season)
        print(f"{season}: {df.height} games ({df.filter(is_completed()).height} completed) -> {season_path(season)}")
        for issue in issues:
            print(f"  ! {issue}")
        for note in notes:
            print(f"  - {note}")

    added = append_snapshot(snapshot_upcoming(schedules, fetched_at))
    print(f"Line snapshot {fetched_at:%Y-%m-%d %H:%M} UTC: {added} new/changed game lines -> {SNAPSHOT_PATH}")


def read_schedules(seasons: list[int] | None = None) -> pl.DataFrame:
    seasons = seasons or default_seasons()
    return pl.concat([pl.read_parquet(season_path(s)) for s in seasons], how="diagonal_relaxed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=None)
    args = parser.parse_args()
    ingest(args.seasons or default_seasons())


if __name__ == "__main__":
    main()
