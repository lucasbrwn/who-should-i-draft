"""This week's Vegas lines for the scoreboard strip, from the line snapshots (src.ingest.schedules)."""

from datetime import date, datetime
from pathlib import Path

import polars as pl

from src.ingest.schedules import SNAPSHOT_PATH, season_path
from src.scoring.config import PROJECT_ROOT


def favorite_label(home: str, away: str, spread: float) -> str:
    """spread > 0 means the home team is favored (nflverse convention)."""
    if spread == 0:
        return "PK"
    return f"{home if spread > 0 else away} -{abs(spread):g}"


def kickoff_label(gameday: str, gametime: str | None) -> str:
    """'2026-09-27', '13:00' -> 'SUN 1:00 PM ET' (nflverse times are Eastern)."""
    if not gametime:
        return datetime.strptime(gameday, "%Y-%m-%d").strftime("%a").upper()
    kickoff = datetime.strptime(f"{gameday} {gametime}", "%Y-%m-%d %H:%M")
    return f"{kickoff.strftime('%a').upper()} {kickoff.strftime('%I:%M %p').lstrip('0')} ET"


def upcoming_lines(today: date | None = None, root: Path = PROJECT_ROOT) -> list[dict]:
    """Latest line for each game in the next week with lines posted. Empty if no snapshots yet."""
    snapshot = root / SNAPSHOT_PATH
    if not snapshot.exists():
        return []
    today = today or date.today()

    lines = (
        pl.read_parquet(snapshot)
        .sort("fetched_at")
        .group_by("game_id")
        .last()
        .filter(pl.col("gameday") >= today.isoformat())
    )
    finished = set()
    for season in lines["season"].unique().to_list():
        path = root / season_path(season)
        if path.exists():
            finished |= set(pl.read_parquet(path).filter(pl.col("result").is_not_null())["game_id"].to_list())
    lines = lines.filter(~pl.col("game_id").is_in(list(finished)))
    if lines.is_empty():
        return []

    season, week = lines.select("season", "week").sort("season", "week").row(0)
    games = lines.filter((pl.col("season") == season) & (pl.col("week") == week)).sort("gameday", "gametime", "game_id")
    return [
        {
            "week": g["week"],
            "away": g["away_team"],
            "home": g["home_team"],
            "favorite": favorite_label(g["home_team"], g["away_team"], g["spread_line"]),
            "total": f"{g['total_line']:g}",
            "kickoff": kickoff_label(g["gameday"], g["gametime"]),
        }
        for g in games.iter_rows(named=True)
    ]
