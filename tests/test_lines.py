from datetime import date, datetime, timezone

import polars as pl

from app.lines import favorite_label, kickoff_label, upcoming_lines
from src.ingest.schedules import SNAPSHOT_PATH, season_path


def test_favorite_label_uses_home_positive_convention():
    assert favorite_label("BUF", "LAC", 7.0) == "BUF -7"
    assert favorite_label("MIA", "KC", -10.0) == "KC -10"
    assert favorite_label("TB", "MIN", -1.5) == "MIN -1.5"
    assert favorite_label("A", "B", 0.0) == "PK"


def test_kickoff_label():
    assert kickoff_label("2026-09-27", "13:00") == "SUN 1:00 PM ET"
    assert kickoff_label("2026-10-04", "09:30") == "SUN 9:30 AM ET"
    assert kickoff_label("2026-10-01", None) == "THU"


def _snapshot_row(game_id, week, gameday, spread, fetched):
    return {
        "game_id": game_id, "season": 2026, "week": week, "gameday": gameday, "gametime": "13:00",
        "home_team": "HOM", "away_team": "AWY", "spread_line": spread, "total_line": 44.5,
        "fetched_at": fetched,
    }


def test_upcoming_lines_picks_next_unplayed_week_with_latest_line(tmp_path):
    t1 = datetime(2026, 9, 26, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 27, tzinfo=timezone.utc)
    snaps = pl.DataFrame([
        _snapshot_row("wk3_done", 3, "2026-09-27", 3.0, t1),
        _snapshot_row("wk4_a", 4, "2026-10-04", 3.0, t1),
        _snapshot_row("wk4_a", 4, "2026-10-04", 4.5, t2),   # line moved: newest wins
        _snapshot_row("wk5_a", 5, "2026-10-11", -2.0, t2),
    ])
    (tmp_path / SNAPSHOT_PATH).parent.mkdir(parents=True)
    snaps.write_parquet(tmp_path / SNAPSHOT_PATH)
    pl.DataFrame({"game_id": ["wk3_done", "wk4_a"], "result": [7, None]}).write_parquet(tmp_path / season_path(2026))

    games = upcoming_lines(today=date(2026, 9, 27), root=tmp_path)
    assert [g["favorite"] for g in games] == ["HOM -4.5"]
    assert games[0]["week"] == 4


def test_upcoming_lines_empty_without_snapshots(tmp_path):
    assert upcoming_lines(root=tmp_path) == []
