from datetime import datetime, timezone

import polars as pl

from src.ingest.schedules import (
    SNAPSHOT_COLS, append_snapshot, check_schedules, latest_lines, snapshot_upcoming,
)

T1 = datetime(2026, 9, 25, 12, tzinfo=timezone.utc)
T2 = datetime(2026, 9, 26, 12, tzinfo=timezone.utc)


def game(game_id, spread=3.0, total=45.5, result=None, roof="outdoors", temp=60, wind=5, game_type="REG"):
    row = {c: None for c in SNAPSHOT_COLS}
    row.update({
        "game_id": game_id, "season": 2026, "week": 4, "game_type": game_type,
        "home_team": "HOM", "away_team": "AWY", "spread_line": spread, "total_line": total,
        "home_moneyline": -150, "away_moneyline": 130, "result": result,
        "roof": roof, "temp": temp, "wind": wind,
    })
    return row


def test_snapshot_only_includes_unplayed_games_with_lines():
    sched = pl.DataFrame([game("played", result=7), game("upcoming"), game("no_line", spread=None)])
    snap = snapshot_upcoming(sched, T1)
    assert snap["game_id"].to_list() == ["upcoming"]
    assert snap["fetched_at"].to_list() == [T1]


def test_append_snapshot_keeps_history_and_skips_unchanged(tmp_path):
    path = tmp_path / "snaps.parquet"
    sched = pl.DataFrame([game("A", spread=3.0), game("B", spread=-2.5)])
    assert append_snapshot(snapshot_upcoming(sched, T1), path) == 2

    moved = pl.DataFrame([game("A", spread=4.5), game("B", spread=-2.5)])  # only A's line moved
    assert append_snapshot(snapshot_upcoming(moved, T2), path) == 1

    history = pl.read_parquet(path)
    assert history.filter(pl.col("game_id") == "A")["spread_line"].to_list() == [3.0, 4.5]
    latest = {r["game_id"]: r["spread_line"] for r in latest_lines(path).to_dicts()}
    assert latest == {"A": 4.5, "B": -2.5}


def test_check_schedules_flags_missing_lines_and_notes_weather_gaps():
    games = [game(f"g{i}", result=3) for i in range(272)]
    games[0]["spread_line"] = None
    games[1]["temp"] = None
    games[2]["roof"] = "dome"
    games[2]["temp"] = None  # domes never have weather: not a gap
    issues, notes = check_schedules(pl.DataFrame(games), 2026)
    assert issues == ["1 completed games missing spread_line"]
    assert notes == ["1/271 completed outdoor games missing temp/wind"]


def test_check_schedules_allows_known_short_seasons():
    assert check_schedules(pl.DataFrame([game(f"g{i}", result=3) for i in range(271)]), 2022)[0] == []
    assert check_schedules(pl.DataFrame([game(f"g{i}", result=3) for i in range(271)]), 2024)[0] == [
        "271 REG games, expected 272"
    ]
