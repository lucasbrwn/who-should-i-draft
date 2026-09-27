from datetime import datetime, timezone

import polars as pl

from src.ingest.availability import check_snaps, clean_injuries, out_but_played
from src.ingest.ids import attach_gsis_id


def snap(player, pfr_id, team="AAA", game_id="G1", offense=40.0, defense=0.0, st=0.0, position="WR"):
    return {
        "game_id": game_id, "player": player, "pfr_player_id": pfr_id, "position": position,
        "team": team, "offense_snaps": offense, "offense_pct": offense / 60,
        "defense_snaps": defense, "st_snaps": st,
    }


def test_attach_gsis_id_uses_directory_then_unique_name_match():
    snaps = pl.DataFrame([
        snap("Known Guy", "KnowGu00"),
        snap("Brian Thomas Jr.", "ThomBr00"),  # not in directory; name matches stats "Brian Thomas"
        snap("John Smith", "SmitJo00"),        # two John Smiths on the team: ambiguous, stays unlinked
    ])
    id_map = pl.DataFrame({"pfr_player_id": ["KnowGu00"], "gsis_id": ["00-1"]})
    stats = pl.DataFrame({
        "game_id": ["G1"] * 3, "team": ["AAA"] * 3,
        "player_id": ["00-2", "00-3", "00-4"],
        "player_display_name": ["Brian Thomas", "John Smith", "John Smith"],
    })
    out = {r["player"]: (r["gsis_id"], r["id_source"]) for r in attach_gsis_id(snaps, id_map, stats).to_dicts()}
    assert out["Known Guy"] == ("00-1", "directory")
    assert out["Brian Thomas Jr."] == ("00-2", "name_match")
    assert out["John Smith"] == (None, None)


def test_check_snaps_reports_link_rate():
    snaps = pl.DataFrame([snap("A", "A0"), snap("B", "B0")]).with_columns(
        pl.Series("gsis_id", ["00-1", None]), pl.Series("id_source", ["directory", None])
    )
    issues, notes = check_snaps(snaps)
    assert issues == []
    assert notes[0].startswith("skill-position player-games linked to gsis_id: 1/2")


def injury(status, modified=None, week=1.0, practice="Limited Participation in Practice"):
    row = {
        "season": 2024.0, "game_type": "REG", "week": week, "team": "AAA", "gsis_id": "00-1",
        "full_name": "Player", "position": "WR", "report_status": status, "practice_status": practice,
    }
    if modified:
        row["date_modified"] = modified
    return row


def test_clean_injuries_keeps_latest_status_and_remembers_first():
    t1 = datetime(2024, 12, 14, 20, tzinfo=timezone.utc)
    t2 = datetime(2024, 12, 15, 14, tzinfo=timezone.utc)
    raw = pl.DataFrame([injury("Out", t2), injury("Questionable", t1)])  # out of time order on purpose
    out = clean_injuries(raw)
    assert out.height == 1
    assert out.row(0, named=True)["report_status"] == "Out"
    assert out.row(0, named=True)["initial_report_status"] == "Questionable"
    assert out.row(0, named=True)["practice_code"] == "LP"
    assert out.schema["week"] == pl.Int32


def test_clean_injuries_without_timestamps_uses_file_order():
    out = clean_injuries(pl.DataFrame([injury("Questionable"), injury("Out")]))
    assert out.row(0, named=True)["report_status"] == "Out"


def test_out_but_played_finds_contradictions():
    injuries = clean_injuries(pl.DataFrame([injury("Out"), {**injury("Out"), "gsis_id": "00-2"}]))
    snaps = pl.DataFrame([
        {**snap("P1", "P1"), "gsis_id": "00-1", "offense_snaps": 0.0},             # sat out: fine
        {**snap("P2", "P2"), "gsis_id": "00-2", "offense_snaps": 0.0, "st_snaps": 5.0},  # played special teams
    ])
    schedules = pl.DataFrame({"game_id": ["G1"], "season": [2024], "week": [1], "game_type": ["REG"],
                              "home_team": ["AAA"], "away_team": ["BBB"]}).with_columns(
        pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    out = out_but_played(injuries, snaps, schedules)
    assert out["gsis_id"].to_list() == ["00-2"]
