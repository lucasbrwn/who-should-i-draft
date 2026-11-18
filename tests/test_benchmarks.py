"""Benchmark snapshot parsing (ESPN JSON -> one projection per player for the requested week)."""

from src.predict.benchmarks import parse_espn


def stat(season, week, source, total, split=1):
    return {"seasonId": season, "scoringPeriodId": week, "statSourceId": source,
            "statSplitTypeId": split, "appliedTotal": total}


def test_parse_espn_keeps_only_this_weeks_projection():
    players = [
        {"player": {"id": 1, "fullName": "A", "injuryStatus": "QUESTIONABLE", "stats": [
            stat(2026, 11, 1, 21.9),     # this week's projection: keep
            stat(2026, 10, 1, 23.8),     # last week's projection
            stat(2026, 10, 0, 44.2),     # last week's actual
            stat(2025, 11, 1, 17.4),     # same week, last season
            stat(2026, 11, 1, 99.0, split=0),  # season total, not a week
        ]}},
        {"player": {"id": 2, "fullName": "B", "stats": [stat(2026, 12, 1, 10.0)]}},  # no week-11 line
    ]
    out = parse_espn(players, 2026, 11)
    assert out.rows() == [("1", "A", 21.9, "QUESTIONABLE")]   # live status kept (v2: rules players out)


def test_parse_espn_empty_response():
    assert parse_espn([], 2026, 11).height == 0
