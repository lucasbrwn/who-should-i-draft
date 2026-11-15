"""Draft state engine (SCRUM-58)."""

import pytest

from src.draft.engine import Draft, load_draft
from src.scoring.config import PRESETS


def draft(**kw):
    return Draft.for_league("test", PRESETS["standard-ppr"], user_slot=kw.pop("user_slot", 5), **kw)


def test_snake_order():
    d = draft()
    assert d.rounds == PRESETS["standard-ppr"].roster.size
    assert [d.team_for(o) for o in (1, 12, 13, 24, 25)] == [0, 11, 11, 0, 0]


def test_third_round_reversal():
    d = draft(third_round_reversal=True)
    # Round 3 runs 12 -> 1 like round 2; round 4 runs 1 -> 12.
    assert [d.team_for(o) for o in (13, 24, 25, 36, 37, 48)] == [11, 0, 11, 0, 0, 11]


def test_user_pick_numbers_from_slot_5():
    d = draft(user_slot=5)
    assert d.team_pick_numbers(4)[:4] == [5, 20, 29, 44]
    assert d.next_pick_for(4, after=5) == 20


def test_duplicates_bad_positions_and_overflow_are_rejected():
    d = draft()
    d.make_pick("p1", "A", "RB")
    with pytest.raises(ValueError, match="already drafted"):
        d.make_pick("p1", "A", "RB")
    with pytest.raises(ValueError, match="unknown position"):
        d.make_pick("p2", "B", "LB")


def test_undo():
    d = draft()
    d.make_pick("p1", "A", "RB")
    d.make_pick("p2", "B", "WR")
    assert d.undo().player_id == "p2"
    assert d.next_overall == 2 and d.on_clock == 1


def test_kdst_reminder_only_when_no_spare_picks_remain():
    cfg = PRESETS["standard-ppr"]   # 1 K + 1 DST slot
    d = draft(user_slot=1)
    for o in range(1, d.total_picks + 1):
        team = d.team_for(o)
        if team == 0 and d.picks_left(0) == 2:
            assert d.must_fill_kdst(0, cfg) == ["K", "DST"]
        if team == 0 and d.picks_left(0) > 2:
            assert d.must_fill_kdst(0, cfg) == []
        pos = "K" if (team == 0 and d.picks_left(0) == 2) else "DST" if (team == 0 and d.picks_left(0) == 1) else "WR"
        d.make_pick(f"p{o}", f"Player {o}", pos)
    assert d.must_fill_kdst(0, cfg) == []


def test_full_12_team_draft_records_and_replays(tmp_path):
    d = draft(user_slot=7, third_round_reversal=True)
    positions = ["RB", "WR", "QB", "TE"]
    for o in range(1, d.total_picks + 1):
        d.make_pick(f"p{o}", f"Player {o}", positions[o % 4])
    assert d.is_complete and d.on_clock is None
    with pytest.raises(ValueError, match="complete"):
        d.make_pick("extra", "X", "WR")
    assert all(len(d.roster(t)) == d.rounds for t in range(d.teams))

    loaded = load_draft(d.save(tmp_path))
    assert loaded == d                                   # identical state after save + load
    # Replaying the same picks into a fresh draft rebuilds the same rosters.
    replay = draft(user_slot=7, third_round_reversal=True)
    for p in loaded.picks:
        replay.make_pick(p.player_id, p.name, p.position)
    assert [p.team for p in replay.picks] == [p.team for p in d.picks]
