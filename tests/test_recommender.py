"""Monte Carlo pick recommender (SCRUM-60)."""

import time

import polars as pl
import pytest

from src.draft.engine import Draft
from src.draft.recommender import BENCH_WEIGHT, Pool, best_vor, recommend, score_roster
from src.draft.vor import vor_board
from src.scoring.config import PRESETS
from src.scoring.lineup import Player

CFG = PRESETS["standard-ppr"]


def fake_board(n=60):
    rows = []
    for pos, top, step in [("QB", 350, 4), ("RB", 300, 7), ("WR", 290, 4), ("TE", 210, 6)]:
        for i in range(n):
            rows.append({"player_id": f"{pos}{i}", "player_display_name": f"{pos} {i}", "position": pos,
                         "season_b": float(max(top - step * i, 5))})
    return pl.DataFrame(rows)


def pool(adp=None):
    vb, _ = vor_board(fake_board(), CFG)
    return Pool.build(vb, adp)


def test_bench_counts_a_little():
    players = [Player("q", "QB", 300), Player("r1", "RB", 200), Player("r2", "RB", 190), Player("w1", "WR", 180),
               Player("w2", "WR", 170), Player("t", "TE", 120), Player("f", "RB", 150), Player("b", "WR", 100)]
    starters = 300 + 200 + 190 + 180 + 170 + 120 + 150
    assert score_roster(players, CFG) == pytest.approx(starters + BENCH_WEIGHT * 100)


def test_recommends_from_the_top_and_is_reproducible():
    p = pool()
    d = Draft.for_league("t", CFG, user_slot=1)
    a = recommend(d, p, CFG, n_sims=30, seed=1)
    b = recommend(d, p, CFG, n_sims=30, seed=1)
    assert [r["player_id"] for r in a] == [r["player_id"] for r in b]
    assert a[0]["expected"] >= a[-1]["expected"]
    assert best_vor(d, p, CFG) == p.ids[0]


def test_strategy_filter_is_respected():
    p = pool()
    d = Draft.for_league("t", CFG, user_slot=1)
    zero_rb = lambda pos, rnd, roster: not (pos == "RB" and rnd < 5)  # noqa: E731
    recs = recommend(d, p, CFG, allowed=zero_rb, n_sims=10)
    assert recs and all(r["position"] != "RB" for r in recs)


def test_kdst_when_out_of_spare_picks():
    p = pool()
    d = Draft.for_league("t", CFG, user_slot=1)
    while d.picks_left(0) > 2:   # everyone takes WRs until our last two picks
        team = d.on_clock
        d.make_pick(f"x{d.next_overall}", "X", "WR")
    while d.on_clock != 0:
        d.make_pick(f"x{d.next_overall}", "X", "WR")
    recs = recommend(d, p, CFG)
    assert recs[0]["position"] in ("K", "DST") and recs[0]["player_id"] is None


def test_fast_enough_on_a_full_size_board():
    p = pool()
    d = Draft.for_league("t", CFG, user_slot=6)
    for _ in range(5):
        i = len(d.picks)
        d.make_pick(p.ids[i], p.names[i], p.pos[i])
    # Live budget (two-stage search): fine for a 60-90 s draft clock. Mock-draft budget: fast enough
    # to run 1,080 drafts.
    start = time.time()
    recommend(d, p, CFG)
    assert time.time() - start < 12.0
    start = time.time()
    recommend(d, p, CFG, n_candidates=6, n_sims=50, stage2_top=0, stage2_sims=0)
    assert time.time() - start < 3.0


def test_second_qb_is_only_a_backup_in_1qb_but_a_starter_in_superflex():
    from src.draft.recommender import Roster
    repl = {"QB": 220.0, "RB": 130.0, "WR": 130.0, "TE": 90.0}
    one = Roster(CFG, {"QB": [320.0]}, repl)
    sf = Roster(PRESETS["superflex-half-ppr"], {"QB": [320.0]}, repl)
    gain_1qb = one.score(("QB", 300.0)) - one.score()
    gain_sf = sf.score(("QB", 300.0)) - sf.score()
    assert gain_1qb == pytest.approx(BENCH_WEIGHT * (300 - 220))   # backup: 15% of what he beats waivers by
    assert gain_sf == pytest.approx(300 - 220)                       # superflex starter over a waiver QB


def test_first_qb_adds_value_over_a_waiver_qb_not_his_raw_points():
    from src.draft.recommender import Roster
    repl = {"QB": 220.0, "RB": 130.0, "WR": 130.0, "TE": 90.0}
    empty = Roster(CFG, {}, repl)
    assert empty.score(("QB", 300.0)) - empty.score() == pytest.approx(80.0)


def test_caps_stop_a_third_qb():
    from src.draft.recommender import Roster, best_by_marginal, position_caps
    p = pool()
    r = Roster(CFG, {"QB": [350.0, 340.0]}, p.replacement)
    best = best_by_marginal(p, r, set(), 12, 10, ["QB", "QB"], None, caps=position_caps(CFG))
    assert all(p.pos[i] != "QB" for _, i in best)
