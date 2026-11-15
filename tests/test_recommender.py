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
    start = time.time()
    recommend(d, p, CFG)
    assert time.time() - start < 3.0
