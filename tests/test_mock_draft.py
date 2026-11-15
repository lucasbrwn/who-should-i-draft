"""Mock-draft test machinery (SCRUM-59)."""

import numpy as np
import polars as pl
import pytest

from src.backtest.mock_draft import run_draft, season_score
from src.draft.recommender import Pool
from src.draft.vor import vor_board
from src.scoring.config import PRESETS

CFG = PRESETS["standard-ppr"]


def test_season_score_uses_the_best_lineup_each_week():
    weeks = {"q": np.array([20.0] + [0.0] * 17), "q2": np.array([0.0, 30.0] + [0.0] * 16)}
    # One QB slot: week 1 starts q (20), week 2 starts q2 (30).
    assert season_score([("q", "QB"), ("q2", "QB")], weeks, CFG) == pytest.approx(50.0)


def small_pool():
    rows = [{"player_id": f"{pos}{i}", "player_display_name": f"{pos}{i}", "position": pos,
             "season_b": float(300 - 3 * i)} for pos in ["QB", "RB", "WR", "TE"] for i in range(80)]
    vb, _ = vor_board(pl.DataFrame(rows), CFG)
    return Pool.build(vb, None)


def test_every_team_gets_a_full_legal_roster_with_kdst_last():
    pool = small_pool()
    ecr = np.arange(len(pool.ids), dtype=float)
    rosters = run_draft(pool, ecr, CFG, slot=3, drafter="vor", seed=1)
    assert all(len(r) == CFG.roster.size for r in rosters)
    for r in rosters:
        pos = [p for _, p in r]
        assert pos.count("K") == 1 and pos.count("DST") == 1 and pos.count("QB") <= 2
        assert set(pos[-2:]) == {"K", "DST"}


def test_same_seed_same_opponent_picks_until_our_first_pick():
    pool = small_pool()
    ecr = np.arange(len(pool.ids), dtype=float)
    a = run_draft(pool, ecr, CFG, slot=6, drafter="vor", seed=7)
    b = run_draft(pool, ecr, CFG, slot=6, drafter="adp", seed=7)
    # Teams picking before slot 6 in round 1 see identical boards, so they take the same players.
    assert [r[0] for r in a[:5]] == [r[0] for r in b[:5]]
