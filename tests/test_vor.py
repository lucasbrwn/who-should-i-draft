"""VOR and tiers (SCRUM-57)."""

import polars as pl
import pytest

from src.draft.vor import league_points, replacement_levels, tiers, vor_board
from src.scoring.config import PRESETS


def board(n_per_pos=40):
    """Fake board: points fall off linearly at each position; QBs score the most."""
    rows = []
    base = {"QB": 380, "RB": 300, "WR": 290, "TE": 200}
    for pos, top in base.items():
        for i in range(n_per_pos):
            rows.append({"player_id": f"{pos}{i}", "player_display_name": f"{pos} {i}", "position": pos,
                         "season_b": float(top - 6 * i)})
    return pl.DataFrame(rows)


def test_superflex_moves_qbs_up():
    one_qb, _ = vor_board(board(), PRESETS["standard-ppr"])
    sflex, _ = vor_board(board(), PRESETS["superflex-half-ppr"])
    qb1 = lambda b: b.filter(pl.col("player_id") == "QB0")["rank"].item()  # noqa: E731
    assert qb1(sflex) < qb1(one_qb)


def test_replacement_is_the_first_non_starters():
    b = board().with_columns(points=pl.col("season_b"))
    repl = replacement_levels(b, PRESETS["standard-ppr"])
    # 12 teams x 1 QB = 12 QB starters (QBs never win a FLEX spot): replacement = avg of QB13-15
    assert repl["QB"] == pytest.approx((380 - 6 * 12 + 380 - 6 * 13 + 380 - 6 * 14) / 3)
    # RB/WR/TE share 12 FLEX spots, so their cutoffs sit deeper than 24 / 24 / 12
    assert repl["RB"] <= 300 - 6 * 24


def test_league_scoring_factor_uses_the_stat_mix():
    b = pl.DataFrame({"player_id": ["rb", "wr"], "position": ["RB", "WR"], "season_b": [200.0, 200.0],
                      "pg_rushing_yards": [80.0, 0.0], "pg_rushing_tds": [0.6, 0.0], "pg_receptions": [2.0, 7.0],
                      "pg_receiving_yards": [15.0, 80.0], "pg_receiving_tds": [0.1, 0.5]})
    half = league_points(b, PRESETS["half-ppr"]).sort("player_id")["points"].to_list()
    rb, wr = half
    assert rb < 200 and wr < 200   # half PPR lowers both
    assert wr < rb                 # but the 7-catch receiver loses more than the 2-catch back


def test_tiers_break_at_big_drops():
    b = pl.DataFrame({"player_id": list("abcdef"), "position": ["RB"] * 6,
                      "points": [300.0, 296.0, 292.0, 250.0, 246.0, 242.0]})
    t = tiers(b, {"RB": 5}).sort("points", descending=True)["tier"].to_list()
    assert t == [1, 1, 1, 2, 2, 2]  # the 42-point cliff after the 3rd back starts tier 2
