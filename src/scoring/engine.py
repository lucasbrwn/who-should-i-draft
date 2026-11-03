"""Scoring engine: turn stat lines into fantasy points for any league's scoring rules.

Works on any table with nflverse stat column names: real weekly stats from M1, or predicted
component stats from M4. Stats a table doesn't have count as zero, so a model that only predicts
yards and TDs still gets scored.

Usage:
    from src.scoring.config import PRESETS
    scored = add_fantasy_points(stats, PRESETS["half-ppr"].scoring)
"""

from dataclasses import fields

import polars as pl

from src.scoring.config import Scoring

# Settings that are simply (points per unit) x (stat column of the same name).
PER_UNIT = [
    "passing_yards", "passing_tds", "passing_interceptions", "passing_2pt_conversions",
    "rushing_yards", "rushing_tds", "rushing_2pt_conversions",
    "receptions", "receiving_yards", "receiving_tds", "receiving_2pt_conversions",
    "special_teams_tds",
]

# Fumbles: use a total column when the table has one, else add up the scrimmage parts.
# nflverse's *_total columns also count return fumbles (e.g. Skyy Moore's muffed punt, 2022 wk 3),
# which ESPN and Sleeper penalize. The parts alone miss those (~0.5% of offensive player-weeks).
FUMBLE_LOST_TOTALS = ["fumbles_lost_total", "fumbles_lost"]
FUMBLE_TOTALS = ["fumbles_total", "fumbles"]
FUMBLE_LOST_PARTS = ["sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"]
FUMBLE_PARTS = ["sack_fumbles", "rushing_fumbles", "receiving_fumbles"]

# Bonus setting -> (stat column, threshold). Awarded once when the stat reaches the threshold.
BONUSES = {
    "bonus_pass_300": ("passing_yards", 300),
    "bonus_rush_100": ("rushing_yards", 100),
    "bonus_rec_100": ("receiving_yards", 100),
}

# Every Scoring field must be handled by exactly one rule above (a test enforces this).
HANDLED = set(PER_UNIT) | set(BONUSES) | {"fumbles_lost", "fumbles", "te_reception_bonus"}


def _stat(columns: set[str], name: str) -> pl.Expr:
    """The column as a float with nulls as 0, or a literal 0 if the table doesn't have it."""
    if name in columns:
        return pl.col(name).cast(pl.Float64).fill_null(0.0)
    return pl.lit(0.0)


def _total_or_parts(columns: set[str], totals: list[str], parts: list[str]) -> pl.Expr:
    """The first total column the table has, otherwise the sum of the parts (never both)."""
    for total in totals:
        if total in columns:
            return _stat(columns, total)
    return pl.sum_horizontal(_stat(columns, c) for c in parts)


def points_expr(scoring: Scoring, columns: set[str] | list[str]) -> pl.Expr:
    """A polars expression computing fantasy points for one row, given the table's columns."""
    columns = set(columns)
    terms = [_stat(columns, name) * getattr(scoring, name) for name in PER_UNIT]

    terms.append(_total_or_parts(columns, FUMBLE_LOST_TOTALS, FUMBLE_LOST_PARTS) * scoring.fumbles_lost)
    # A lost fumble is also a fumble, so a league scoring both takes both penalties.
    terms.append(_total_or_parts(columns, FUMBLE_TOTALS, FUMBLE_PARTS) * scoring.fumbles)

    if scoring.te_reception_bonus:
        if "position" not in columns:
            raise ValueError("TE premium needs a 'position' column to know who is a TE")
        is_te = (pl.col("position") == "TE").cast(pl.Float64)
        terms.append(_stat(columns, "receptions") * is_te * scoring.te_reception_bonus)

    for setting_name, (stat, threshold) in BONUSES.items():
        bonus = getattr(scoring, setting_name)
        if bonus:
            reached = (_stat(columns, stat) >= threshold).cast(pl.Float64)
            terms.append(reached * bonus)

    return pl.sum_horizontal(terms)


def add_fantasy_points(stats: pl.DataFrame, scoring: Scoring, column: str = "points") -> pl.DataFrame:
    """Return `stats` with a new `column` of fantasy points under `scoring`."""
    return stats.with_columns(points_expr(scoring, stats.columns).alias(column))


def unhandled_settings() -> set[str]:
    """Scoring fields the engine doesn't know how to apply (should always be empty)."""
    return {f.name for f in fields(Scoring)} - HANDLED
