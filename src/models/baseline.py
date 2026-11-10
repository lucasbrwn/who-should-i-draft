"""Baseline projection model (SCRUM-40): a recency-weighted average of each player's recent games.

For every target stat, the prediction is a weighted average of the player's last 8 games (across
seasons, so Week 1 uses last year). A game counts half as much as one 3 games more recent.
Players with no history get their position's average. Every real model must beat this.

All models share one interface, used by the backtest harness (src/backtest/harness.py):
    model.fit(history)          history = every player-game before the week being predicted
    model.predict(rows) -> df   rows = this week's player-games; returns player_id + one column per target
"""

import polars as pl

from src.features.assemble import TARGETS

COMPONENTS = [t for t in TARGETS if t != "fantasy_points_ppr"]  # points come from the scoring engine
LAST_N_GAMES = 8
HALF_LIFE_GAMES = 3


class Baseline:
    name = "baseline"

    def fit(self, history: pl.DataFrame) -> "Baseline":
        games = history.filter(~pl.col("left_injured")).sort("player_id", "season", "week", descending=[False, True, True])
        recent = games.with_columns(age=pl.col("week").cum_count().over("player_id") - 1).filter(pl.col("age") < LAST_N_GAMES)
        weight = 0.5 ** (pl.col("age") / HALF_LIFE_GAMES)
        self.player_avgs = recent.group_by("player_id").agg(
            [((pl.col(f"y_{c}") * weight).sum() / weight.sum()).alias(c) for c in COMPONENTS]
        )
        self.position_avgs = games.group_by("position").agg([pl.col(f"y_{c}").mean().alias(c) for c in COMPONENTS])
        return self

    def predict(self, rows: pl.DataFrame) -> pl.DataFrame:
        by_player = rows.select("player_id", "position").join(self.player_avgs, on="player_id", how="left")
        by_position = rows.select("position").join(self.position_avgs, on="position", how="left")
        return by_player.select(
            "player_id",
            *[pl.coalesce(pl.col(c), by_position[c]).alias(c) for c in COMPONENTS],
        )
