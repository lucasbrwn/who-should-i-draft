"""Walk-forward backtest harness (SCRUM-42).

For each week of 2021-2024: fit the model on every game BEFORE that week, predict that week,
turn predicted stats into fantasy points with our scoring engine, and compare to what happened.
2025 is never touched (final exam, run once when the model is frozen).

Metrics per position (averaged over weeks):
    mae        average fantasy points off per player
    spearman   rank correlation within the week: did we order the players right?
    top_n_hit  of the players we projected as starters (top 12 QB/TE, top 24 RB/WR),
               the share who finished as starters
95% confidence intervals come from resampling whole weeks (bootstrap), and model-vs-model
comparisons resample the SAME weeks for both (paired), so the interval is on the difference.

Every game played is scored, including games where the player left injured: at prediction
time nobody knows who will get hurt. (Training leaves those games out.)
"""

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from src.scoring.config import PRESETS
from src.scoring.engine import add_fantasy_points

SEASONS = [2021, 2022, 2023, 2024]
FINAL_EXAM_SEASON = 2025
POSITIONS = ["QB", "RB", "WR", "TE"]
STARTERS = {"QB": 12, "TE": 12, "RB": 24, "WR": 24}
SCORING = PRESETS["standard-ppr"].scoring
N_BOOT = 2000


def points(stats: pl.DataFrame) -> pl.Series:
    """Fantasy points from a table of stat columns (nflverse names) plus 'position'."""
    return add_fantasy_points(stats, SCORING)["points"]


def actual_stats(rows: pl.DataFrame) -> pl.DataFrame:
    """The real stats, from the y_ target columns, renamed back to nflverse names."""
    return rows.select("position", *[pl.col(c).alias(c[2:]) for c in rows.columns if c.startswith("y_")])


def run(model, feats: pl.DataFrame, seasons: list[int] = SEASONS) -> pl.DataFrame:
    """Walk forward through every week; return one row per player-game with predicted + actual points."""
    if FINAL_EXAM_SEASON in seasons:
        raise ValueError("2025 is the final exam: run it once, on purpose, not in the backtest")
    out = []
    for season in seasons:
        for week in sorted(feats.filter(pl.col("season") == season)["week"].unique()):
            before = (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week))
            rows = feats.filter((pl.col("season") == season) & (pl.col("week") == week))
            preds = model.fit(feats.filter(before)).predict(rows)
            pred_stats = rows.select("player_id", "position").join(preds, on="player_id", how="left")
            out.append(rows.select("player_id", "player_display_name", "position", "season", "week").with_columns(
                pred=points(pred_stats.drop("player_id")),
                actual=points(actual_stats(rows)),
            ))
    return pl.concat(out).with_columns(model=pl.lit(model.name))


def weekly_metrics(results: pl.DataFrame) -> pl.DataFrame:
    """One row per model x position x week: mae, spearman, top_n_hit."""
    rows = []
    for (model, pos, season, week), g in results.group_by("model", "position", "season", "week"):
        n = STARTERS[pos]
        if g.height < n * 2:
            continue
        pred, actual = g["pred"].to_numpy(), g["actual"].to_numpy()
        projected_starters = set(g.sort("pred", descending=True)["player_id"][:n])
        real_starters = set(g.sort("actual", descending=True)["player_id"][:n])
        rows.append({
            "model": model, "position": pos, "season": season, "week": week,
            "mae": float(np.abs(pred - actual).mean()),
            "spearman": float(spearmanr(pred, actual).statistic),
            "top_n_hit": len(projected_starters & real_starters) / n,
        })
    return pl.DataFrame(rows)


@dataclass
class Interval:
    mean: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.low:.3f}, {self.high:.3f}]"


def bootstrap(values: np.ndarray, seed: int = 0) -> Interval:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(N_BOOT, len(values)), replace=True).mean(axis=1)
    return Interval(float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def summarize(weekly: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for (model, pos), g in weekly.group_by("model", "position"):
        rows.append({"model": model, "position": pos, "weeks": g.height,
                     **{m: str(bootstrap(g[m].to_numpy())) for m in ["mae", "spearman", "top_n_hit"]}})
    return pl.DataFrame(rows).sort("position", "model")


def compare(weekly: pl.DataFrame, a: str, b: str) -> pl.DataFrame:
    """Paired difference (a - b) per position, resampling the same weeks for both models."""
    rows = []
    for pos in POSITIONS:
        wide = weekly.filter(pl.col("position") == pos).pivot(on="model", index=["season", "week"], values=["mae", "spearman"])
        row = {"position": pos}
        for m in ["mae", "spearman"]:
            diff = (wide[f"{m}_{a}"] - wide[f"{m}_{b}"]).drop_nulls().to_numpy()
            row[f"{m}_diff"] = str(bootstrap(diff))
        rows.append(row)
    return pl.DataFrame(rows)
