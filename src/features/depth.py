"""Depth-chart features: is he the starter going into this game?

Why: the model's usage features are averages from games a player DID play. A benched starter or a
backup QB who started earlier still looks like a starter (Week 11 2026: Kyler Murray, MIN QB2, was
projected QB13). The team's pre-game depth chart says who's starting now.

    depth_rank     1 = starter at his spot, 2 = first backup, ... (best rank across the spots he's
                   listed at); blank if he isn't on the chart
    depth_starter  depth_rank == 1

Source and pre-game timing: src/ingest/depth_charts.py. Check: the QB who actually started was
listed #1 before the game in 83-91% of games (the misses are late-week injuries and benchings, which
a pre-game chart can't know - evidence it isn't filled in after the fact).

Usage:
    python -m src.features.depth
"""

from pathlib import Path

import polars as pl

from src.features.team import read_schedules
from src.ingest.depth_charts import read_weekly_depth

PLAYER_WEEKS_PATH = Path("data/processed/player_weeks.parquet")
OUT_PATH = Path("data/processed/features_depth.parquet")
KEY = ["player_id", "season", "week"]
FEATURES = ["depth_rank", "depth_starter"]


def build_depth_features(pw: pl.DataFrame, depth: pl.DataFrame) -> pl.DataFrame:
    return pw.select(KEY).join(depth, on=KEY, how="left").with_columns(
        depth_starter=pl.when(pl.col("depth_rank").is_null()).then(None).otherwise(pl.col("depth_rank") == 1)
    ).sort(KEY)


def read_depth(pw: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    return read_weekly_depth(sorted(pw["season"].unique().to_list()), schedules)


def main() -> None:
    pw = pl.read_parquet(PLAYER_WEEKS_PATH)
    feats = build_depth_features(pw, read_depth(pw, read_schedules()))
    if feats.height != pw.height:
        raise SystemExit(f"CHECK FAILED: {feats.height} rows vs {pw.height} player_weeks rows")
    feats.write_parquet(OUT_PATH)
    filled = feats.group_by("season").agg(pl.col("depth_rank").is_not_null().mean().round(2)).sort("season")
    print(f"Wrote {OUT_PATH}: {feats.height:,} rows; share with a depth rank: {dict(filled.rows())}")


if __name__ == "__main__":
    main()
