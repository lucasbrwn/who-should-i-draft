"""FantasyPros weekly expert rank as a model input (2026-11-18, Lucas's idea: "feed FP rankings into our model").

    fp_ecr   the player's FantasyPros weekly PPR position ECR (expert consensus rank, e.g. 12.4 = about
             WR12), from the Friday snapshot inside that week. Blank when:
               - his game was on or before the snapshot day (Thursday games: the experts may have seen it)
               - FantasyPros didn't rank him (deep backups) or no snapshot exists that week
             Trees send blanks down their own branch, so "the experts don't rank him" is information too.

Why FantasyPros and not ESPN: our ESPN history is ESPN's FINAL number (can include post-Friday news),
so a model trained on it would learn to over-trust ESPN. The FP history is a dated pre-game snapshot.
Never fit on 2025 rows' targets by the harness (2025 is the final exam); 2025 ranks may exist but are
only seen if a run includes 2025.

RESULT 2026-11-18 (src/backtest/expert_feature_test.py): FAILED its bar -> NOT in assemble.build_all.
With fp_ecr as an input the model went 49.9% vs FantasyPros (same as without it) and its MAE (4.065)
was far worse than the 1/3 blend (3.953). The model barely changes its calls with the rank: ECR is one of
185 inputs to per-stat models trained mostly on seasons without it. The blend stays the product number.
"""

import polars as pl

from src.backtest.weekly_vs_consensus import game_days, weekly_ecr

KEY = ["player_id", "season", "week"]
FEATURES = ["fp_ecr"]


def build_expert_features(seasons: list[int]) -> pl.DataFrame:
    days = game_days(seasons)
    return (weekly_ecr(seasons).join(days, on=KEY, how="inner")
            .filter(pl.col("gameday") > pl.col("d"))
            .select(*KEY, fp_ecr=pl.col("rank").round(2))
            .unique(KEY, keep="first"))
