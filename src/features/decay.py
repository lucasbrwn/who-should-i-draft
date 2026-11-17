"""Exponentially weighted recent form (peer suggestion, 2026-11-17).

Our other windows have hard cutoffs: l3 / l5 count the last 3 / 5 games equally and drop everything
older; std counts every game this season equally; prev is last season. An exponentially weighted
average does it smoothly: each older game counts a fixed fraction less.

    HALF_LIFE = 3 games: last game weight 1, three games back 1/2, six back 1/4, nine back 1/8 ...

Chosen before testing (not tuned), so the A/B can't overfit it.

Carries across seasons on purpose: in week 1 it is mostly last season's final games, and it shifts
to this season as games are played (the smooth version of blending std with prev). Like every other
feature it looks back at most to LAST season (the preseason board in src/models/season.py rebuilds
features from two seasons only; older games would carry ~2% weight anyway). Games where the
player left injured are skipped, like every other window. Each row only sees games BEFORE its week.

Usage:
    python -m src.features.decay
"""

from pathlib import Path

import polars as pl

from src.features.usage import ID_COLS, KEY, MEAN_STATS, add_game_columns

PLAYER_WEEKS_PATH = Path("data/processed/player_weeks.parquet")
OUT_PATH = Path("data/processed/features_decay.parquet")
HALF_LIFE = 3
FEATURES = [f"{s}_ewm" for s in MEAN_STATS]


def through_each_game(usable: pl.DataFrame) -> pl.DataFrame:
    """Weighted average INCLUDING each game, over the player's whole career so far."""
    usable = usable.sort("player_id", "season", "week")
    return usable.select(
        "player_id", "t",
        *[pl.col(s).ewm_mean(half_life=HALF_LIFE, adjust=True, ignore_nulls=True).over("player_id").alias(f"{s}_ewm")
          for s in MEAN_STATS],
    )


def build_decay_features(pw: pl.DataFrame) -> pl.DataFrame:
    pw = add_game_columns(pw).with_columns(t=pl.col("season") * 100 + pl.col("week"))
    # games with a stat line only (blank upcoming rows have nothing to average)
    usable = pw.filter(~pl.col("left_injured") & pl.col("fantasy_points_ppr").is_not_null())
    out = []
    for season in sorted(pw["season"].unique().to_list()):
        rows = pw.filter(pl.col("season") == season).select(ID_COLS + ["t"]).sort("t")
        history = through_each_game(usable.filter(pl.col("season").is_in([season - 1, season]))).sort("t")
        # latest game strictly before this row's week, same player, this season or last
        out.append(rows.join_asof(history, on="t", by="player_id", strategy="backward",
                                  allow_exact_matches=False, check_sortedness=False))
    feats = pl.concat(out)
    return feats.with_columns([pl.col(f).round(9) for f in FEATURES]).drop("t").sort(KEY)


def check_decay_features(feats: pl.DataFrame, pw: pl.DataFrame) -> list[str]:
    problems = []
    if feats.height != pw.height:
        problems.append(f"{feats.height} rows vs {pw.height} player_weeks rows")
    if feats.group_by(KEY).len().filter(pl.col("len") > 1).height:
        problems.append("duplicate player-weeks")
    return problems


def main() -> None:
    pw = pl.read_parquet(PLAYER_WEEKS_PATH)
    feats = build_decay_features(pw)
    problems = check_decay_features(feats, pw)
    for p in problems:
        print("CHECK FAILED:", p)
    if problems:
        raise SystemExit(1)
    feats.write_parquet(OUT_PATH)
    filled = feats.group_by("season").agg(pl.col("fantasy_points_ppr_ewm").is_not_null().mean().round(2)).sort("season")
    print(f"Wrote {OUT_PATH}: {feats.height:,} rows x {feats.width} columns")
    print("share of rows with a weighted average:", dict(filled.rows()))


if __name__ == "__main__":
    main()
