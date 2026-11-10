"""Usage and recent-form features (SCRUM-34): how a player was used BEFORE each week.

One row per row of player_weeks. Every feature on a week-N row comes only from games before
week N, so the model never sees the game it is predicting (no leakage).

Windows (all averages over games actually played, so byes and missed games are skipped, not zeros):
    *_l3, *_l5  last 3 / last 5 games this season
    *_std       season to date
    *_prev      the whole previous season (the model learns how much to trust it early in
                the year, using games_std; we don't hand-pick blend weights)

Games where the player left injured (left_injured) are left out of every average: a 3%-snap
game says nothing about his normal role.

Usage:
    python -m src.features.usage
"""

from pathlib import Path

import polars as pl

PLAYER_WEEKS_PATH = Path("data/processed/player_weeks.parquet")
OUT_PATH = Path("data/processed/features_usage.parquet")

KEY = ["player_id", "season", "week"]
GROUP = ["player_id", "season"]
ID_COLS = ["player_id", "player_display_name", "position", "season", "week", "game_id", "team"]

# Weighted opportunity (literature review): in PPR a target is worth ~1.59 points, a carry ~0.58.
TARGET_WEIGHT = 1.59
CARRY_WEIGHT = 0.58

# Averaged per game.
MEAN_STATS = [
    "offense_pct", "target_share", "air_yards_share", "wopr", "carry_share", "weighted_opportunity",
    "targets", "carries", "receptions", "receiving_yards", "rushing_yards",
    "attempts", "passing_yards",
    "total_fantasy_points_exp", "total_touchdown_exp", "fantasy_points_ppr",
]

# Efficiency: total numerator / total denominator over the window (not an average of
# per-game ratios, which would let a 1-carry game count as much as a 20-carry game).
RATIO_STATS = {
    "yards_per_touch": ("scrimmage_yards", "touches"),
    "rushing_epa_per_carry": ("rushing_epa", "carries"),
    "receiving_epa_per_target": ("receiving_epa", "targets"),
    "passing_epa_per_dropback": ("passing_epa", "dropbacks"),
}

SUM_COLS = list(dict.fromkeys(MEAN_STATS + [c for pair in RATIO_STATS.values() for c in pair]))
WINDOWS = {"l3": 3, "l5": 5}


def add_game_columns(pw: pl.DataFrame) -> pl.DataFrame:
    """Per-game columns the features need that player_weeks doesn't have yet."""
    team_carries = pl.col("carries").sum().over("game_id", "team")
    return pw.with_columns(
        carry_share=pl.when(team_carries > 0).then(pl.col("carries") / team_carries).otherwise(0.0),
        weighted_opportunity=TARGET_WEIGHT * pl.col("targets") + CARRY_WEIGHT * pl.col("carries"),
        touches=pl.col("carries") + pl.col("receptions"),
        scrimmage_yards=pl.col("rushing_yards") + pl.col("receiving_yards"),
        dropbacks=pl.col("attempts") + pl.col("sacks_suffered"),
    )


def finish(frame: pl.DataFrame, suffix: str, games_col: str) -> pl.DataFrame:
    """Turn window sums (columns named '<stat>__sum') into averages and ratios named '<stat>_<suffix>'."""
    s = lambda c: pl.col(f"{c}__sum")  # noqa: E731
    means = [(s(m) / pl.col(games_col)).alias(f"{m}_{suffix}") for m in MEAN_STATS]
    ratios = [
        pl.when(s(den) > 0).then(s(num) / s(den)).alias(f"{name}_{suffix}")
        for name, (num, den) in RATIO_STATS.items()
    ]
    # Round off floating-point noise: a rolling sum adds the new game and subtracts the old one,
    # so a true 0 can come out as -1e-17.
    return frame.with_columns([e.round(9) for e in means + ratios]).drop([f"{c}__sum" for c in SUM_COLS])


def through_each_game(usable: pl.DataFrame) -> pl.DataFrame:
    """Window values that INCLUDE each game. build_usage_features then attaches each week to the
    most recent game strictly before it, which is what keeps the current game out."""
    usable = usable.sort(KEY)
    one = pl.col("week").is_not_null().cast(pl.Int32)  # 1 per game, for counting games
    out = usable.select(KEY)
    windows = {**WINDOWS, "std": None}
    for suffix, w in windows.items():
        if w is None:
            roll = lambda e: e.cum_sum().over(GROUP)  # noqa: E731
        else:
            roll = lambda e, w=w: e.rolling_sum(w, min_samples=1).over(GROUP)  # noqa: E731
        frame = usable.select(
            KEY
            + [roll(pl.col(c)).alias(f"{c}__sum") for c in SUM_COLS]
            + [roll(one).alias(f"games_{suffix}")]
        )
        frame = finish(frame, suffix, f"games_{suffix}")
        if suffix != "std":
            frame = frame.drop(f"games_{suffix}")  # only the season-to-date count is a feature
        out = out.join(frame, on=KEY)
    return out


def previous_season(usable: pl.DataFrame) -> pl.DataFrame:
    """Each player's full-season averages, keyed to the FOLLOWING season."""
    totals = usable.group_by(GROUP).agg(
        [pl.col(c).sum().alias(f"{c}__sum") for c in SUM_COLS] + [pl.len().alias("games_prev")]
    )
    return finish(totals, "prev", "games_prev").with_columns(pl.col("season") + 1)


def build_usage_features(pw: pl.DataFrame) -> pl.DataFrame:
    pw = add_game_columns(pw)
    usable = pw.filter(~pl.col("left_injured"))

    rows = pw.select(ID_COLS).sort("week")
    # For each row, take the latest game strictly before this week (same player, same season).
    # check_sortedness=False: polars can't verify sortedness within groups; both sides are sorted by week.
    feats = rows.join_asof(
        through_each_game(usable).sort("week"),
        on="week", by=GROUP, strategy="backward", allow_exact_matches=False, check_sortedness=False,
    )
    feats = feats.join(previous_season(usable), on=GROUP, how="left")
    return feats.with_columns(
        pl.col("games_std").fill_null(0), pl.col("games_prev").fill_null(0)
    ).sort(KEY)


def check_usage_features(feats: pl.DataFrame, pw: pl.DataFrame) -> list[str]:
    problems = []
    if feats.height != pw.height:
        problems.append(f"{feats.height} feature rows vs {pw.height} player_weeks rows")
    dupes = feats.group_by(KEY).len().filter(pl.col("len") > 1).height
    if dupes:
        problems.append(f"{dupes} duplicate player-weeks")
    first_games = feats.filter(pl.col("games_std") == 0)
    if first_games.select(pl.col("offense_pct_std").is_not_null().any()).item():
        problems.append("a player's first game of the season has season-to-date features")
    shares = [c for c in feats.columns if c.startswith(("offense_pct_", "target_share_", "carry_share_"))]
    bad = feats.filter(pl.any_horizontal([(pl.col(c) < 0) | (pl.col(c) > 1.0001) for c in shares])).height
    if bad:
        problems.append(f"{bad} rows with a share outside 0-1")
    return problems


def main() -> None:
    pw = pl.read_parquet(PLAYER_WEEKS_PATH)
    feats = build_usage_features(pw)
    problems = check_usage_features(feats, pw)
    for p in problems:
        print("CHECK FAILED:", p)
    if problems:
        raise SystemExit(1)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats.write_parquet(OUT_PATH)
    print(f"Wrote {OUT_PATH}: {feats.height:,} rows x {feats.width} columns")


if __name__ == "__main__":
    main()
