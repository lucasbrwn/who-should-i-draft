"""Defensive ratings (SCRUM-36): how good each defense was BEFORE each game.

One row per defense per scheduled regular-season game (upcoming games included, so live
predictions get a rating too). Every rating comes only from the defense's earlier games.

Ratings (lower = better defense, except sack and QB hit rates where higher = better):
    epa_per_dropback   EPA allowed per opponent dropback
    epa_per_rush       EPA allowed per designed run
    epa_short, epa_deep  EPA allowed per short / deep pass attempt
    sack_rate, qb_hit_rate  per opponent dropback (pressure data stops at 2025, these don't)
    (all of the above leave out garbage time: offense win probability under 5% or over 95%)
    fpoe_qb/rb/wr/te   fantasy points ABOVE EXPECTED allowed to each position per game.
                       Raw points allowed mostly reflect how often teams threw at a defense;
                       actual minus expected (from ff_opportunity) measures the defense itself.

Windows: _l5 (last 5 games), _std (season to date), _prev (whole previous season).

Usage:
    python -m src.features.defense
"""

from pathlib import Path

import polars as pl

from src.features.team import read_schedules
from src.ingest.pbp import read_pbp

PLAYER_WEEKS_PATH = Path("data/processed/player_weeks.parquet")
OUT_PATH = Path("data/processed/features_defense.parquet")

DEF_KEY = ["defteam", "season", "week"]
GROUP = ["defteam", "season"]
POSITIONS = ["QB", "RB", "WR", "TE"]
PBP_COLS = ["game_id", "season_type", "defteam", "play_type", "qb_dropback", "rush_attempt", "pass_attempt",
            "pass_length", "epa", "sack", "qb_hit", "wp"]
# Garbage time: leave out plays where the offense's win probability is below 5% or above 95%
# (decided 2026-11-10). Prevent defenses give up easy yards there, and pass rushers tee off on
# teams that must throw, so those plays say little about how good a defense is.
WP_MIN, WP_MAX = 0.05, 0.95

# rating name -> (numerator column, denominator column). Per-game sums are rolled up, then divided,
# so a 60-dropback game counts more than a 20-dropback game.
RATIOS = {
    "epa_per_dropback": ("epa_db", "dropbacks"),
    "epa_per_rush": ("epa_rush", "rushes"),
    "epa_short": ("epa_short", "short_att"),
    "epa_deep": ("epa_deep", "deep_att"),
    "sack_rate": ("sacks", "dropbacks"),
    "qb_hit_rate": ("qb_hits", "dropbacks"),
}
PER_GAME = [f"fpoe_{p.lower()}" for p in POSITIONS]
SUM_COLS = list(dict.fromkeys([c for pair in RATIOS.values() for c in pair] + PER_GAME))


def defense_plays(pbp: pl.DataFrame) -> pl.DataFrame:
    """Per defense-game sums from play-by-play, garbage time excluded."""
    plays = pbp.filter(
        (pl.col("season_type") == "REG") & pl.col("defteam").is_not_null() & pl.col("play_type").is_in(["pass", "run"])
        & pl.col("wp").fill_null(0.5).is_between(WP_MIN, WP_MAX)
    )
    db = pl.col("qb_dropback") == 1
    rush = (pl.col("rush_attempt") == 1) & ~db           # designed runs (scrambles are dropbacks)
    short = (pl.col("pass_attempt") == 1) & (pl.col("pass_length") == "short")
    deep = (pl.col("pass_attempt") == 1) & (pl.col("pass_length") == "deep")
    epa = pl.col("epa").fill_null(0)
    return plays.group_by("game_id", "defteam").agg(
        dropbacks=db.sum(), epa_db=epa.filter(db).sum(),
        rushes=rush.sum(), epa_rush=epa.filter(rush).sum(),
        short_att=short.sum(), epa_short=epa.filter(short).sum(),
        deep_att=deep.sum(), epa_deep=epa.filter(deep).sum(),
        sacks=(db & (pl.col("sack") == 1)).sum(), qb_hits=(db & (pl.col("qb_hit") == 1)).sum(),
    )


def fantasy_over_expected_allowed(pw: pl.DataFrame) -> pl.DataFrame:
    """Per defense-game: sum over opposing players of (actual - expected) PPR points, by position."""
    over = pl.col("fantasy_points_ppr") - pl.col("total_fantasy_points_exp")
    return pw.group_by("game_id", pl.col("opponent_team").alias("defteam")).agg(
        [over.filter(pl.col("position") == p).sum().alias(f"fpoe_{p.lower()}") for p in POSITIONS]
    )


def defense_games(schedules: pl.DataFrame) -> pl.DataFrame:
    """Every scheduled game twice: once for each defense."""
    cols = ["game_id", "season", "week"]
    return pl.concat([
        schedules.select(cols, defteam=pl.col("home_team"), offense=pl.col("away_team")),
        schedules.select(cols, defteam=pl.col("away_team"), offense=pl.col("home_team")),
    ])


def finish(frame: pl.DataFrame, suffix: str, games_col: str) -> pl.DataFrame:
    s = lambda c: pl.col(f"{c}__sum")  # noqa: E731
    ratios = [pl.when(s(den) > 0).then(s(num) / s(den)).round(9).alias(f"{name}_{suffix}")
              for name, (num, den) in RATIOS.items()]
    per_game = [(s(c) / pl.col(games_col)).round(9).alias(f"{c}_{suffix}") for c in PER_GAME]
    return frame.with_columns(ratios + per_game).drop([f"{c}__sum" for c in SUM_COLS] + [games_col])


def through_each_game(played: pl.DataFrame) -> pl.DataFrame:
    played = played.sort(DEF_KEY)
    one = pl.col("week").is_not_null().cast(pl.Int32)
    out = played.select(DEF_KEY)
    for suffix, w in {"l5": 5, "std": None}.items():
        roll = (lambda e: e.cum_sum().over(GROUP)) if w is None else (lambda e, w=w: e.rolling_sum(w, min_samples=1).over(GROUP))
        frame = played.select(DEF_KEY + [roll(pl.col(c).fill_null(0)).alias(f"{c}__sum") for c in SUM_COLS]
                              + [roll(one).alias("n")])
        out = out.join(finish(frame, suffix, "n"), on=DEF_KEY)
    return out


def previous_season(played: pl.DataFrame) -> pl.DataFrame:
    totals = played.group_by(GROUP).agg([pl.col(c).fill_null(0).sum().alias(f"{c}__sum") for c in SUM_COLS] + [pl.len().alias("n")])
    return finish(totals, "prev", "n").with_columns(pl.col("season") + 1)


def build_defense_features(schedules: pl.DataFrame, pbp_sums: pl.DataFrame, fpoe: pl.DataFrame) -> pl.DataFrame:
    games = defense_games(schedules)
    played = games.join(pbp_sums, on=["game_id", "defteam"], how="inner").join(fpoe, on=["game_id", "defteam"], how="left")
    feats = games.sort("week").join_asof(
        through_each_game(played).sort("week"),
        on="week", by=GROUP, strategy="backward", allow_exact_matches=False, check_sortedness=False,
    )
    return feats.join(previous_season(played), on=GROUP, how="left").sort(DEF_KEY)


def check_defense_features(feats: pl.DataFrame) -> list[str]:
    problems = []
    if feats.group_by("game_id", "defteam").len().filter(pl.col("len") > 1).height:
        problems.append("duplicate defense-games")
    rates = [c for c in feats.columns if c.startswith(("sack_rate_", "qb_hit_rate_"))]
    if feats.filter(pl.any_horizontal([(pl.col(c) < 0) | (pl.col(c) > 1) for c in rates])).height:
        problems.append("a sack or QB hit rate outside 0-1")
    return problems


def main() -> None:
    pw = pl.read_parquet(PLAYER_WEEKS_PATH)
    feats = build_defense_features(read_schedules(), defense_plays(read_pbp(columns=PBP_COLS)),
                                   fantasy_over_expected_allowed(pw))
    problems = check_defense_features(feats)
    for p in problems:
        print("CHECK FAILED:", p)
    if problems:
        raise SystemExit(1)
    feats.write_parquet(OUT_PATH)
    print(f"Wrote {OUT_PATH}: {feats.height:,} defense-games x {feats.width} columns")


if __name__ == "__main__":
    main()
