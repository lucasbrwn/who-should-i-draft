"""QB rushing split: designed runs vs scrambles (2026-11-18, weak-points follow-up).

Why: QB is ESPN's biggest edge on us (MAE 5.71 vs 5.39, 2022-24) and TDs are 58% of QB points
error. Our usage features treat every QB carry the same. A designed run is a coach's call and
repeats week to week; a scramble depends on how a play breaks down and is noisier. Splitting them
(and counting runs near the goal line) may let the model separate a Hurts/Allen-style rusher
from a pocket QB who happened to scramble a few times.

Per game, from play-by-play (regular season, no 2-point tries, no kneels or spikes):
    designed run   play_type 'run', not a scramble, he is the rusher
    scramble       qb_scramble == 1, he is the rusher
    dropback       qb_dropback == 1, he is the passer (or the scrambler)

Features (decayed, half-life 3 games, this season + last like decay.py; only games BEFORE the row):
    qb_designed_runs_ewm        designed carries per game
    qb_scrambles_ewm            scrambles per game
    qb_scramble_rate_ewm        scrambles / dropbacks (decayed sums)
    qb_designed_rush_yds_ewm    yards on designed runs per game
    qb_rz_designed_runs_ewm     designed runs inside the opponent's 20 per game
    qb_gl_runs_ewm              any run (designed or scramble) inside the 5 per game
QBs only: other positions get nulls (the trees send them down the "missing" branch).

Rule locked 2026-11-17 BEFORE testing (walk-forward 2021-24, boosting vs boosting_no_qbrush):
KEEP if the QB MAE diff CI is entirely < 0 or the QB Spearman diff CI is entirely > 0, AND no
position is significantly worse. Otherwise (a tie, or better but not significant) DROP.

RESULT 2026-11-18: tie -> NOT used (removed from assemble.build_all). Without these columns QB MAE
was 0.006 lower [-0.023, +0.011] and Spearman 0.001 higher [-0.004, +0.005]; RB/WR/TE identical.
The existing carries / rushing-yards / rushing-EPA windows already carry the QB rushing signal.
PBP_COLS stays in assemble.PBP_COLS so the A/B can be re-run.

    python -m src.features.qb_rush
"""

from pathlib import Path

import polars as pl

PLAYER_WEEKS_PATH = Path("data/processed/player_weeks.parquet")
OUT_PATH = Path("data/processed/features_qb_rush.parquet")
PBP_COLS = ["game_id", "season_type", "play_type", "yardline_100", "rusher_player_id", "passer_player_id",
            "qb_scramble", "qb_dropback", "qb_spike", "two_point_attempt", "rushing_yards"]
KEY = ["player_id", "season", "week"]
HALF_LIFE = 3
COUNTS = ["designed_runs", "scrambles", "dropbacks", "designed_rush_yds", "rz_designed_runs", "gl_runs"]
FEATURES = ["qb_designed_runs_ewm", "qb_scrambles_ewm", "qb_scramble_rate_ewm", "qb_designed_rush_yds_ewm",
            "qb_rz_designed_runs_ewm", "qb_gl_runs_ewm"]


def game_counts(pbp: pl.DataFrame) -> pl.DataFrame:
    """Per (game, player): designed runs, scrambles, dropbacks, designed rush yards, red-zone / goal-line runs."""
    plays = pbp.filter((pl.col("season_type") == "REG") & (pl.col("two_point_attempt").fill_null(0) != 1)
                       & (pl.col("qb_spike").fill_null(0) != 1))
    scramble = pl.col("qb_scramble").fill_null(0) == 1
    runs = plays.filter((pl.col("play_type") == "run") & pl.col("rusher_player_id").is_not_null()).select(
        "game_id", player_id="rusher_player_id",
        designed_runs=(~scramble).cast(pl.Int32),
        scrambles=scramble.cast(pl.Int32),
        dropbacks=pl.lit(0),  # scramble dropbacks are counted below from qb_dropback
        designed_rush_yds=pl.when(~scramble).then(pl.col("rushing_yards").fill_null(0)).otherwise(0),
        rz_designed_runs=((~scramble) & (pl.col("yardline_100") <= 20)).cast(pl.Int32),
        gl_runs=(pl.col("yardline_100") <= 5).cast(pl.Int32),
    )
    drops = plays.filter(pl.col("qb_dropback") == 1).select(
        "game_id", player_id=pl.coalesce("passer_player_id", "rusher_player_id"),
        designed_runs=pl.lit(0), scrambles=pl.lit(0), dropbacks=pl.lit(1), designed_rush_yds=pl.lit(0),
        rz_designed_runs=pl.lit(0), gl_runs=pl.lit(0),
    ).filter(pl.col("player_id").is_not_null())
    both = pl.concat([runs, drops], how="vertical_relaxed")
    return both.group_by("game_id", "player_id").agg([pl.col(c).sum().cast(pl.Float64) for c in COUNTS])


def through_each_game(games: pl.DataFrame) -> pl.DataFrame:
    """Decayed values INCLUDING each game."""
    games = games.sort("player_id", "season", "week")
    ewm = lambda c: pl.col(c).ewm_mean(half_life=HALF_LIFE, adjust=True).over("player_id")  # noqa: E731
    return games.select(
        "player_id", "t",
        qb_designed_runs_ewm=ewm("designed_runs"),
        qb_scrambles_ewm=ewm("scrambles"),
        qb_scramble_rate_ewm=pl.when(ewm("dropbacks") > 0).then(ewm("scrambles") / ewm("dropbacks")),
        qb_designed_rush_yds_ewm=ewm("designed_rush_yds"),
        qb_rz_designed_runs_ewm=ewm("rz_designed_runs"),
        qb_gl_runs_ewm=ewm("gl_runs"),
    )


def build_qb_rush_features(pw: pl.DataFrame, pbp: pl.DataFrame) -> pl.DataFrame:
    pw = pw.with_columns(t=pl.col("season") * 100 + pl.col("week"))
    qb_games = (pw.filter((pl.col("position") == "QB") & ~pl.col("left_injured")
                          & pl.col("fantasy_points_ppr").is_not_null())
                .select("player_id", "season", "week", "game_id", "t")
                .join(game_counts(pbp), on=["game_id", "player_id"], how="left")
                .with_columns([pl.col(c).fill_null(0.0) for c in COUNTS]))
    out = []
    for season in sorted(pw["season"].unique().to_list()):
        rows = pw.filter(pl.col("season") == season).select(KEY + ["position", "t"]).sort("t")
        hist = through_each_game(qb_games.filter(pl.col("season").is_in([season - 1, season]))).sort("t")
        out.append(rows.join_asof(hist, on="t", by="player_id", strategy="backward",
                                  allow_exact_matches=False, check_sortedness=False))
    feats = pl.concat(out, how="diagonal_relaxed")
    is_qb = pl.col("position") == "QB"
    return feats.select(*KEY, *[pl.when(is_qb).then(pl.col(f).round(9)).alias(f) for f in FEATURES]).sort(KEY)


def main() -> None:
    from src.ingest.pbp import read_pbp

    pw = pl.read_parquet(PLAYER_WEEKS_PATH)
    seasons = sorted(pw["season"].unique().to_list())
    feats = build_qb_rush_features(pw, read_pbp(seasons, columns=PBP_COLS))
    if feats.height != pw.height or feats.group_by(KEY).len().filter(pl.col("len") > 1).height:
        raise SystemExit("CHECK FAILED: qb_rush rows don't match player_weeks one-to-one")
    feats.write_parquet(OUT_PATH)
    qb = feats.join(pw.select(KEY + ["position", "player_display_name"]), on=KEY).filter(pl.col("position") == "QB")
    print(f"Wrote {OUT_PATH}: {feats.height:,} rows; QB rows with values: "
          f"{qb['qb_designed_runs_ewm'].is_not_null().mean():.0%}")
    print(qb.filter(pl.col("season") == 2024).group_by("player_display_name")
          .agg(pl.col(FEATURES[:3]).mean().round(2)).sort("qb_designed_runs_ewm", descending=True).head(8))


if __name__ == "__main__":
    main()
