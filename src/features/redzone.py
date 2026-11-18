"""Red-zone opportunity and Vegas-linked TD expectation (SCRUM-91 + weak-points finding, 2026-11-17).

Why: TDs are 40-58% of our points error, and we under-project TDs on teams Vegas expects to score a
lot (2024: RB 0.26 projected vs 0.30 actual, WR 0.20 vs 0.22 on high-implied-total teams). The model
knows the team total but not WHO gets the scoring chances. These features connect the two.

Per game, from play-by-play (regular season, runs and passes, no 2-point tries):
    red zone   snap at the opponent's 20 or closer;  goal line  at the 5 or closer
    carries    plays where he's the rusher (includes QB runs and scrambles)
    targets    plays where he's the intended receiver

Shares = his opportunities / his team's, averaged over games he played:
    *_share_ewm   decayed (half-life 3 games, this season + last, like decay.py)
    *_share_std   this season so far
Vegas link (pre-game): implied team TDs = implied total / 7, times his decayed share:
    td_exp_rz_rush, td_exp_rz_rec, td_exp_gl_rush

Every value uses only games BEFORE the row's week.

RESULT 2026-11-17: A/B (2021-24 walk-forward) tied at every position (MAE diffs within +-0.004,
Spearman within +-0.001, all CIs span 0) -> NOT used (removed from assemble.build_all). The model's
existing usage + Vegas features already carry this; the remaining TD error looks like randomness.

    python -m src.features.redzone
"""

from pathlib import Path

import polars as pl

from src.features.team import read_schedules, team_games

PLAYER_WEEKS_PATH = Path("data/processed/player_weeks.parquet")
OUT_PATH = Path("data/processed/features_redzone.parquet")
PBP_COLS = ["game_id", "season_type", "posteam", "play_type", "yardline_100", "rusher_player_id",
            "receiver_player_id", "two_point_attempt"]
KEY = ["player_id", "season", "week"]
HALF_LIFE = 3
COUNTS = ["rz_carry", "rz_target", "gl_carry", "gl_target"]
SHARES = [f"{c}_share" for c in COUNTS]
FEATURES = ([f"{s}_ewm" for s in SHARES] + ["rz_carry_share_std", "rz_target_share_std"]
            + ["td_exp_rz_rush", "td_exp_rz_rec", "td_exp_gl_rush"])


def game_counts(pbp: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(player counts per game, team counts per game)."""
    plays = pbp.filter((pl.col("season_type") == "REG") & pl.col("play_type").is_in(["run", "pass"])
                       & (pl.col("two_point_attempt").fill_null(0) != 1) & (pl.col("yardline_100") <= 20)
                       ).with_columns(gl=pl.col("yardline_100") <= 5)
    rush = plays.filter(pl.col("rusher_player_id").is_not_null()).select(
        "game_id", team="posteam", player_id="rusher_player_id", rz_carry=pl.lit(1), gl_carry=pl.col("gl").cast(pl.Int32),
        rz_target=pl.lit(0), gl_target=pl.lit(0))
    rec = plays.filter(pl.col("receiver_player_id").is_not_null()).select(
        "game_id", team="posteam", player_id="receiver_player_id", rz_carry=pl.lit(0), gl_carry=pl.lit(0),
        rz_target=pl.lit(1), gl_target=pl.col("gl").cast(pl.Int32))
    both = pl.concat([rush, rec], how="vertical_relaxed")
    player = both.group_by("game_id", "player_id").agg([pl.col(c).sum() for c in COUNTS])
    team = both.group_by("game_id", "team").agg([pl.col(c).sum().alias(f"team_{c}") for c in COUNTS])
    return player, team


def per_game(pw: pl.DataFrame, pbp: pl.DataFrame) -> pl.DataFrame:
    """Each player-game he played (not left injured), with his and his team's red-zone counts."""
    player, team = game_counts(pbp)
    played = pw.filter(~pl.col("left_injured") & pl.col("fantasy_points_ppr").is_not_null()).select(
        "player_id", "season", "week", "game_id", "team")
    return (played.join(player, on=["game_id", "player_id"], how="left")
            .join(team, on=["game_id", "team"], how="left")
            .with_columns([pl.col(c).fill_null(0) for c in COUNTS + [f"team_{c}" for c in COUNTS]])
            .with_columns(t=pl.col("season") * 100 + pl.col("week")))


def through_each_game(games: pl.DataFrame) -> pl.DataFrame:
    """Shares INCLUDING each game: decayed (across the window given) and this-season cumulative."""
    games = games.sort("player_id", "season", "week")
    ewm = lambda c: pl.col(c).ewm_mean(half_life=HALF_LIFE, adjust=True).over("player_id")  # noqa: E731
    cum = lambda c: pl.col(c).cum_sum().over("player_id", "season")  # noqa: E731
    share = lambda num, den: pl.when(den > 0).then(num / den)  # noqa: E731
    return games.select(
        "player_id", "season", "t",
        *[share(ewm(c), ewm(f"team_{c}")).alias(f"{c}_share_ewm") for c in COUNTS],
        share(cum("rz_carry"), cum("team_rz_carry")).alias("rz_carry_share_std_all"),
        share(cum("rz_target"), cum("team_rz_target")).alias("rz_target_share_std_all"),
    )


def build_redzone_features(pw: pl.DataFrame, pbp: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    games = per_game(pw, pbp)
    rows_all = pw.select("player_id", "season", "week", "game_id", "team").with_columns(
        t=pl.col("season") * 100 + pl.col("week"))
    out = []
    for season in sorted(rows_all["season"].unique().to_list()):
        rows = rows_all.filter(pl.col("season") == season).sort("t")
        hist = through_each_game(games.filter(pl.col("season").is_in([season - 1, season]))).sort("t")
        j = rows.join_asof(hist.drop("season"), on="t", by="player_id", strategy="backward",
                           allow_exact_matches=False, check_sortedness=False)
        # season-to-date only counts if the latest earlier game was THIS season
        last_season = rows.join_asof(hist.select("player_id", "t", last_season="season"), on="t", by="player_id",
                                     strategy="backward", allow_exact_matches=False, check_sortedness=False)
        j = j.join(last_season.select("player_id", "t", "last_season"), on=["player_id", "t"], how="left")
        this = pl.col("last_season") == season
        out.append(j.with_columns(
            rz_carry_share_std=pl.when(this).then(pl.col("rz_carry_share_std_all")),
            rz_target_share_std=pl.when(this).then(pl.col("rz_target_share_std_all")),
        ))
    feats = pl.concat(out, how="diagonal_relaxed")
    vegas = team_games(schedules).select("game_id", "team", implied_tds=pl.col("implied_total") / 7)
    feats = feats.join(vegas, on=["game_id", "team"], how="left").with_columns(
        td_exp_rz_rush=pl.col("implied_tds") * pl.col("rz_carry_share_ewm"),
        td_exp_rz_rec=pl.col("implied_tds") * pl.col("rz_target_share_ewm"),
        td_exp_gl_rush=pl.col("implied_tds") * pl.col("gl_carry_share_ewm"),
    )
    return feats.select(*KEY, *[pl.col(f).round(9) for f in FEATURES]).sort(KEY)


def main() -> None:
    from src.ingest.pbp import read_pbp

    pw = pl.read_parquet(PLAYER_WEEKS_PATH)
    seasons = sorted(pw["season"].unique().to_list())
    feats = build_redzone_features(pw, read_pbp(seasons, columns=PBP_COLS), read_schedules())
    if feats.height != pw.height or feats.group_by(KEY).len().filter(pl.col("len") > 1).height:
        raise SystemExit("CHECK FAILED: redzone rows don't match player_weeks one-to-one")
    feats.write_parquet(OUT_PATH)
    filled = feats.group_by("season").agg(pl.col("rz_carry_share_ewm").is_not_null().mean().round(2)).sort("season")
    print(f"Wrote {OUT_PATH}: {feats.height:,} rows; share with a red-zone share: {dict(filled.rows())}")


if __name__ == "__main__":
    main()
