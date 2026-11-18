"""QB red-zone passing (2026-11-18, Lucas: "QB points might improve from red zone data").

Why: passing TDs are most of a QB's TD points, and TDs are 58% of our QB points error. Red-zone
RUNNING was already tested twice (redzone.py carry shares, qb_rush.py red-zone/goal-line runs)
and tied. Red-zone PASSING never was: the model knows the team's Vegas total and the QB's overall
volume, but not how often his offense gets inside the 20 or whether it throws once there.

Per game, from play-by-play (regular season, no 2-point tries, no spikes):
    rz pass        play_type 'pass' (sacks included: they are pass calls) at the opponent's 20 or
                   closer, he is the passer
    gl10 pass      the same at the 10 or closer
    team rz play   any run or pass by his team at the 20 or closer
    team rz trip   a drive by his team with at least one snap at the 20 or closer

Features (decayed, half-life 3 games, this season + last like decay.py; only games BEFORE the row):
    qb_rz_passes_ewm        red-zone pass attempts per game
    qb_rz_pass_share_ewm    his red-zone passes / his team's red-zone plays (decayed sums)
    qb_team_rz_trips_ewm    his team's red-zone trips per game (games he played)
    qb_gl10_passes_ewm      passes at the 10 or closer per game
    qb_rz_pass_td_exp       this game's Vegas implied team TDs (implied total / 7) x qb_rz_pass_share_ewm
QBs only: other positions get nulls.

Rule locked 2026-11-18 BEFORE testing (walk-forward 2021-24, boosting vs boosting_no_qbredzone):
KEEP if the QB MAE diff CI is entirely < 0 or the QB Spearman diff CI is entirely > 0, AND no
position is significantly worse. Otherwise (a tie, or better but not significant) DROP.

RESULT 2026-11-18: tie -> NOT used (removed from assemble.build_all). WITH these columns QB MAE was
0.013 lower [-0.006, +0.030] and Spearman 0.003 higher [-0.001, +0.008]: the best lean of the TD
feature tests, but neither CI clears 0. RB/WR/TE identical. Candidate to re-test with 2025+ data
after the final exam (more QB-weeks would tighten the CI).

    python -m src.features.qb_redzone
"""

from pathlib import Path

import polars as pl

from src.features.team import read_schedules, team_games

PLAYER_WEEKS_PATH = Path("data/processed/player_weeks.parquet")
OUT_PATH = Path("data/processed/features_qb_redzone.parquet")
PBP_COLS = ["game_id", "season_type", "posteam", "play_type", "yardline_100", "passer_player_id",
            "fixed_drive", "qb_spike", "two_point_attempt"]
KEY = ["player_id", "season", "week"]
HALF_LIFE = 3
COUNTS = ["rz_passes", "gl10_passes", "team_rz_plays", "team_rz_trips"]
FEATURES = ["qb_rz_passes_ewm", "qb_rz_pass_share_ewm", "qb_team_rz_trips_ewm", "qb_gl10_passes_ewm",
            "qb_rz_pass_td_exp"]


def game_counts(pbp: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(passer counts per game, team counts per game) inside the opponent's 20."""
    rz = pbp.filter((pl.col("season_type") == "REG") & pl.col("play_type").is_in(["run", "pass"])
                    & (pl.col("two_point_attempt").fill_null(0) != 1) & (pl.col("qb_spike").fill_null(0) != 1)
                    & (pl.col("yardline_100") <= 20))
    passer = (rz.filter((pl.col("play_type") == "pass") & pl.col("passer_player_id").is_not_null())
              .group_by("game_id", player_id="passer_player_id")
              .agg(rz_passes=pl.len().cast(pl.Float64),
                   gl10_passes=(pl.col("yardline_100") <= 10).sum().cast(pl.Float64)))
    team = (rz.group_by("game_id", team="posteam")
            .agg(team_rz_plays=pl.len().cast(pl.Float64),
                 team_rz_trips=pl.col("fixed_drive").n_unique().cast(pl.Float64)))
    return passer, team


def through_each_game(games: pl.DataFrame) -> pl.DataFrame:
    """Decayed values INCLUDING each game."""
    games = games.sort("player_id", "season", "week")
    ewm = lambda c: pl.col(c).ewm_mean(half_life=HALF_LIFE, adjust=True).over("player_id")  # noqa: E731
    return games.select(
        "player_id", "t",
        qb_rz_passes_ewm=ewm("rz_passes"),
        qb_rz_pass_share_ewm=pl.when(ewm("team_rz_plays") > 0).then(ewm("rz_passes") / ewm("team_rz_plays")),
        qb_team_rz_trips_ewm=ewm("team_rz_trips"),
        qb_gl10_passes_ewm=ewm("gl10_passes"),
    )


def build_qb_redzone_features(pw: pl.DataFrame, pbp: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    passer, team = game_counts(pbp)
    pw = pw.with_columns(t=pl.col("season") * 100 + pl.col("week"))
    qb_games = (pw.filter((pl.col("position") == "QB") & ~pl.col("left_injured")
                          & pl.col("fantasy_points_ppr").is_not_null())
                .select("player_id", "season", "week", "game_id", "team", "t")
                .join(passer, on=["game_id", "player_id"], how="left")
                .join(team, on=["game_id", "team"], how="left")
                .with_columns([pl.col(c).fill_null(0.0) for c in COUNTS]))
    out = []
    for season in sorted(pw["season"].unique().to_list()):
        rows = pw.filter(pl.col("season") == season).select(KEY + ["position", "game_id", "team", "t"]).sort("t")
        hist = through_each_game(qb_games.filter(pl.col("season").is_in([season - 1, season]))).sort("t")
        out.append(rows.join_asof(hist, on="t", by="player_id", strategy="backward",
                                  allow_exact_matches=False, check_sortedness=False))
    feats = pl.concat(out, how="diagonal_relaxed")
    vegas = team_games(schedules).select("game_id", "team", implied_tds=pl.col("implied_total") / 7)
    feats = feats.join(vegas, on=["game_id", "team"], how="left").with_columns(
        qb_rz_pass_td_exp=pl.col("implied_tds") * pl.col("qb_rz_pass_share_ewm"))
    is_qb = pl.col("position") == "QB"
    return feats.select(*KEY, *[pl.when(is_qb).then(pl.col(f).round(9)).alias(f) for f in FEATURES]).sort(KEY)


def main() -> None:
    from src.ingest.pbp import read_pbp

    pw = pl.read_parquet(PLAYER_WEEKS_PATH)
    seasons = sorted(pw["season"].unique().to_list())
    feats = build_qb_redzone_features(pw, read_pbp(seasons, columns=PBP_COLS), read_schedules())
    if feats.height != pw.height or feats.group_by(KEY).len().filter(pl.col("len") > 1).height:
        raise SystemExit("CHECK FAILED: qb_redzone rows don't match player_weeks one-to-one")
    feats.write_parquet(OUT_PATH)
    qb = feats.join(pw.select(KEY + ["position", "player_display_name"]), on=KEY).filter(pl.col("position") == "QB")
    print(f"Wrote {OUT_PATH}: {feats.height:,} rows; QB rows with values: "
          f"{qb['qb_rz_passes_ewm'].is_not_null().mean():.0%}")
    print(qb.filter(pl.col("season") == 2024).group_by("player_display_name")
          .agg(pl.col(FEATURES[:3]).mean().round(2), n=pl.len()).filter(pl.col("n") >= 10)
          .sort("qb_rz_passes_ewm", descending=True).head(8))


if __name__ == "__main__":
    main()
