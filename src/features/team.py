"""Team context features (SCRUM-35): Vegas lines, pace, pass tendency, rest, and weather.

Two kinds of columns, joined onto every player_weeks row by (game_id, team):

Known before kickoff (taken from the game itself, which is fine because they exist before it starts):
    team_spread       points the team is favored by (negative = underdog), from the closing line
    implied_total     points Vegas expects the team to score = (total + team_spread) / 2
    opp_implied_total points Vegas expects the opponent to score
    game_total        the over/under
    rest_days, is_home, div_game, is_dome, temp, wind

Team tendencies, from the team's EARLIER games only (same rule as usage features):
    plays_l5 / plays_std / plays_prev   offensive plays per game (pace)
    proe_l5 / proe_std / proe_prev      pass rate over expected, in percentage points: how much more
                                        the team throws than the game situation predicts. Unlike raw
                                        pass rate, it isn't inflated by falling behind.
    points_l5 / points_std / points_prev

Usage:
    python -m src.features.team
"""

from pathlib import Path

import polars as pl

from src.ingest.pbp import read_pbp

PLAYER_WEEKS_PATH = Path("data/processed/player_weeks.parquet")
SCHEDULES_DIR = Path("data/raw/schedules")
OUT_PATH = Path("data/processed/features_team.parquet")

TEAM_KEY = ["team", "season", "week"]
GROUP = ["team", "season"]
TREND_STATS = ["plays", "proe", "points"]
DOME_ROOFS = ["dome", "closed"]
# Domes have no weather; fill with neutral conditions and keep an is_dome flag (decided 2026-11-10).
DOME_TEMP, DOME_WIND = 70.0, 0.0


def read_schedules() -> pl.DataFrame:
    files = sorted(p for p in SCHEDULES_DIR.glob("schedules_*.parquet"))
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed").filter(pl.col("game_type") == "REG")


def team_games(schedules: pl.DataFrame) -> pl.DataFrame:
    """One row per team per game (each game becomes a home row and an away row)."""
    common = ["game_id", "season", "week", "total_line", "div_game", "roof", "temp", "wind"]
    # nflverse spread_line: positive = home team favored.
    home = schedules.select(
        common, team=pl.col("home_team"), is_home=pl.lit(True), team_spread=pl.col("spread_line"),
        rest_days=pl.col("home_rest"), points=pl.col("home_score"),
    )
    away = schedules.select(
        common, team=pl.col("away_team"), is_home=pl.lit(False), team_spread=-pl.col("spread_line"),
        rest_days=pl.col("away_rest"), points=pl.col("away_score"),
    )
    games = pl.concat([home, away])
    is_dome = pl.col("roof").is_in(DOME_ROOFS)
    return games.with_columns(
        game_total=pl.col("total_line"),
        implied_total=(pl.col("total_line") + pl.col("team_spread")) / 2,
        opp_implied_total=(pl.col("total_line") - pl.col("team_spread")) / 2,
        div_game=pl.col("div_game").cast(pl.Boolean),
        is_dome=is_dome,
        temp=pl.when(is_dome).then(DOME_TEMP).otherwise(pl.col("temp").cast(pl.Float64)),
        wind=pl.when(is_dome).then(DOME_WIND).otherwise(pl.col("wind").cast(pl.Float64)),
    ).drop("total_line", "roof")


def team_plays(pbp: pl.DataFrame) -> pl.DataFrame:
    """Offensive plays and average pass rate over expected per team-game."""
    plays = pbp.filter(
        (pl.col("season_type") == "REG") & pl.col("posteam").is_not_null() & pl.col("play_type").is_in(["pass", "run"])
    )
    return plays.group_by("game_id", pl.col("posteam").alias("team")).agg(
        plays=pl.len(), proe=pl.col("pass_oe").mean(),
    )


def rolling_trends(games: pl.DataFrame) -> pl.DataFrame:
    """Last-5 and season-to-date averages that INCLUDE each game; attached later to the next week."""
    played = games.filter(pl.col("plays").is_not_null()).sort(TEAM_KEY)
    exprs = []
    for s in TREND_STATS:
        exprs.append(pl.col(s).rolling_mean(5, min_samples=1).over(GROUP).round(9).alias(f"{s}_l5"))
        exprs.append((pl.col(s).cum_sum().over(GROUP) / pl.col(s).cum_count().over(GROUP)).round(9).alias(f"{s}_std"))
    return played.select(TEAM_KEY + exprs)


def previous_season(games: pl.DataFrame) -> pl.DataFrame:
    played = games.filter(pl.col("plays").is_not_null())
    return played.group_by(GROUP).agg([pl.col(s).mean().alias(f"{s}_prev") for s in TREND_STATS]).with_columns(
        pl.col("season") + 1
    )


def build_team_features(schedules: pl.DataFrame, pbp_plays: pl.DataFrame) -> pl.DataFrame:
    """One row per team-game with pre-game context and earlier-games-only tendencies."""
    games = team_games(schedules).join(pbp_plays, on=["game_id", "team"], how="left")
    trends = games.select(TEAM_KEY + ["game_id"]).sort("week").join_asof(
        rolling_trends(games).sort("week"),
        on="week", by=GROUP, strategy="backward", allow_exact_matches=False, check_sortedness=False,
    )
    # Drop this game's own plays/points: they're outcomes, not features.
    out = games.drop(TREND_STATS).join(trends.drop("season", "week"), on=["game_id", "team"], how="left")
    return out.join(previous_season(games), on=GROUP, how="left").sort(TEAM_KEY)


def join_to_player_weeks(pw: pl.DataFrame, team_feats: pl.DataFrame) -> pl.DataFrame:
    keys = pw.select("player_id", "season", "week", "game_id", "team")
    return keys.join(team_feats.drop("season", "week"), on=["game_id", "team"], how="left")


def check_team_features(feats: pl.DataFrame, pw: pl.DataFrame) -> list[str]:
    problems = []
    if feats.height != pw.height:
        problems.append(f"{feats.height} rows vs {pw.height} player_weeks rows")
    missing = feats.filter(pl.col("is_home").is_null()).height
    if missing:
        problems.append(f"{missing} player-weeks didn't match a scheduled game")
    lines = feats.filter(pl.col("implied_total").is_not_null())
    if lines.filter((pl.col("implied_total") < 5) | (pl.col("implied_total") > 45)).height:
        problems.append("implied team total outside 5-45 points")
    if lines.filter((pl.col("implied_total") + pl.col("opp_implied_total") - pl.col("game_total")).abs() > 1e-6).height:
        problems.append("implied totals don't add up to the game total")
    return problems


def main() -> None:
    pw = pl.read_parquet(PLAYER_WEEKS_PATH)
    pbp = read_pbp(columns=["game_id", "season_type", "posteam", "play_type", "pass_oe"])
    team_feats = build_team_features(read_schedules(), team_plays(pbp))
    feats = join_to_player_weeks(pw, team_feats)
    problems = check_team_features(feats, pw)
    for p in problems:
        print("CHECK FAILED:", p)
    if problems:
        raise SystemExit(1)
    feats.write_parquet(OUT_PATH)
    print(f"Wrote {OUT_PATH}: {feats.height:,} rows x {feats.width} columns")


if __name__ == "__main__":
    main()
