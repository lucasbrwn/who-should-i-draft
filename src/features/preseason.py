"""Preseason grade + offseason change flags (SCRUM-88).

Last season's averages miss what changed over the offseason (Lucas, 2026-11-10: trades, coaching
changes, new QBs; preseason games don't show how good a team will be). These columns cover that.
Everything here is known before kickoff.

    win_total           Vegas preseason win total (market consensus after all offseason moves),
                        from data/reference/win_totals.csv; blank until that file exists
    win_total_change    win_total minus the team's actual wins last season: did the market think
                        the team got better or worse this offseason?
    new_head_coach      the team's Week 1 coach differs from its coach in last season's final game
    new_starting_qb     the team's Week 1 starter differs from last season's most-used starter
    backup_qb_starting  this game's announced starter isn't the team's usual starter (most starts in
                        earlier games this season, or last season's in Week 1). Blank for games
                        with no starter listed yet.
    player_new_team     the player's team differs from his team in his last game of last season
    no_games_last_season  rookie, or missed all of last season

2020 has no previous season in our data, so its change flags are blank.

Usage:
    python -m src.features.preseason
"""

from pathlib import Path

import polars as pl

from src.features.team import read_schedules

PLAYER_WEEKS_PATH = Path("data/processed/player_weeks.parquet")
WIN_TOTALS_PATH = Path("data/reference/win_totals.csv")  # columns: season, team, win_total
OUT_PATH = Path("data/processed/features_preseason.parquet")

KEY = ["player_id", "season", "week"]


def team_games(schedules: pl.DataFrame) -> pl.DataFrame:
    """One row per team per game with its coach, starting QB, and whether it won."""
    cols = ["game_id", "season", "week"]
    home = schedules.select(cols, team=pl.col("home_team"), coach=pl.col("home_coach"), qb=pl.col("home_qb_id"),
                            won=pl.col("home_score") > pl.col("away_score"))
    away = schedules.select(cols, team=pl.col("away_team"), coach=pl.col("away_coach"), qb=pl.col("away_qb_id"),
                            won=pl.col("away_score") > pl.col("home_score"))
    return pl.concat([home, away]).sort("team", "season", "week")


def most_used_qb(games: pl.DataFrame) -> pl.Expr:
    return pl.col("qb").drop_nulls().mode().sort().first()


def team_season_changes(games: pl.DataFrame) -> pl.DataFrame:
    """Per team-season: coach and starter changes vs last season, and last season's wins."""
    by_season = games.group_by("team", "season").agg(
        week1_coach=pl.col("coach").sort_by("week").first(),
        final_coach=pl.col("coach").sort_by("week").last(),
        week1_qb=pl.col("qb").sort_by("week").first(),
        top_qb=most_used_qb(games),
        wins=pl.col("won").sum(),
        games_played=pl.col("won").is_not_null().sum(),
    )
    prev = by_season.select("team", (pl.col("season") + 1).alias("season"),
                            prev_coach=pl.col("final_coach"), prev_qb=pl.col("top_qb"), prev_wins=pl.col("wins"))
    out = by_season.join(prev, on=["team", "season"], how="left")
    return out.select(
        "team", "season", "prev_wins",
        new_head_coach=pl.when(pl.col("prev_coach").is_not_null()).then(pl.col("week1_coach") != pl.col("prev_coach")),
        new_starting_qb=pl.when(pl.col("prev_qb").is_not_null() & pl.col("week1_qb").is_not_null())
                          .then(pl.col("week1_qb") != pl.col("prev_qb")),
        prev_top_qb=pl.col("prev_qb"),
    )


def backup_qb_flags(games: pl.DataFrame, changes: pl.DataFrame) -> pl.DataFrame:
    """Per team-game: is this game's starter someone other than the usual starter so far?"""
    rows = []
    for (team, season), g in games.group_by("team", "season"):
        g = g.sort("week")
        prev_top = changes.filter((pl.col("team") == team) & (pl.col("season") == season))["prev_top_qb"]
        prev_top = prev_top[0] if prev_top.len() else None
        counts: dict[str, int] = {}
        for game_id, week, qb in g.select("game_id", "week", "qb").iter_rows():
            usual = max(counts, key=counts.get) if counts else prev_top
            flag = None if qb is None or usual is None else qb != usual
            rows.append({"game_id": game_id, "team": team, "backup_qb_starting": flag})
            if qb is not None:
                counts[qb] = counts.get(qb, 0) + 1
    return pl.DataFrame(rows, schema={"game_id": pl.String, "team": pl.String, "backup_qb_starting": pl.Boolean})


def player_changes(pw: pl.DataFrame) -> pl.DataFrame:
    last_team = pw.sort("week").group_by("player_id", "season").agg(last_team=pl.col("team").last()).with_columns(
        pl.col("season") + 1
    )
    seasons_with_data = pw["season"].unique()
    return pw.select(KEY + ["team"]).join(last_team, on=["player_id", "season"], how="left").select(
        KEY,
        player_new_team=pl.when(pl.col("last_team").is_not_null()).then(pl.col("team") != pl.col("last_team")),
        no_games_last_season=pl.when((pl.col("season") - 1).is_in(seasons_with_data.implode()))
                               .then(pl.col("last_team").is_null()),
    )


def read_win_totals() -> pl.DataFrame:
    schema = {"season": pl.Int32, "team": pl.String, "win_total": pl.Float64}
    if not WIN_TOTALS_PATH.exists():
        return pl.DataFrame(schema=schema)
    return pl.read_csv(WIN_TOTALS_PATH, schema_overrides=schema).select(list(schema))


def build_preseason_features(pw: pl.DataFrame, schedules: pl.DataFrame, win_totals: pl.DataFrame) -> pl.DataFrame:
    games = team_games(schedules)
    changes = team_season_changes(games)
    team_cols = (
        changes.join(win_totals.with_columns(pl.col("season").cast(changes.schema["season"])), on=["team", "season"], how="left")
        .with_columns(win_total_change=pl.col("win_total") - pl.col("prev_wins"))
        .select("team", "season", "win_total", "win_total_change", "new_head_coach", "new_starting_qb")
    )
    return (
        pw.select(KEY + ["game_id", "team"])
        .join(team_cols, on=["team", "season"], how="left")
        .join(backup_qb_flags(games, changes), on=["game_id", "team"], how="left")
        .join(player_changes(pw), on=KEY, how="left")
        .drop("game_id", "team")
        .sort(KEY)
    )


def check_preseason_features(feats: pl.DataFrame, pw: pl.DataFrame) -> list[str]:
    problems = []
    if feats.height != pw.height:
        problems.append(f"{feats.height} rows vs {pw.height} player_weeks rows")
    totals = feats["win_total"].drop_nulls()
    if totals.len() and ((totals < 2) | (totals > 15)).any():
        problems.append("win total outside 2-15")
    return problems


def main() -> None:
    pw = pl.read_parquet(PLAYER_WEEKS_PATH)
    feats = build_preseason_features(pw, read_schedules(), read_win_totals())
    problems = check_preseason_features(feats, pw)
    for p in problems:
        print("CHECK FAILED:", p)
    if problems:
        raise SystemExit(1)
    feats.write_parquet(OUT_PATH)
    print(f"Wrote {OUT_PATH}: {feats.height:,} rows x {feats.width} columns"
          + ("" if WIN_TOTALS_PATH.exists() else f" (no {WIN_TOTALS_PATH} yet: win_total columns blank)"))


if __name__ == "__main__":
    main()
