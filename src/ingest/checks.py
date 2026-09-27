"""Sanity checks for raw weekly player stats. Each check returns human-readable issue strings;
an empty list means the data passed."""

import polars as pl

KEY = ["player_id", "game_id"]


def reg_season_weeks(season: int) -> int:
    """The 17-game schedule (18 weeks) started in 2021."""
    return 17 if season <= 2020 else 18


def standard_fantasy_points(df: pl.DataFrame, ppr: float = 0.0) -> pl.Series:
    """Fantasy points under nflverse's default scoring, used to verify their fantasy_points columns.

    4-pt pass TD, 0.04/pass yd, -2 INT, 0.1/rush+rec yd, 6-pt rush/rec/special-teams TD,
    2 per 2-pt conversion, -2 per fumble lost, plus `ppr` points per reception.
    """
    c = lambda name: pl.col(name).fill_null(0)
    expr = (
        c("passing_yards") * 0.04
        + c("passing_tds") * 4
        - c("passing_interceptions") * 2
        + c("rushing_yards") * 0.1
        + c("rushing_tds") * 6
        + c("receiving_yards") * 0.1
        + c("receiving_tds") * 6
        + c("special_teams_tds") * 6
        + (c("passing_2pt_conversions") + c("rushing_2pt_conversions") + c("receiving_2pt_conversions")) * 2
        - (c("sack_fumbles_lost") + c("rushing_fumbles_lost") + c("receiving_fumbles_lost")) * 2
        + c("receptions") * ppr
    )
    return df.select(expr.alias("pts")).to_series()


def check_player_stats(df: pl.DataFrame, season: int) -> list[str]:
    issues = []
    reg = df.filter(pl.col("season_type") == "REG")

    dupes = df.height - df.unique(subset=KEY).height
    if dupes:
        issues.append(f"{dupes} duplicate player-game rows")

    missing_opp = df.filter(pl.col("opponent_team").is_null()).height
    if missing_opp:
        issues.append(f"{missing_opp} rows missing opponent_team")

    if reg.height:
        expected = reg_season_weeks(season)
        max_week = reg["week"].max()
        if max_week > expected:
            issues.append(f"REG week {max_week} exceeds {expected}-week schedule")

        games = reg.group_by("team").agg(pl.col("game_id").n_unique().alias("games"))
        # A finished season has exactly expected-1 games per team (one bye); an in-progress one has fewer.
        too_many = games.filter(pl.col("games") > expected - 1)
        if too_many.height:
            issues.append(f"teams with more than {expected - 1} REG games: {too_many['team'].to_list()}")

    bad_rec = df.filter(pl.col("receptions") > pl.col("targets")).height
    if bad_rec:
        issues.append(f"{bad_rec} rows with receptions > targets")

    bad_cmp = df.filter(pl.col("completions") > pl.col("attempts")).height
    if bad_cmp:
        issues.append(f"{bad_cmp} rows with completions > attempts")

    for ppr, col in [(0.0, "fantasy_points"), (1.0, "fantasy_points_ppr")]:
        diff = (standard_fantasy_points(df, ppr) - df[col].fill_null(0)).abs()
        off = int((diff > 0.01).sum())
        if off:
            issues.append(f"{off} rows where {col} doesn't match the standard formula (max diff {diff.max():.2f})")

    return issues
