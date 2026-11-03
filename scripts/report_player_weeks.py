"""Data-quality report for the player-week table -> docs/reports/player_weeks_report.md

Usage (from the project folder, after `python -m src.ingest`):
    python scripts/report_player_weeks.py
"""

import sys
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ingest.player_weeks import OUT_PATH, check_player_weeks  # noqa: E402
from src.scoring.config import PRESETS  # noqa: E402
from src.scoring.engine import add_fantasy_points  # noqa: E402

REPORT = Path("docs/reports/player_weeks_report.md")


PLAIN = {"season", "week", "last week"}   # numbers that are labels, not counts


def cell(column: str, value) -> str:
    if value is None:
        return ""
    if column == "snap %":
        return f"{value:.0%}"
    if isinstance(value, int) and column not in PLAIN:
        return f"{value:,}"
    return str(value)


def table(df: pl.DataFrame) -> str:
    """Markdown table from a small DataFrame."""
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(cell(c, v) for c, v in zip(df.columns, row)) + " |" for row in df.iter_rows()]
    return "\n".join([head, sep, *body])


def next_week_missed(df: pl.DataFrame) -> pl.DataFrame:
    """For each row: did the player miss his team's next game (no row) or get listed Out/Doubtful?"""
    seasons = sorted(df["season"].unique())
    sched = pl.concat([pl.read_parquet(f"data/raw/schedules/schedules_{s}.parquet") for s in seasons],
                      how="diagonal_relaxed").filter((pl.col("game_type") == "REG") & pl.col("result").is_not_null())
    team_games = pl.concat([
        sched.select("season", "week", pl.col("home_team").alias("team")),
        sched.select("season", "week", pl.col("away_team").alias("team")),
    ]).sort("team", "season", "week").with_columns(pl.col("week").shift(-1).over("team", "season").alias("next_week"))
    injuries = pl.concat([pl.read_parquet(f"data/raw/injuries/injuries_{s}.parquet") for s in seasons],
                         how="diagonal_relaxed")
    out = injuries.filter((pl.col("game_type") == "REG") & pl.col("report_status").is_in(["Out", "Doubtful"])).select(
        pl.col("gsis_id").alias("player_id"), pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32).alias("next_week"), pl.lit(True).alias("out_next"),
    ).unique()
    played = df.select("player_id", "season", pl.col("week").alias("next_week"), pl.lit(True).alias("played_next"))
    return (
        df.join(team_games, on=["season", "week", "team"], how="left")
        .filter(pl.col("next_week").is_not_null())
        .join(played, on=["player_id", "season", "next_week"], how="left")
        .join(out, on=["player_id", "season", "next_week"], how="left")
        .with_columns((pl.col("played_next").is_null() | pl.col("out_next").fill_null(False)).alias("missed_next"))
    )


def main() -> None:
    df = add_fantasy_points(pl.read_parquet(OUT_PATH), PRESETS["standard-ppr"].scoring, "ppr")
    issues = check_player_weeks(df)
    lines = [
        "# Player-week table: data report",
        "",
        f"Generated {date.today()} by `python scripts/report_player_weeks.py` from `{OUT_PATH.as_posix()}`.",
        "",
        f"**{df.height:,} player-weeks**, {df.width - 1} columns, seasons {df['season'].min()}–{df['season'].max()}. "
        f"Checks: {'all passed' if not issues else '; '.join(issues)}.",
        "",
        "## Rows by season and position",
        "",
    ]
    by = df.group_by("season").agg(
        *[(pl.col("position") == p).sum().alias(p) for p in ["QB", "RB", "WR", "TE"]],
        pl.len().alias("total"), pl.col("week").max().alias("last week"),
        (pl.col("snaps_only").mean() * 100).round(1).alias("snap-only %"),
    ).sort("season")
    lines += [table(by), "",
              "*Snap-only* = played on offense but recorded no stat, so nflverse has no row for it. "
              "These are real zero games (busts), and they'd be missing without this table.", ""]

    busts = (
        df.filter(pl.col("position").is_in(["WR", "TE"]) & (pl.col("offense_pct") >= 0.75))
        .sort("ppr", "offense_pct", descending=[False, True]).head(8)
        .select("season", "week", pl.col("player_display_name").alias("player"), "position", "team",
                pl.col("offense_pct").alias("snap %"), "targets", pl.col("ppr").round(1).alias("PPR"))
    )
    lines += ["## Biggest bust games (WR/TE on 75%+ of snaps)", "", table(busts), ""]

    report = df.group_by("report_status").agg(pl.len().alias("player-weeks")).sort("player-weeks", descending=True)
    lines += ["## Injury report status of players who played", "",
              "Players ruled Out mostly don't appear (they didn't play). Blank = not on the game-status report.", "",
              table(report.with_columns(pl.col("report_status").fill_null("(none)"))), ""]

    missed = next_week_missed(df)
    drop = pl.col("offense_pct") < 0.5 * pl.col("usual_offense_pct")
    group = (
        pl.when(pl.col("left_injured")).then(pl.lit("flagged left_injured"))
        .when(pl.col("injury_mentions") > 0).then(pl.lit("injury in play text, normal snaps"))
        .when(drop & (pl.col("usual_offense_pct") >= 0.5)).then(pl.lit("starter, snaps < half usual, no injury text"))
        .otherwise(pl.lit("everyone else"))
    )
    validation = missed.group_by(group.alias("group")).agg(
        pl.len().alias("games"), (pl.col("missed_next").mean() * 100).round(1).alias("missed next game %"),
    ).sort("missed next game %", descending=True)
    lines += ["## Does `left_injured` find real injuries?", "",
              "If the flag is right, flagged players should miss their next game far more often than everyone else.", "",
              table(validation), "",
              "Groups 2 and 3 are why the flag needs *both* signals: injury text alone includes players who "
              "returned to the game, and a snap drop alone is mostly benchings and QB changes.", ""]

    per_season = df.group_by("season").agg(pl.col("left_injured").sum().alias("left injured"),
                                           (pl.col("injury_mentions") > 0).sum().alias("injury in play text")).sort("season")
    lines += [table(per_season), ""]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {REPORT}")


if __name__ == "__main__":
    main()
