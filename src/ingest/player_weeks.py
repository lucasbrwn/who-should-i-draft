"""Player-week master table: one row per offensive player per regular-season game played.

A player gets a row when he took an offensive snap OR recorded a stat. That keeps "played and
did nothing" games (Gabriel Davis 2023 wk 11: 88% of snaps, 0 catches), which nflverse's stats
table drops because it only has rows for players who recorded a stat (~12.5% of games played).
No row means he didn't play (bye, inactive, injured): those are never zeros.

Playoff games are left out (not every team plays them). The raw files still have them.

Usage:
    python -m src.ingest.player_weeks
"""

import argparse
from pathlib import Path

import polars as pl

from src.ingest.ff_opportunity import read_ff_opportunity
from src.ingest.in_game_injuries import read_in_game_injuries
from src.ingest.player_stats import default_seasons, read_player_stats

OUT_PATH = Path("data/processed/player_weeks.parquet")
SNAPS_DIR = Path("data/raw/snap_counts")
INJURIES_DIR = Path("data/raw/injuries")
SCHEDULES_DIR = Path("data/raw/schedules")

POSITIONS = ["QB", "RB", "WR", "TE"]
KEY = ["player_id", "game_id"]
ID_COLS = ["player_id", "player_display_name", "position", "season", "week", "game_id", "team", "opponent_team"]
SNAP_COLS = ["offense_snaps", "offense_pct", "st_snaps", "st_pct"]

# Ratios are undefined when the player had no plays, so a snap-only row leaves them null
# instead of 0 (0 would be a made-up value the model treats as real).
RATIO_STATS = ["passing_cpoe", "pacr", "racr"]
# Kicking, punting, and defensive columns: not used for offensive players.
DROP_PREFIXES = ("def_", "fg_", "pat_", "pt_", "gwfg_")
DROP_COLS = ["player_name", "position_group", "headshot_url", "season_type"]


def stat_columns(stats: pl.DataFrame) -> list[str]:
    """Numeric offensive stat columns from nflverse weekly stats."""
    return [
        c for c, dtype in stats.schema.items()
        if c not in ID_COLS and c not in DROP_COLS and not c.startswith(DROP_PREFIXES) and dtype.is_numeric()
    ]


def player_positions(stats: pl.DataFrame) -> pl.DataFrame:
    """Each player's most common position in the stats table, so one player = one position
    (e.g. Taysom Hill is a TE even in weeks he lines up at QB)."""
    return (
        stats.group_by("player_id", "position").len()
        .sort("len", "position", descending=[True, False])
        .unique("player_id", keep="first")
        .select("player_id", pl.col("position").alias("main_position"))
    )


INJURY_COLS = ["on_injury_report", "report_status", "practice_code", "injury"]
IN_GAME_COLS = ["injury_mentions", "injury_qtr", "usual_offense_pct", "left_injured"]

# left_injured = mentioned as injured in the play text AND played well under his usual snap share.
# "Usual" = median of his previous 5 games (prior games only, so no future data leaks in).
LEFT_EARLY_SHARE = 0.5      # under half his usual share
BASELINE_GAMES = 5
NO_BASELINE_PCT = 0.5       # first career game in the data: under 50% of snaps


def attach_injury_report(rows: pl.DataFrame, injuries: pl.DataFrame) -> pl.DataFrame:
    """That week's final injury report. Released before kickoff, so it's fair to use for that game."""
    report = injuries.filter(pl.col("game_type") == "REG").select(
        pl.col("gsis_id").alias("player_id"), pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32),
        "team", "report_status", "practice_code", pl.col("report_primary_injury").alias("injury"),
        pl.lit(True).alias("on_injury_report"),
    ).unique(["player_id", "season", "week", "team"])
    return rows.join(report, on=["player_id", "season", "week", "team"], how="left").with_columns(
        pl.col("on_injury_report").fill_null(False)
    )


def attach_in_game_injuries(rows: pl.DataFrame, in_game: pl.DataFrame) -> pl.DataFrame:
    rows = (
        rows.join(in_game, on=["player_id", "game_id"], how="left")
        .with_columns(pl.col("injury_mentions").fill_null(0))
        .sort("player_id", "season", "week")
        .with_columns(
            pl.col("offense_pct").shift(1)
            .rolling_median(window_size=BASELINE_GAMES, min_samples=1)
            .over("player_id").alias("usual_offense_pct")
        )
    )
    short_day = (
        pl.when(pl.col("usual_offense_pct").is_null())
        .then(pl.col("offense_pct") < NO_BASELINE_PCT)
        .otherwise(pl.col("offense_pct") < LEFT_EARLY_SHARE * pl.col("usual_offense_pct"))
    )
    return rows.with_columns(((pl.col("injury_mentions") > 0) & short_day).alias("left_injured"))


def expected_columns(opportunity: pl.DataFrame) -> list[str]:
    """The player's own expected stats (e.g. rec_touchdown_exp). Skips *_diff (= actual - expected,
    easy to recompute) and *_team (team totals belong with team-context features)."""
    return [c for c in opportunity.columns if c.endswith("_exp")]


def attach_expected_points(rows: pl.DataFrame, opportunity: pl.DataFrame) -> pl.DataFrame:
    cols = expected_columns(opportunity)
    exp = opportunity.select("player_id", "game_id", *cols).unique(["player_id", "game_id"])
    # no target/carry/pass that game -> nothing was expected, a real 0
    return rows.join(exp, on=["player_id", "game_id"], how="left").with_columns(
        [pl.col(c).fill_null(0.0) for c in cols]
    )


def build_player_weeks(stats: pl.DataFrame, snaps: pl.DataFrame, schedules: pl.DataFrame,
                       injuries: pl.DataFrame | None = None, in_game: pl.DataFrame | None = None,
                       opportunity: pl.DataFrame | None = None) -> pl.DataFrame:
    """injuries / in_game / opportunity are optional so the base table can be tested on its own."""
    stat_cols = stat_columns(stats)
    reg_stats = stats.filter(pl.col("season_type") == "REG")
    positions = player_positions(stats)

    played = (
        snaps.filter((pl.col("game_type") == "REG") & (pl.col("offense_snaps") > 0) & pl.col("gsis_id").is_not_null())
        .select(
            pl.col("gsis_id").alias("player_id"), "game_id",
            pl.col("player").alias("snap_name"), pl.col("position").alias("snap_position"),
            pl.col("team").alias("snap_team"), pl.col("opponent").alias("snap_opponent"),
            pl.col("season").cast(pl.Int32).alias("snap_season"), pl.col("week").cast(pl.Int32).alias("snap_week"),
            *SNAP_COLS,
        )
        .unique(KEY)
    )

    rows = (
        reg_stats.select(*ID_COLS, *stat_cols)
        .join(played, on=KEY, how="full", coalesce=True)
        .join(positions, on="player_id", how="left")
        .with_columns(
            pl.coalesce("player_display_name", "snap_name").alias("player_display_name"),
            pl.coalesce("main_position", "position", "snap_position").alias("position"),
            pl.coalesce("team", "snap_team").alias("team"),
            pl.coalesce("opponent_team", "snap_opponent").alias("opponent_team"),
            pl.coalesce("season", "snap_season").alias("season"),
            pl.coalesce("week", "snap_week").alias("week"),
            # stats only, no snap row (e.g. a trick-play pass by a player listed elsewhere): 0 snaps
            pl.col("offense_snaps").fill_null(0),
            pl.col("st_snaps").fill_null(0),
            pl.col("offense_pct").fill_null(0.0),
            pl.col("st_pct").fill_null(0.0),
            pl.col("player_display_name").is_null().alias("snaps_only"),
        )
        .with_columns(
            [pl.col(c).fill_null(0) for c in stat_cols if c not in RATIO_STATS]
        )
        .filter(pl.col("position").is_in(POSITIONS))
    )

    games = schedules.filter(pl.col("game_type") == "REG").select("game_id", "gameday", "home_team")
    rows = rows.join(games, on="game_id", how="left").with_columns(
        (pl.col("team") == pl.col("home_team")).alias("is_home")
    )
    extra = []
    if injuries is not None:
        rows = attach_injury_report(rows, injuries)
        extra += INJURY_COLS
    if in_game is not None:
        rows = attach_in_game_injuries(rows, in_game)
        extra += IN_GAME_COLS
    if opportunity is not None:
        rows = attach_expected_points(rows, opportunity)
        extra += expected_columns(opportunity)
    return (
        rows.select(*ID_COLS, "gameday", "is_home", "snaps_only", *SNAP_COLS, *extra, *stat_cols)
        .sort("season", "week", "team", "player_id")
    )


def check_player_weeks(df: pl.DataFrame) -> list[str]:
    issues = []
    dupes = df.height - df.unique(KEY).height
    if dupes:
        issues.append(f"{dupes} duplicate player-games")
    for col in ["opponent_team", "team", "position", "gameday", "is_home"]:
        missing = df[col].null_count()
        if missing:
            issues.append(f"{missing} rows missing {col}")
    games_per_team = df.group_by("season", "team").agg(pl.col("game_id").n_unique()).filter(pl.col("game_id") > 17)
    if games_per_team.height:
        issues.append(f"{games_per_team.height} team-seasons with more than 17 games")
    return issues


def read_dir(directory: Path, prefix: str, seasons: list[int]) -> pl.DataFrame:
    return pl.concat(
        [pl.read_parquet(directory / f"{prefix}_{s}.parquet") for s in seasons], how="diagonal_relaxed"
    )


def build(seasons: list[int]) -> pl.DataFrame:
    df = build_player_weeks(
        read_player_stats(seasons), read_dir(SNAPS_DIR, "snap_counts", seasons),
        read_dir(SCHEDULES_DIR, "schedules", seasons),
        injuries=read_dir(INJURIES_DIR, "injuries", seasons),
        in_game=read_in_game_injuries(seasons),
        opportunity=read_ff_opportunity(seasons),
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PATH)
    print(f"{df.height:,} player-weeks ({df['snaps_only'].sum():,} snap-only, "
          f"{df['on_injury_report'].sum():,} on injury report, {df['left_injured'].sum():,} left injured) -> {OUT_PATH}")
    for issue in check_player_weeks(df):
        print(f"  ! {issue}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=None)
    args = parser.parse_args()
    build(args.seasons or default_seasons())


if __name__ == "__main__":
    main()
