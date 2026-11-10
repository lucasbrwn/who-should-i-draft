"""Vacated-opportunity features (SCRUM-38): work left behind by teammates who aren't playing.

For each team-week, a teammate counts as "out" if he played for this team in one of its last 3
games but isn't playing this week (injury, suspension, cut, or trade). In live use that's "Out on the injury
report or inactive", which is known before kickoff (inactives come 90 minutes before), so no leakage.

What's vacated = each missing teammate's season-to-date target share and carry share through his
last game, summed by his position. The model then learns, across all teams, who absorbs it
(per-team history is too thin: a team may have 2-3 WR1-out games in 5 years).

    vacated_tgt_share_wr / _te / _rb / _total   targets left by missing WRs / TEs / RBs / all
    vacated_carry_share                         carries left by missing players
    teammates_out                               how many earlier teammates aren't playing
    on_injury_report, report_status, practice_code  the player's OWN status (from SCRUM-22)

Week 1 has nothing vacated yet (no earlier games); offseason departures are SCRUM-88.

Usage:
    python -m src.features.vacated
"""

from pathlib import Path

import polars as pl

from src.features.usage import add_game_columns

PLAYER_WEEKS_PATH = Path("data/processed/player_weeks.parquet")
OUT_PATH = Path("data/processed/features_vacated.parquet")

KEY = ["player_id", "season", "week"]
TEAM_WEEK = ["team", "season", "week"]
OWN_STATUS = ["on_injury_report", "report_status", "practice_code"]
TGT_POSITIONS = ["WR", "TE", "RB"]
RECENT_GAMES = 3  # a teammate counts as "out" only if he played in one of the team's last 3 games


def shares_through_each_game(pw: pl.DataFrame) -> pl.DataFrame:
    """Each player's season-to-date target/carry share INCLUDING that game (left-injured games skipped)."""
    usable = add_game_columns(pw).filter(~pl.col("left_injured")).sort(KEY)
    n = pl.col("week").cum_count().over("player_id", "season")
    return usable.select(
        KEY,
        tgt_share=(pl.col("target_share").cum_sum().over("player_id", "season") / n).round(9),
        carry_share=(pl.col("carry_share").cum_sum().over("player_id", "season") / n).round(9),
    )


def missing_teammates(pw: pl.DataFrame) -> pl.DataFrame:
    """(team, season, week, player_id) for each teammate who played in one of the team's last
    RECENT_GAMES games but isn't playing for the team this week.

    Only recent teammates count: a backup cut in Week 2 or a season-ending injury from a month ago
    isn't "vacated" anymore, because the remaining players' recent averages already show the new roles.
    """
    team_weeks = pw.select(TEAM_WEEK).unique().sort(TEAM_WEEK).with_columns(
        game_no=pl.col("week").rank("ordinal").over("team", "season").cast(pl.Int32)  # signed: game_no - 3 can go negative
    )
    played = pw.select(TEAM_WEEK + ["player_id"]).join(team_weeks, on=TEAM_WEEK)
    # Teammate appearances in each of the team's previous RECENT_GAMES games.
    recent = (
        team_weeks.join(played.select("team", "season", "player_id", pl.col("game_no").alias("played_no")),
                        on=["team", "season"])
        .filter((pl.col("played_no") < pl.col("game_no")) & (pl.col("played_no") >= pl.col("game_no") - RECENT_GAMES))
        .select(TEAM_WEEK + ["player_id"]).unique()
    )
    return recent.join(played.select(TEAM_WEEK + ["player_id"]), on=TEAM_WEEK + ["player_id"], how="anti")


def vacated_by_team_week(pw: pl.DataFrame) -> pl.DataFrame:
    positions = pw.group_by("player_id").agg(pl.col("position").mode().first().alias("out_position"))
    missing = missing_teammates(pw).join(positions, on="player_id")
    # Each missing player's share as of his last game before this week.
    missing = missing.sort("week").join_asof(
        shares_through_each_game(pw).sort("week"),
        on="week", by=["player_id", "season"], strategy="backward", allow_exact_matches=False, check_sortedness=False,
    ).with_columns(pl.col("tgt_share", "carry_share").fill_null(0))
    tgt = pl.col("tgt_share")
    return missing.group_by(TEAM_WEEK).agg(
        [tgt.filter(pl.col("out_position") == p).sum().alias(f"vacated_tgt_share_{p.lower()}") for p in TGT_POSITIONS]
        + [tgt.sum().alias("vacated_tgt_share_total"), pl.col("carry_share").sum().alias("vacated_carry_share"),
           pl.len().alias("teammates_out")]
    )


def build_vacated_features(pw: pl.DataFrame) -> pl.DataFrame:
    vac = vacated_by_team_week(pw)
    fill = [c for c in vac.columns if c not in TEAM_WEEK]
    return (
        pw.select(KEY + ["team"] + OWN_STATUS)
        .join(vac, on=TEAM_WEEK, how="left")
        .with_columns(pl.col(fill).fill_null(0))
        .drop("team")
        .sort(KEY)
    )


def check_vacated_features(feats: pl.DataFrame, pw: pl.DataFrame) -> list[str]:
    problems = []
    if feats.height != pw.height:
        problems.append(f"{feats.height} rows vs {pw.height} player_weeks rows")
    if feats.filter(pl.col("vacated_tgt_share_total") < 0).height:
        problems.append("negative vacated share")
    week1 = feats.filter(pl.col("week") == 1)
    if week1.filter(pl.col("teammates_out") > 0).height:
        problems.append("week 1 has vacated opportunity (there are no earlier games)")
    return problems


def main() -> None:
    pw = pl.read_parquet(PLAYER_WEEKS_PATH)
    feats = build_vacated_features(pw)
    problems = check_vacated_features(feats, pw)
    for p in problems:
        print("CHECK FAILED:", p)
    if problems:
        raise SystemExit(1)
    feats.write_parquet(OUT_PATH)
    print(f"Wrote {OUT_PATH}: {feats.height:,} rows x {feats.width} columns")


if __name__ == "__main__":
    main()
