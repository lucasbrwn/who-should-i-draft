"""Validate the scoring engine against outside numbers (SCRUM-33).

1. nflverse: every player-week's `fantasy_points` (standard) and `fantasy_points_ppr` columns.
2. Sleeper: a sample of weeks from Sleeper's public stats API, which reports each player's
   points under its own default PPR / half-PPR / standard scoring.

Usage:
    python -m src.scoring.validate                      # nflverse on all seasons + Sleeper sample
    python -m src.scoring.validate --weeks 2024-5 2023-12
"""

import argparse
import json
import urllib.request
from dataclasses import replace

import nflreadpy
import polars as pl

from src.ingest.player_stats import default_seasons, read_player_stats
from src.scoring.config import PRESETS, Scoring
from src.scoring.engine import FUMBLE_LOST_PARTS, add_fantasy_points

OFFENSE = ["QB", "RB", "WR", "TE"]
TOLERANCE = 0.01
SLEEPER_URL = "https://api.sleeper.app/v1/stats/nfl/regular/{season}/{week}"
# Skips 2022 weeks 1-3, where Sleeper's stored points are inconsistent (docs/research/scoring_validation.md).
DEFAULT_WEEKS = ["2022-4", "2022-10", "2023-1", "2023-12", "2024-5", "2024-17", "2025-9"]

# Sleeper's default scoring differs from nflverse and our presets: -1 per interception (not -2).
# (Recovered fumbles are NOT penalized: 25 sampled player-weeks with fumbles match without it.)
SLEEPER_DIFFS = {"passing_interceptions": -1.0}
SLEEPER_SCORING = {
    "pts_ppr": replace(PRESETS["standard-ppr"].scoring, **SLEEPER_DIFFS),
    "pts_half_ppr": replace(PRESETS["half-ppr"].scoring, **SLEEPER_DIFFS),
    "pts_std": replace(PRESETS["standard"].scoring, **SLEEPER_DIFFS),
}


def mismatches(df: pl.DataFrame, scoring: Scoring, reference: str) -> pl.DataFrame:
    """Rows where our points differ from the `reference` column by more than TOLERANCE."""
    return (
        add_fantasy_points(df, scoring, "ours")
        .with_columns((pl.col("ours") - pl.col(reference).fill_null(0)).alias("diff"))
        .filter(pl.col("diff").abs() > TOLERANCE)
    )


def check_nflverse(stats: pl.DataFrame) -> dict[str, dict[str, int]]:
    """nflverse's fantasy_points only count scrimmage fumbles; we also count return fumbles.
    A difference is 'explained' when it is exactly that: fumbles_lost x the uncounted fumbles."""
    uncounted = pl.col("fumbles_lost_total").fill_null(0) - pl.sum_horizontal(
        pl.col(c).fill_null(0) for c in FUMBLE_LOST_PARTS
    )
    result = {}
    for ref, preset in [("fantasy_points", "standard"), ("fantasy_points_ppr", "standard-ppr")]:
        scoring = PRESETS[preset].scoring
        bad = mismatches(stats, scoring, ref)
        explained = (pl.col("diff") - uncounted * scoring.fumbles_lost).abs() <= TOLERANCE
        result[ref] = {"mismatches": bad.height, "unexplained": bad.filter(~explained).height}
    return result


def fetch_sleeper_week(season: int, week: int) -> pl.DataFrame:
    with urllib.request.urlopen(SLEEPER_URL.format(season=season, week=week), timeout=30) as resp:
        data = json.load(resp)
    rows = [
        {"sleeper_id": int(sid), **{k: s.get(k) for k in SLEEPER_SCORING}}
        for sid, s in data.items() if sid.isdigit() and "pts_ppr" in s
    ]
    return pl.DataFrame(rows).with_columns(pl.lit(season).alias("season"), pl.lit(week).alias("week"))


def join_sleeper(stats: pl.DataFrame, sleeper: pl.DataFrame, id_map: pl.DataFrame) -> pl.DataFrame:
    """Our stat rows for offensive players, with Sleeper's points attached (inner join)."""
    return (
        stats.filter(pl.col("position").is_in(OFFENSE) & (pl.col("season_type") == "REG"))
        .join(id_map, left_on="player_id", right_on="gsis_id", how="inner")
        .join(sleeper, on=["sleeper_id", "season", "week"], how="inner")
    )


def check_sleeper(stats: pl.DataFrame, weeks: list[tuple[int, int]]) -> dict[str, pl.DataFrame | int]:
    id_map = nflreadpy.load_ff_playerids().select("gsis_id", "sleeper_id").drop_nulls().unique("gsis_id")
    sleeper = pl.concat([fetch_sleeper_week(s, w) for s, w in weeks])
    joined = join_sleeper(stats, sleeper, id_map)
    result = {"compared": joined.height}
    for col, scoring in SLEEPER_SCORING.items():
        result[col] = mismatches(joined, scoring, col).select(
            "season", "week", "player_display_name", "position", "ours", col, "diff"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weeks", nargs="+", default=DEFAULT_WEEKS, help="season-week, e.g. 2024-5")
    args = parser.parse_args()

    stats = read_player_stats(default_seasons())
    print(f"nflverse: {stats.height:,} player-weeks")
    for col, counts in check_nflverse(stats).items():
        print(f"  {col}: {counts['mismatches']} differ, {counts['unexplained']} not explained by return fumbles")

    weeks = [tuple(int(x) for x in w.split("-")) for w in args.weeks]
    result = check_sleeper(stats, weeks)
    print(f"Sleeper: {result['compared']:,} offensive player-weeks across {len(weeks)} weeks")
    for col in SLEEPER_SCORING:
        bad = result[col]
        print(f"  {col}: {bad.height} mismatches")
        if bad.height:
            with pl.Config(tbl_rows=20, tbl_width_chars=120):
                print(bad.sort(pl.col("diff").abs(), descending=True).head(20))


if __name__ == "__main__":
    main()
