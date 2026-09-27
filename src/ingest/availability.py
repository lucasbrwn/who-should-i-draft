"""Download snap counts (who actually played, and how much) and injury reports (what teams
listed before each game), link both to gsis_id, and cross-check them against each other.

Usage:
    python -m src.ingest.availability                 # 2020 through the current season
    python -m src.ingest.availability --seasons 2024
"""

import argparse
from pathlib import Path

import nflreadpy
import polars as pl

from src.ingest.ids import attach_gsis_id, pfr_to_gsis
from src.ingest.player_stats import default_seasons, read_player_stats

SNAPS_DIR = Path("data/raw/snap_counts")
INJURIES_DIR = Path("data/raw/injuries")

SKILL_POSITIONS = ["QB", "RB", "FB", "WR", "TE"]

PRACTICE_CODES = {
    "Did Not Participate In Practice": "DNP",
    "Limited Participation in Practice": "LP",
    "Full Participation in Practice": "FP",
}


def snaps_path(season: int) -> Path:
    return SNAPS_DIR / f"snap_counts_{season}.parquet"


def injuries_path(season: int) -> Path:
    return INJURIES_DIR / f"injuries_{season}.parquet"


def load_snaps(season: int, id_map: pl.DataFrame) -> pl.DataFrame:
    snaps = nflreadpy.load_snap_counts([season])
    return attach_gsis_id(snaps, id_map, read_player_stats([season]))


INJURY_KEY = ["season", "game_type", "week", "team", "gsis_id"]


def load_injuries(season: int) -> pl.DataFrame:
    """One row per player per week: final game status plus last practice status of the week.

    A player can be reported more than once in a week when their status changes (e.g.
    Questionable -> Out on game morning). We keep the latest report as `report_status` and
    the first as `initial_report_status`, so late downgrades stay visible.
    """
    return clean_injuries(nflreadpy.load_injuries([season]))


def clean_injuries(raw: pl.DataFrame) -> pl.DataFrame:
    raw = raw.with_columns(
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("practice_status").replace_strict(PRACTICE_CODES, default=None).alias("practice_code"),
    )
    # 2025+ files dropped date_modified; fall back to file order (reports are appended in time order)
    order = "date_modified" if "date_modified" in raw.columns else "row_nr"
    return (
        raw.with_row_index("row_nr")
        .sort(order, maintain_order=True)
        .with_columns(pl.col("report_status").first().over(INJURY_KEY).alias("initial_report_status"))
        .unique(INJURY_KEY, keep="last", maintain_order=True)
        .drop("row_nr")
    )


def check_snaps(snaps: pl.DataFrame) -> tuple[list[str], list[str]]:
    issues, notes = [], []
    dupes = snaps.height - snaps.unique(["game_id", "pfr_player_id"]).height
    if dupes:
        issues.append(f"{dupes} duplicate player-game rows")
    bad_pct = snaps.filter(~pl.col("offense_pct").is_between(0, 1)).height
    if bad_pct:
        issues.append(f"{bad_pct} rows with offense_pct outside 0-1")

    skill = snaps.filter(pl.col("position").is_in(SKILL_POSITIONS) & (pl.col("offense_snaps") > 0))
    unmapped = skill.filter(pl.col("gsis_id").is_null())
    by_name = skill.filter(pl.col("id_source") == "name_match").height
    notes.append(
        f"skill-position player-games linked to gsis_id: {skill.height - unmapped.height}/{skill.height}"
        f" ({by_name} via name match)"
    )
    if unmapped.height:
        names = unmapped.group_by("player").agg(pl.col("offense_snaps").sum()).sort("offense_snaps", descending=True)
        notes.append(f"unlinked: {', '.join(f'{p} ({s:.0f} snaps)' for p, s in names.head(5).rows())}")
    return issues, notes


def check_injuries(injuries: pl.DataFrame) -> list[str]:
    issues = []
    dupes = injuries.height - injuries.unique(INJURY_KEY).height
    if dupes:
        issues.append(f"{dupes} duplicate player-week rows")
    missing_id = injuries["gsis_id"].null_count()
    if missing_id:
        issues.append(f"{missing_id} rows missing gsis_id")
    return issues


def out_but_played(injuries: pl.DataFrame, snaps: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    """Players listed Out who still recorded snaps in that week's game. Should be (nearly) empty."""
    games = pl.concat([
        schedules.select("game_id", "season", "week", "game_type", pl.col("home_team").alias("team")),
        schedules.select("game_id", "season", "week", "game_type", pl.col("away_team").alias("team")),
    ])
    out = injuries.filter(pl.col("report_status") == "Out").join(games, on=["season", "week", "game_type", "team"])
    played = snaps.filter((pl.col("offense_snaps") + pl.col("defense_snaps") + pl.col("st_snaps")) > 0)
    return out.join(played.select("game_id", "gsis_id", "offense_snaps", "defense_snaps", "st_snaps"),
                    on=["game_id", "gsis_id"])


def ingest(seasons: list[int]) -> None:
    from src.ingest.schedules import read_schedules

    SNAPS_DIR.mkdir(parents=True, exist_ok=True)
    INJURIES_DIR.mkdir(parents=True, exist_ok=True)
    id_map = pfr_to_gsis()

    for season in seasons:
        snaps = load_snaps(season, id_map)
        snaps.write_parquet(snaps_path(season))
        injuries = load_injuries(season)
        injuries.write_parquet(injuries_path(season))

        issues, notes = check_snaps(snaps)
        issues += check_injuries(injuries)
        contradictions = out_but_played(injuries, snaps, read_schedules([season]))
        n_out = injuries.filter(pl.col("report_status") == "Out").height
        changed = injuries.filter(pl.col("report_status").ne_missing(pl.col("initial_report_status"))).height

        print(f"{season}: {snaps.height:,} snap rows, {injuries.height:,} injury rows "
              f"({n_out:,} Out; {contradictions.height} of those played; {changed} status changes in-week)")
        for issue in issues:
            print(f"  ! {issue}")
        for note in notes:
            print(f"  - {note}")
        for r in contradictions.head(3).iter_rows(named=True):
            print(f"  - Out but played: {r['full_name']} ({r['team']}) wk {r['week']}: "
                  f"{r['offense_snaps']:.0f} off / {r['defense_snaps']:.0f} def / {r['st_snaps']:.0f} ST snaps")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=None)
    args = parser.parse_args()
    ingest(args.seasons or default_seasons())


if __name__ == "__main__":
    main()
