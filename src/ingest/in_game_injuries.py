"""Find players hurt during a game, from play-by-play descriptions.

The NFL's play text says e.g. "BUF-7-Ta.Johnson was injured during the play. T.Johnson walks off."
We pull out team + jersey number + abbreviated name and match them to a player using that week's
roster. Being "injured during the play" doesn't mean he left the game (many players walk off and
return), so player_weeks combines this with a snap-share drop before flagging `left_injured`.

Usage:
    python -m src.ingest.in_game_injuries          # downloads weekly rosters, then parses plays
"""

import argparse
from pathlib import Path

import nflreadpy
import polars as pl

from src.ingest.ids import normalize_name
from src.ingest.player_stats import default_seasons

ROSTERS_DIR = Path("data/raw/rosters")
PBP_DIR = Path("data/raw/pbp")
OUT_DIR = Path("data/processed/in_game_injuries")

MENTION = r"([A-Z]{2,3})-(\d{1,2})-([A-Za-z][\w.'\-]*?) was injured"


def rosters_path(season: int) -> Path:
    return ROSTERS_DIR / f"rosters_{season}.parquet"


def out_path(season: int) -> Path:
    return OUT_DIR / f"in_game_injuries_{season}.parquet"


def parse_mentions(pbp: pl.DataFrame) -> pl.DataFrame:
    """One row per injury mention: game, play, quarter, team, jersey, abbreviated name."""
    return (
        pbp.select("game_id", "play_id", "season", "week", "qtr",
                   pl.col("desc").str.extract_all(MENTION).alias("hit"))
        .explode("hit", empty_as_null=False)   # plays with no mention produce no row
        .drop_nulls("hit")
        .with_columns(
            pl.col("hit").str.extract(MENTION, 1).alias("team"),
            pl.col("hit").str.extract(MENTION, 2).cast(pl.Int32).alias("jersey_number"),
            pl.col("hit").str.extract(MENTION, 3).alias("abbr_name"),
        )
        .drop("hit")
    )


def _abbr_matches(abbr: pl.Expr, first: pl.Expr, last: pl.Expr) -> pl.Expr:
    """'Ta.Johnson' matches Tanner Johnson: last name equal, first name starts with 'ta'."""
    initials = normalize_name(abbr.str.extract(r"^([^.]+)\.", 1))
    surname = normalize_name(abbr.str.replace(r"^[^.]+\.", ""))
    return (normalize_name(last) == surname) & normalize_name(first).str.starts_with(initials)


def match_players(mentions: pl.DataFrame, rosters: pl.DataFrame) -> pl.DataFrame:
    """Attach gsis_id. If two roster players share the number (offense + defense can), keep the
    one whose name matches the abbreviation; if that's still ambiguous, leave gsis_id null."""
    roster = rosters.select(
        "season", "week", "team", "jersey_number", "gsis_id", "first_name", "last_name", "position",
    ).drop_nulls(["jersey_number", "gsis_id"]).unique(["season", "week", "team", "jersey_number", "gsis_id"])

    joined = (
        mentions.with_row_index("mention_id")
        .join(roster, on=["season", "week", "team", "jersey_number"], how="left")
        .with_columns(
            _abbr_matches(pl.col("abbr_name"), pl.col("first_name"), pl.col("last_name"))
            .fill_null(False).alias("name_ok"),
            pl.len().over("mention_id").alias("candidates"),
        )
    )
    # unique number -> trust it; shared number -> require the name to match
    keep = joined.filter((pl.col("candidates") == 1) | pl.col("name_ok"))
    keep = keep.filter(pl.len().over("mention_id") == 1)
    lost = joined.select("mention_id").unique().join(keep.select("mention_id"), on="mention_id", how="anti")
    unmatched = mentions.with_row_index("mention_id").join(lost, on="mention_id").with_columns(
        pl.lit(None, pl.String).alias("gsis_id"), pl.lit(None, pl.String).alias("position"),
    )
    cols = [*mentions.columns, "gsis_id", "position"]
    return pl.concat([keep.select(cols), unmatched.select(cols)]).sort("game_id", "play_id")


def per_player_game(matched: pl.DataFrame) -> pl.DataFrame:
    """One row per injured player per game: how many mentions and the first quarter it happened."""
    return (
        matched.drop_nulls("gsis_id")
        .group_by("game_id", "gsis_id")
        .agg(pl.len().alias("injury_mentions"), pl.col("qtr").min().alias("injury_qtr"))
        .rename({"gsis_id": "player_id"})
    )


def download_rosters(seasons: list[int]) -> None:
    ROSTERS_DIR.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        nflreadpy.load_rosters_weekly([season]).write_parquet(rosters_path(season))
        print(f"{season}: weekly rosters -> {rosters_path(season)}")


def ingest(seasons: list[int], download: bool = True) -> None:
    if download:
        download_rosters(seasons)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        rosters = pl.read_parquet(rosters_path(season))
        pbp = pl.read_parquet(PBP_DIR / f"pbp_{season}.parquet",
                              columns=["game_id", "play_id", "season", "week", "qtr", "desc"])
        matched = match_players(parse_mentions(pbp), rosters)
        result = per_player_game(matched)
        result.write_parquet(out_path(season))
        missing = matched["gsis_id"].null_count()
        print(f"{season}: {matched.height:,} injury mentions, {missing} unmatched, "
              f"{result.height:,} player-games -> {out_path(season)}")


def read_in_game_injuries(seasons: list[int]) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(out_path(s)) for s in seasons])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=None)
    args = parser.parse_args()
    ingest(args.seasons or default_seasons())


if __name__ == "__main__":
    main()
