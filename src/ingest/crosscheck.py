"""Cross-check weekly player stats against totals rebuilt from play-by-play.

nflverse builds both datasets from the same NFL feed, so agreement here proves the weekly table
aggregates plays correctly (no dropped games, double counts, or misattributed players). It does
not prove the feed itself matches official stats; see the external spot check for that.

Usage:
    python -m src.ingest.crosscheck --seasons 2023 2024
"""

import argparse
from pathlib import Path

import nflreadpy
import polars as pl

from src.ingest.ids import normalize_name
from src.ingest.player_stats import default_seasons, read_player_stats

OUT_DIR = Path("data/processed/crosscheck")

STATS = [
    "attempts", "completions", "passing_yards", "passing_tds", "passing_interceptions",
    "carries", "rushing_yards", "rushing_tds",
    "targets", "receptions", "receiving_yards", "receiving_tds",
]

PBP_COLS = [
    "game_id", "play_type", "two_point_attempt", "pass_attempt", "complete_pass", "sack",
    "interception", "rush_attempt", "passer_player_id", "receiver_player_id", "rusher_player_id",
    "passing_yards", "receiving_yards", "rushing_yards", "pass_touchdown", "rush_touchdown",
    "td_player_id",
]


def _sum_by(plays: pl.DataFrame, id_col: str, aggs: list[pl.Expr]) -> pl.DataFrame:
    return (
        plays.filter(pl.col(id_col).is_not_null())
        .group_by("game_id", id_col)
        .agg(aggs)
        .rename({id_col: "player_id"})
    )


def stats_from_pbp(pbp: pl.DataFrame) -> pl.DataFrame:
    """Per player-game box score rebuilt from individual plays."""
    plays = pbp.select(PBP_COLS).filter(
        pl.col("play_type").is_in(["pass", "run", "qb_kneel", "qb_spike"])
        & (pl.col("two_point_attempt").fill_null(0) == 0)
    )
    dropback = (pl.col("pass_attempt") == 1) & (pl.col("sack").fill_null(0) == 0)
    flag = lambda cond: cond.cast(pl.Int64).sum()

    passing = _sum_by(plays, "passer_player_id", [
        flag(dropback).alias("attempts"),
        flag(pl.col("complete_pass") == 1).alias("completions"),
        pl.col("passing_yards").fill_null(0).sum().alias("passing_yards"),
        flag(pl.col("pass_touchdown") == 1).alias("passing_tds"),
        flag(pl.col("interception") == 1).alias("passing_interceptions"),
    ])
    rushing = _sum_by(plays, "rusher_player_id", [
        flag(pl.col("rush_attempt") == 1).alias("carries"),
        pl.col("rushing_yards").fill_null(0).sum().alias("rushing_yards"),
        flag((pl.col("rush_touchdown") == 1) & (pl.col("td_player_id") == pl.col("rusher_player_id"))).alias("rushing_tds"),
    ])
    receiving = _sum_by(plays, "receiver_player_id", [
        flag(dropback).alias("targets"),
        flag(pl.col("complete_pass") == 1).alias("receptions"),
        pl.col("receiving_yards").fill_null(0).sum().alias("receiving_yards"),
        flag((pl.col("pass_touchdown") == 1) & (pl.col("td_player_id") == pl.col("receiver_player_id"))).alias("receiving_tds"),
    ])

    out = passing
    for part in (rushing, receiving):
        out = out.join(part, on=["game_id", "player_id"], how="full", coalesce=True)
    return out.with_columns(pl.col(STATS).fill_null(0))


def compare(weekly: pl.DataFrame, rebuilt: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Returns (per-stat summary, mismatched rows). Only player-games with offensive activity count."""
    weekly = weekly.select(["game_id", "player_id", "player_display_name", "position", *STATS]).with_columns(
        pl.col(STATS).fill_null(0)
    )
    weekly = weekly.filter(pl.sum_horizontal(pl.col(STATS).abs()) > 0)
    joined = weekly.join(rebuilt, on=["game_id", "player_id"], how="full", coalesce=True, suffix="_pbp").with_columns(
        pl.col(STATS).fill_null(0), pl.col([f"{s}_pbp" for s in STATS]).fill_null(0)
    )

    rows = []
    for s in STATS:
        diff = (joined[s] - joined[f"{s}_pbp"]).abs()
        active = (joined[s] != 0) | (joined[f"{s}_pbp"] != 0)
        n_active = int(active.sum())
        n_off = int((diff > 0).sum())
        rows.append({
            "stat": s,
            "player_games": n_active,
            "mismatches": n_off,
            "match_pct": round(100 * (1 - n_off / n_active), 3) if n_active else 100.0,
            "total_weekly": float(joined[s].sum()),
            "total_pbp": float(joined[f"{s}_pbp"].sum()),
        })
    summary = pl.DataFrame(rows)

    any_off = pl.any_horizontal([pl.col(s) != pl.col(f"{s}_pbp") for s in STATS])
    mismatches = joined.filter(any_off)
    return summary, mismatches


def compare_to_official(season: int, official_csv: Path) -> pl.DataFrame:
    """Compare regular-season totals to an official reference file (columns: player, stat, official)."""
    official = pl.read_csv(official_csv).with_columns(normalize_name(pl.col("player")).alias("key"))
    weekly = read_player_stats([season]).filter(pl.col("season_type") == "REG")
    totals = (
        weekly.with_columns(normalize_name(pl.col("player_display_name")).alias("key"))
        .group_by("key")
        .agg(pl.col(STATS).sum())
        .unpivot(index="key", variable_name="stat", value_name="nflverse")
    )
    return (
        official.join(totals, on=["key", "stat"], how="left")
        .with_columns((pl.col("nflverse") - pl.col("official")).alias("diff"))
        .select("category", "player", "stat", "official", "nflverse", "diff")
    )


def run(seasons: list[int]) -> pl.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for season in seasons:
        weekly = read_player_stats([season])
        rebuilt = stats_from_pbp(nflreadpy.load_pbp([season]))
        summary, mismatches = compare(weekly, rebuilt)
        mismatches.write_csv(OUT_DIR / f"mismatches_{season}.csv")
        summaries.append(summary.with_columns(pl.lit(season).alias("season")))
        print(f"{season}: {mismatches.height} player-games with any mismatch")

    combined = (
        pl.concat(summaries)
        .group_by("stat", maintain_order=True)
        .agg(
            pl.col("player_games").sum(),
            pl.col("mismatches").sum(),
            pl.col("total_weekly").sum(),
            pl.col("total_pbp").sum(),
        )
        .with_columns((100 * (1 - pl.col("mismatches") / pl.col("player_games"))).round(3).alias("match_pct"))
    )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=None)
    args = parser.parse_args()
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(run(args.seasons or default_seasons()[:-1]))


if __name__ == "__main__":
    main()
