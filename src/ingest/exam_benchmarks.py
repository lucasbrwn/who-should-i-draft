"""SCRUM-99: cache the 2025 expert FORECASTS the final exam needs to grade the blend.

The product's weekly number is the blend (ours + ESPN + FantasyPros). The final exam (SCRUM-66) must be
able to grade it on 2025, so this caches, BEFORE the exam:
    ESPN         weekly PPR projections, 2025 weeks 1-18 (same public JSON and cache as vs_espn.py;
                 statSourceId 1 = projected only, so no actual results are downloaded)
    FantasyPros  weekly ECR snapshots for 2025 already in data/raw/ff_rankings_all.parquet

It reports only coverage: how many players each week and how many match our player ids.
It never reads 2025 points, player_weeks outcomes, or game scores.

    python -m src.ingest.exam_benchmarks
"""

from pathlib import Path

import polars as pl

from src.backtest.weekly_vs_consensus import PAGES
from src.ingest.consensus import IDS_PATH, RANKINGS_PATH
from src.predict.benchmarks import espn_projections

SEASON = 2025
WEEKS = list(range(1, 19))
CACHE = Path("data/raw/espn")


def cache_espn() -> pl.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    for week in WEEKS:
        path = CACHE / f"espn_proj_{SEASON}_wk{week:02d}.parquet"
        if not path.exists():
            espn_projections(SEASON, week).write_parquet(path)
        p = pl.read_parquet(path)
        rows.append({"week": week, "espn_players": p.height, "espn_matched": p["player_id"].is_not_null().sum(),
                     "espn_nonzero": (p["espn_proj"] > 0).sum()})
    return pl.DataFrame(rows)


def fp_coverage() -> pl.DataFrame:
    """Snapshots per scrape date in the 2025 season window, position pages only."""
    ids = pl.read_parquet(IDS_PATH).select(pl.col("fantasypros_id").cast(pl.String)).drop_nulls().unique()
    r = (pl.read_parquet(RANKINGS_PATH).filter(pl.col("fp_page").is_in(list(PAGES.values())))
         .with_columns(d=pl.col("scrape_date").str.to_date(strict=False)))
    r = r.filter((pl.col("d") >= pl.date(2025, 9, 1)) & (pl.col("d") <= pl.date(2026, 1, 10)))
    return (r.with_columns(matched=pl.col("id").cast(pl.String).is_in(ids["fantasypros_id"].implode()))
            .group_by("d").agg(fp_players=pl.len(), fp_matched=pl.col("matched").sum(),
                                weekday=pl.col("d").first().dt.weekday())
            .sort("d"))


def main() -> None:
    pl.Config.set_tbl_rows(40)
    espn = cache_espn()
    print("ESPN 2025 projections (cached in data/raw/espn):")
    print(espn)
    fp = fp_coverage()
    print(f"\nFantasyPros 2025 snapshots in {RANKINGS_PATH}: {fp.height} scrape dates "
          f"({(fp['weekday'] == 5).sum()} on Fridays)")
    print(fp)


if __name__ == "__main__":
    main()
