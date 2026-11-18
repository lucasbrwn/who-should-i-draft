"""Snapshot the benchmarks we're graded against, at the same moment as our own predictions.

    ESPN         weekly point projections (what most casual players see in their app), PPR scoring.
                 ESPN's public fantasy data (the same JSON the ESPN site loads; no login, nothing about
                 the user is sent). Gives points, so we compare points error AND head-to-head calls.
    FantasyPros  weekly expert consensus rankings (ECR) by position, PPR, via nflverse. The "experts"
                 bar from our backtest (tied 49.8%). Ranks only, so head-to-head calls only.

Taking them at the same time as our run matters: projections move all week with injury news, so
grabbing ESPN's Sunday numbers would hand them information we didn't have.

Usage (also runs automatically inside python -m src.predict):
    python -m src.predict.benchmarks --season 2026 --week 11 --out predictions/<run folder>
"""

import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy
import polars as pl

from src.ingest.consensus import IDS_PATH

ESPN_URL = ("https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}"
            "/segments/0/leaguedefaults/3?scoringPeriodId={week}&view=kona_player_info")
ESPN_SLOTS = {0: "QB", 2: "RB", 4: "WR", 6: "TE"}   # ESPN lineup slot ids
PROJECTION = 1                                       # ESPN statSourceId: 0 = actual, 1 = projected
FP_PAGES = {"qb": "QB", "ppr-rb": "RB", "ppr-wr": "WR", "ppr-te": "TE"}


def fetch_espn(season: int, week: int) -> list[dict]:
    flt = {"players": {"limit": 1500, "filterSlotIds": {"value": list(ESPN_SLOTS)},
                       "filterStatsForSourceIds": {"value": [PROJECTION]},
                       "filterStatsForScoringPeriodIds": {"value": [week]},
                       "sortPercOwned": {"sortAsc": False, "sortPriority": 1}}}
    req = urllib.request.Request(ESPN_URL.format(season=season, week=week), headers={
        "X-Fantasy-Filter": json.dumps(flt), "User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)["players"]


def parse_espn(players: list[dict], season: int, week: int) -> pl.DataFrame:
    """One row per player with an ESPN projection for that week (PPR points)."""
    rows = []
    for entry in players:
        p = entry["player"]
        for s in p.get("stats", []):
            if (s.get("seasonId") == season and s.get("scoringPeriodId") == week
                    and s.get("statSourceId") == PROJECTION and s.get("statSplitTypeId") == 1):
                rows.append({"espn_id": str(p["id"]), "espn_name": p["fullName"],
                             "espn_proj": round(float(s.get("appliedTotal", 0.0)), 2)})
    return pl.DataFrame(rows, schema={"espn_id": pl.String, "espn_name": pl.String, "espn_proj": pl.Float64})


def player_ids() -> pl.DataFrame:
    return pl.read_parquet(IDS_PATH)


def espn_projections(season: int, week: int) -> pl.DataFrame:
    ids = player_ids().select(pl.col("espn_id").cast(pl.String), pl.col("gsis_id").alias("player_id")) \
        .drop_nulls().unique("espn_id")
    return parse_espn(fetch_espn(season, week), season, week).join(ids, on="espn_id", how="left")


def fantasypros_week() -> pl.DataFrame:
    """This week's FantasyPros expert consensus rankings (position pages, PPR)."""
    ids = player_ids().select(pl.col("fantasypros_id").cast(pl.String), pl.col("gsis_id").alias("player_id")) \
        .drop_nulls().unique("fantasypros_id")
    r = nflreadpy.load_ff_rankings("week").filter(pl.col("page").is_in(list(FP_PAGES)))
    r = r.filter(pl.col("page").replace_strict(FP_PAGES, default=None) == pl.col("pos"))
    return r.select(fantasypros_id=pl.col("fantasypros_id").cast(pl.String), fp_name=pl.col("player_name"),
                    position=pl.col("pos"), fp_ecr=pl.col("ecr"), fp_pos_rank=pl.col("pos_rank"),
                    fp_scrape_date=pl.col("scrape_date")).join(ids, on="fantasypros_id", how="left")


def snapshot(season: int, week: int, out: Path) -> str:
    """Save espn.csv and fantasypros.csv into a prediction run folder; return a one-line summary."""
    fetched = datetime.now(timezone.utc).isoformat()
    espn = espn_projections(season, week).with_columns(fetched_utc=pl.lit(fetched))
    fp = fantasypros_week().with_columns(fetched_utc=pl.lit(fetched))
    espn.write_csv(out / "espn.csv")
    fp.write_csv(out / "fantasypros.csv")
    return (f"benchmarks: ESPN {espn.height} projections ({espn['player_id'].is_not_null().sum()} matched to our ids), "
            f"FantasyPros {fp.height} ranked (scraped {fp['fp_scrape_date'].max()}, "
            f"{fp['player_id'].is_not_null().sum()} matched)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(snapshot(args.season, args.week, args.out))


if __name__ == "__main__":
    main()
