"""Weekly start/sit vs FantasyPros weekly expert rankings (SCRUM-55 part 2 / SCRUM-56).

The season-level test (vs_consensus.py) tied the experts, and its CI is wide: three seasons of
draft pairs aren't many decisions. Weekly lineups give thousands more.

Rules (set 2026-11-14 before seeing results; Lucas: "go with your recommended"):
    market    FantasyPros weekly PPR position rankings (qb.php, ppr-rb/wr/te.php), the Friday snapshot
              for that week. Players whose game was before the snapshot (Thursday night) are dropped,
              since the experts may have seen the result.
    pool      the experts' weekly top 24 QB / 48 RB / 60 WR / 24 TE: realistic start/sit range
    pairs     same season, week and position, within 6 spots in the experts' ranks
    outcome   actual PPR points that week; both players must have played (a row in player_weeks)
    ours      walk-forward weekly boosting model (src.backtest.harness.run): trained only on earlier weeks
    headline  on pairs where we disagree with the experts, how often we're right; player-level bootstrap CI
    seasons   2022-2024. 2025 is the final exam.

Usage:
    python -m src.backtest.weekly_vs_consensus
"""

from datetime import date
from pathlib import Path

import polars as pl

from src.backtest.harness import run
from src.backtest.vs_consensus import bootstrap_rate, score_pairs, table
from src.features.team import read_schedules
from src.ingest.consensus import IDS_PATH, RANKINGS_PATH
from src.models.boosting import TUNED_PATH, BoostingModel

FEATURES_PATH = Path("data/processed/features.parquet")
PREDICTIONS_PATH = Path("data/processed/weekly_predictions_boosting.parquet")
REPORT_DIR = Path("docs/reports")
SEASONS = [2022, 2023, 2024]
PAGES = {"QB": "/nfl/rankings/qb.php", "RB": "/nfl/rankings/ppr-rb.php",
         "WR": "/nfl/rankings/ppr-wr.php", "TE": "/nfl/rankings/ppr-te.php"}
POOL = {"QB": 24, "RB": 48, "WR": 60, "TE": 24}
WINDOW = 6


def snapshot_weeks(schedules: pl.DataFrame) -> pl.DataFrame:
    """Each regular-season week's first and last game day."""
    return (schedules.filter(pl.col("game_type") == "REG")
            .group_by("season", "week").agg(first=pl.col("gameday").str.to_date().min(),
                                            last=pl.col("gameday").str.to_date().max()))


def weekly_ecr(seasons: list[int]) -> pl.DataFrame:
    """season, week, player_id, position, rank (expert position rank) from the snapshot inside each week."""
    sched = read_schedules()
    weeks = snapshot_weeks(sched).filter(pl.col("season").is_in(seasons))
    ids = pl.read_parquet(IDS_PATH).select(pl.col("fantasypros_id").cast(pl.String), "gsis_id").drop_nulls().unique("fantasypros_id")
    r = (pl.read_parquet(RANKINGS_PATH).filter(pl.col("fp_page").is_in(list(PAGES.values())))
         .with_columns(d=pl.col("scrape_date").cast(pl.Date), position=pl.col("pos")))
    r = r.filter(pl.col("fp_page") == pl.col("position").replace_strict(PAGES, default=None))
    # A snapshot belongs to the week whose games straddle it (Friday between TNF and MNF).
    snaps = r.join(weeks, how="cross").filter(pl.col("d") >= pl.col("first"), pl.col("d") <= pl.col("last"))
    snaps = snaps.filter(pl.col("d") == pl.col("d").max().over("season", "week"))  # latest if several
    out = (snaps.select("season", "week", "position", "d", fantasypros_id=pl.col("id").cast(pl.String),
                        rank=pl.col("ecr").cast(pl.Float64))
           .join(ids, on="fantasypros_id", how="inner").rename({"gsis_id": "player_id"})
           .unique(["season", "week", "player_id"], keep="first"))
    return out


def predictions(rebuild: bool = False) -> pl.DataFrame:
    if rebuild or not PREDICTIONS_PATH.exists():
        feats = pl.read_parquet(FEATURES_PATH)
        res = run(BoostingModel(tuned=TUNED_PATH, name="boosting"), feats, SEASONS)
        res = res.join(feats.select("player_id", "season", "week", "gameday" if "gameday" in feats.columns else "game_id"),
                       on=["player_id", "season", "week"], how="left")
        res.write_parquet(PREDICTIONS_PATH)
    return pl.read_parquet(PREDICTIONS_PATH)


def game_days(seasons: list[int]) -> pl.DataFrame:
    pw = pl.read_parquet("data/processed/player_weeks.parquet", columns=["player_id", "season", "week", "gameday"])
    return pw.filter(pl.col("season").is_in(seasons)).with_columns(gameday=pl.col("gameday").str.to_date())


def build() -> pl.DataFrame:
    ecr = weekly_ecr(SEASONS)
    pred = predictions().select("player_id", "season", "week", "pred", "actual")
    days = game_days(SEASONS)
    p = (ecr.join(days, on=["player_id", "season", "week"], how="inner")      # he played that week
         .filter(pl.col("gameday") > pl.col("d"))                              # game after the snapshot
         .join(pred, on=["player_id", "season", "week"], how="inner"))
    p = p.filter(pl.col("rank").rank("ordinal").over("season", "week", "position")
                 <= pl.col("position").replace_strict(POOL, default=0))
    return p


def pairs_weekly(p: pl.DataFrame) -> pl.DataFrame:
    j = p.join(p, on=["season", "week", "position"], suffix="_b").filter(
        pl.col("rank") < pl.col("rank_b"), (pl.col("rank_b") - pl.col("rank")) <= WINDOW)
    return j


def main() -> None:
    p = build()
    scored = score_pairs(pairs_weekly(p), "actual", ours="pred")
    rows = []
    for label, s in [("ALL", scored), *[(pos, scored.filter(pl.col("position") == pos)) for pos in POOL],
                     *[(str(y), scored.filter(pl.col("season") == y)) for y in SEASONS]]:
        dis = s.filter("disagree")
        win, lo, hi = bootstrap_rate(dis, "we_right")
        rows.append({"group": label, "pairs": s.height, "disagree": dis.height, "win_rate": win, "lo": lo, "hi": hi,
                     "ours_acc": s["we_right"].mean(), "market_acc": s["market_right"].mean()})
    text = "\n".join([f"# Weekly start/sit vs FantasyPros weekly expert rankings ({date.today()})", "",
                      f"Seasons {SEASONS} (2025 untouched). Pool: experts' weekly top {POOL}. Same-position pairs within "
                      f"{WINDOW} expert spots, both played; outcome = that week's PPR points. Thursday games dropped "
                      f"(snapshot is Friday). Weeks covered: {p.select('season', 'week').unique().height}.", "",
                      *table(rows)]) + "\n"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"weekly_vs_consensus_{date.today()}.md"
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
