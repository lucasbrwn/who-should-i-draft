"""Should 2026 use ESTIMATED routes, or leave route features blank? (rule set 2026-11-17, before results)

2026 has no participation data, so route features are blank. src/features/routes.py can fill them with
an estimate (snaps x team pass rate x position scale). We can't test on 2026 (no truth), so replay
2024 as if its participation data were missing, three ways:

    real       2024 route features from real participation (the ceiling: what true routes are worth)
    blank      2024 routes missing, like 2026 today
    estimated  2024 routes estimated (scale fit on the other seasons' participation)

Rule: use the estimate only if `estimated` beats `blank` on MAE or Spearman at some position with no
position significantly worse (paired week bootstrap, harness.compare).

    python -m src.backtest.route_estimate_test
"""

from datetime import date
from pathlib import Path

import polars as pl

from src.backtest.harness import compare, run, summarize, weekly_metrics
from src.features.routes import FEATURES, PBP_COLS, build_route_features, read_participation
from src.ingest.pbp import read_pbp
from src.models.boosting import TUNED_PATH, BoostingModel

TEST = 2024
FEATURES_PATH = Path("data/processed/features.parquet")
PW_PATH = Path("data/processed/player_weeks.parquet")
REPORT = Path(f"docs/reports/route_estimate_test_{date.today()}.md")
KEY = ["player_id", "season", "week"]


def variant(feats: pl.DataFrame, routes: pl.DataFrame) -> pl.DataFrame:
    return feats.drop(FEATURES).join(routes.select(*KEY, *FEATURES), on=KEY, how="left")


def main() -> None:
    feats = pl.read_parquet(FEATURES_PATH).filter(pl.col("season") <= TEST)
    pw = pl.read_parquet(PW_PATH).filter(pl.col("season") <= TEST)
    seasons = sorted(pw["season"].unique().to_list())
    pbp = read_pbp(seasons, columns=PBP_COLS)
    part = read_participation(seasons)
    without_test = part.filter(~pl.col("game_id").str.starts_with(f"{TEST}_"))
    variants = {
        "real": feats,
        "blank": variant(feats, build_route_features(pw, without_test, pbp, estimate=False)),
        "estimated": variant(feats, build_route_features(pw, without_test, pbp, estimate=True)),
    }
    results = pl.concat([run(BoostingModel(tuned=TUNED_PATH, name=name), f, [TEST]) for name, f in variants.items()])
    weekly = weekly_metrics(results)
    lines = [f"# Route estimate test ({date.today()})", "", __doc__.split("\n\n")[2], "",
             "## Summary (2024)", "", str(summarize(weekly)), "",
             "## estimated minus blank (MAE: negative = better; Spearman: positive = better)", "",
             str(compare(weekly, "estimated", "blank")), "",
             "## real minus blank (the ceiling)", "", str(compare(weekly, "real", "blank"))]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
