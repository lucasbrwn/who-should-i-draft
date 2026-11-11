"""Run the walk-forward backtest and write a comparison report (SCRUM-42).

    python -m src.backtest                      # every registered model
    python -m src.backtest --models baseline
"""

import argparse
from datetime import date
from pathlib import Path

import polars as pl

from src.backtest.harness import SEASONS, compare, run, summarize, weekly_metrics
from src.models.baseline import Baseline
from src.models.boosting import STAT_ERROR_PATH, TD_STATS, TUNED_PATH, BoostingModel
from src.models.ridge import RidgeModel

FEATURES_PATH = Path("data/processed/features.parquet")
REPORT_DIR = Path("docs/reports")
# "boosting" uses the rank-tuned settings (SCRUM-50 winner); "boosting_default" is the untuned version.
MODELS = {"baseline": Baseline, "ridge": RidgeModel,
          "boosting": lambda: BoostingModel(tuned=TUNED_PATH, name="boosting"),
          "boosting_default": BoostingModel,
          "boosting_tuned": lambda: BoostingModel(tuned=STAT_ERROR_PATH, name="boosting_tuned"),
          "boosting_td": lambda: BoostingModel(tuned=STAT_ERROR_PATH, tuned_stats=TD_STATS, name="boosting_td"),
          "boosting_rank": lambda: BoostingModel(tuned=TUNED_PATH, name="boosting_rank")}
DEFAULT_MODELS = ["baseline", "ridge", "boosting"]
TOP_FEATURES = 5


def markdown(df: pl.DataFrame) -> str:
    rows = [df.columns, ["---"] * df.width] + [[str(v) for v in r] for r in df.rows()]
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=list(MODELS))
    parser.add_argument("--seasons", nargs="+", type=int, default=SEASONS)
    args = parser.parse_args()

    feats = pl.read_parquet(FEATURES_PATH)
    models = {name: MODELS[name]() for name in args.models}
    results = pl.concat([run(model, feats, args.seasons) for model in models.values()])
    weekly = weekly_metrics(results)
    summary = summarize(weekly)

    lines = [f"# Backtest report ({date.today()})", "",
             f"Walk-forward over {args.seasons} (2025 untouched). Standard PPR. Values: mean [95% CI over weeks].", "",
             markdown(summary)]
    names = args.models
    for i, older in enumerate(names):  # every later model vs every earlier one
        for newer in names[i + 1:]:
            lines += ["", f"## {newer} minus {older} (paired; MAE: negative = better, Spearman: positive = better)", "",
                      markdown(compare(weekly, newer, older))]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        if hasattr(model, "coefficients"):
            coefs = model.coefficients()
            csv = REPORT_DIR / f"{name}_coefficients_{date.today()}.csv"
            coefs.write_csv(csv)
            top = coefs.group_by("position", "stat", maintain_order=True).head(TOP_FEATURES)
            lines += ["", f"## {name}: top {TOP_FEATURES} features per stat (latest fit, scaled; full list in {csv.name})", "",
                      markdown(top.with_columns(pl.col("coef").round(3)))]
    report = "\n".join(lines) + "\n"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"backtest_{date.today()}_{'-'.join(args.models)}.md"
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
