"""Run the walk-forward backtest and write a comparison report (SCRUM-42).

    python -m src.backtest                      # every registered model
    python -m src.backtest --models baseline
"""

import argparse
from datetime import date
from pathlib import Path

import polars as pl

from src.backtest.harness import compare, run, summarize, weekly_metrics
from src.models.baseline import Baseline
from src.models.ridge import RidgeModel

FEATURES_PATH = Path("data/processed/features.parquet")
REPORT_DIR = Path("docs/reports")
MODELS = {"baseline": Baseline, "ridge": RidgeModel}
TOP_FEATURES = 5


def markdown(df: pl.DataFrame) -> str:
    rows = [df.columns, ["---"] * df.width] + [[str(v) for v in r] for r in df.rows()]
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    args = parser.parse_args()

    feats = pl.read_parquet(FEATURES_PATH)
    models = {name: MODELS[name]() for name in args.models}
    results = pl.concat([run(model, feats) for model in models.values()])
    weekly = weekly_metrics(results)
    summary = summarize(weekly)

    lines = [f"# Backtest report ({date.today()})", "",
             "Walk-forward over 2021-2024 (2025 untouched). Standard PPR. Values: mean [95% CI over weeks].", "",
             markdown(summary)]
    names = args.models
    for other in names[1:]:
        lines += ["", f"## {other} minus {names[0]} (paired; MAE: negative = better, Spearman: positive = better)", "",
                  markdown(compare(weekly, other, names[0]))]
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
    path = REPORT_DIR / f"backtest_{date.today()}.md"
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
