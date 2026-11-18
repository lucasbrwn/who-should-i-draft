"""Does tier calibration help weekly projections? (rule set 2026-11-17, before results)

For each test season S, fit the isotonic curves (src/models/weekly_calibration.py) on the walk-forward
predictions of EARLIER seasons only (2021+, never 2025), apply them to season S, compare per week.

    rule seasons   2022, 2023, 2024
    extra check    2026 weeks 1-10 (curves fit on 2021-24): not used when finding the problem, so the
                   cleanest out-of-sample look (the bend itself was spotted in 2022-24 data)

Rule: keep only if MAE improves significantly (paired week-bootstrap 95% CI below 0) at some position,
no position significantly worse on MAE or Spearman, AND the top-5% bias shrinks at RB and WR.

    python -m src.backtest.calibration_test
"""

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from src.backtest.harness import run
from src.backtest.play_chance_test import paired_ci
from src.models.boosting import TUNED_PATH, BoostingModel
from src.models.weekly_calibration import POSITIONS, WeeklyCalibration

PRED = "data/processed/weekly_predictions_{}.parquet"
RULE_SEASONS, EXTRA = [2022, 2023, 2024], 2026
REPORT = Path(f"docs/reports/calibration_test_{date.today()}.md")


def preds(season: int) -> pl.DataFrame:
    path = Path(PRED.format(season))
    if not path.exists():
        feats = pl.read_parquet("data/processed/features.parquet").filter(pl.col("season") <= season)
        run(BoostingModel(tuned=TUNED_PATH, name="boosting"), feats, [season]).write_parquet(path)
    return pl.read_parquet(path).select("player_id", "position", "season", "week", "pred", "actual")


def calibrated(season: int) -> pl.DataFrame:
    history = pl.concat([preds(s) for s in range(2021, season) if s != 2025])
    test = preds(season)
    cal = WeeklyCalibration().fit(history)
    return test.with_columns(cal=pl.Series(cal.apply(test["pred"].to_numpy(), test["position"].to_numpy())))


def compare(d: pl.DataFrame) -> tuple[list[str], bool]:
    lines, ok_mae, ok_none_worse = [], False, True
    lines += ["| position | MAE before | MAE after | diff [95% CI] | Spearman diff [95% CI] |",
              "| --- | --- | --- | --- | --- |"]
    for pos in POSITIONS:
        g = d.filter(pl.col("position") == pos)
        rows = []
        for _, wk in g.group_by("season", "week"):
            a = wk["actual"].to_numpy()
            rows.append({"mae0": np.abs(wk["pred"].to_numpy() - a).mean(), "mae1": np.abs(wk["cal"].to_numpy() - a).mean(),
                         "sp0": spearmanr(wk["pred"], a).statistic, "sp1": spearmanr(wk["cal"], a).statistic})
        w = pl.DataFrame(rows)
        md, ml, mh = paired_ci((w["mae1"] - w["mae0"]).to_numpy())
        sd, sl, sh = paired_ci((w["sp1"] - w["sp0"]).drop_nans().to_numpy())
        ok_mae |= mh < 0
        ok_none_worse &= (ml <= 0) and (sh >= 0)
        lines.append(f"| {pos} | {w['mae0'].mean():.3f} | {w['mae1'].mean():.3f} | {md:+.3f} [{ml:+.3f}, {mh:+.3f}] "
                     f"| {sd:+.4f} [{sl:+.4f}, {sh:+.4f}] |")
    return lines, ok_mae and ok_none_worse


def tier_bias(d: pl.DataFrame) -> tuple[list[str], dict]:
    t = d.with_columns(tier=pl.col("pred").qcut([0.5, 0.8, 0.95], labels=["bottom 50%", "50-80%", "80-95%", "top 5%"])
                       .over("position"))
    g = t.group_by("position", "tier").agg(before=(pl.col("actual") - pl.col("pred")).mean(),
                                           after=(pl.col("actual") - pl.col("cal")).mean(), n=pl.len()).sort("position", "tier")
    lines = ["| position | tier | n | bias before (actual - proj) | bias after |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {r['position']} | {r['tier']} | {r['n']} | {r['before']:+.2f} | {r['after']:+.2f} |" for r in g.iter_rows(named=True)]
    top = {r["position"]: (r["before"], r["after"]) for r in g.filter(pl.col("tier") == "top 5%").iter_rows(named=True)}
    return lines, top


def main() -> None:
    d = pl.concat([calibrated(s) for s in RULE_SEASONS])
    cmp_lines, ok = compare(d)
    bias_lines, top = tier_bias(d)
    shrinks = all(abs(top[p][1]) < abs(top[p][0]) for p in ["RB", "WR"])
    extra = calibrated(EXTRA)
    extra_lines, _ = compare(extra)
    lines = [f"# Tier calibration test ({date.today()})", "", __doc__.split("\n\n")[1], "",
             "## Rule seasons 2022-2024", "", *cmp_lines, "", *bias_lines, "",
             f"## Extra check: {EXTRA} weeks 1-10 (curves fit on 2021-2024)", "", *extra_lines, "",
             "RULE: " + ("PASS -> calibrate live projections" if ok and shrinks
                         else f"FAIL (MAE gain & none worse: {ok}; top-5% bias shrinks RB+WR: {shrinks})")]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
