"""Where is the weekly model weakest? Evidence for choosing the next improvement (2026-11-17).

Uses the walk-forward predictions from src/backtest/vs_espn.py (2022-2024, current model, each week
trained only on earlier games; players who played). No model changes here, only diagnosis.

    1. segments     MAE and bias (pred - actual) by situation, vs the overall average
    2. components   which stat drives the points error (TDs vs yards vs catches), 2024
    3. calibration  mean actual vs mean projection by projection tier (are stars under-projected?)
    4. vs ESPN      where ESPN's error is lower than ours, by segment

    python -m src.backtest.weak_points
"""

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from src.backtest.harness import SCORING
from src.backtest.vs_espn import CACHE
from src.models.boosting import TUNED_PATH, BoostingModel

SEASONS = [2022, 2023, 2024]
FEATURES_PATH = Path("data/processed/features.parquet")
REPORT = Path(f"docs/reports/weak_points_{date.today()}.md")
KEY = ["player_id", "season", "week"]


def load() -> pl.DataFrame:
    preds = pl.concat([pl.read_parquet(f"data/processed/weekly_predictions_{s}.parquet") for s in SEASONS])
    f = pl.read_parquet(FEATURES_PATH).select(
        *KEY, "years_exp", "player_new_team", "depth_rank", "report_status", "games_std", "implied_total",
        "team_spread", "offense_pct_l3", "fantasy_points_ppr_l3", "fantasy_points_ppr_std", "left_injured")
    espn = pl.concat([pl.read_parquet(p).with_columns(season=pl.lit(int(p.stem.split("_")[2]), pl.Int32),
                                                     week=pl.lit(int(p.stem.split("wk")[1]), pl.Int32))
                      for p in sorted(CACHE.glob("espn_proj_20*_wk*.parquet")) if int(p.stem.split("_")[2]) in SEASONS])
    espn = espn.drop_nulls("player_id").unique(KEY).select(*KEY, "espn_proj")
    d = preds.join(f, on=KEY, how="left").join(espn, on=KEY, how="left").with_columns(
        err=pl.col("pred") - pl.col("actual"), abs_err=(pl.col("pred") - pl.col("actual")).abs())
    return d.with_columns(
        week_group=pl.when(pl.col("week") <= 3).then(pl.lit("weeks 1-3")).when(pl.col("week") <= 9)
        .then(pl.lit("weeks 4-9")).otherwise(pl.lit("weeks 10-18")),
        experience=pl.when(pl.col("years_exp") == 0).then(pl.lit("rookie")).when(pl.col("years_exp") <= 2)
        .then(pl.lit("2-3rd year")).otherwise(pl.lit("veteran")),
        new_team=pl.col("player_new_team").fill_null(0).cast(pl.Boolean),
        depth=pl.when(pl.col("depth_rank").is_null()).then(pl.lit("off chart"))
        .when(pl.col("depth_rank") == 1).then(pl.lit("starter")).otherwise(pl.lit("backup")),
        injury=pl.col("report_status").fill_null("none"),
        game_script=pl.when(pl.col("team_spread") >= 7).then(pl.lit("big favorite"))
        .when(pl.col("team_spread") <= -7).then(pl.lit("big underdog")).otherwise(pl.lit("close")),
        sample=pl.when(pl.col("games_std") <= 2).then(pl.lit("0-2 games this season")).otherwise(pl.lit("3+ games")),
        tier=pl.col("pred").qcut([0.5, 0.8, 0.95], labels=["bottom 50%", "50-80%", "80-95%", "top 5%"]).over("position"),
    )


def segments(d: pl.DataFrame) -> list[str]:
    base = d.group_by("position").agg(base_mae=pl.col("abs_err").mean())
    lines = ["## 1. Segments: where errors are bigger than usual (MAE relative to that position's average)", "",
             "| segment | value | player-weeks | MAE vs position avg | bias (pred - actual) |", "| --- | --- | --- | --- | --- |"]
    rows = []
    for seg in ["week_group", "experience", "new_team", "depth", "injury", "game_script", "sample", "tier"]:
        g = d.join(base, on="position").group_by(seg).agg(
            n=pl.len(), rel=(pl.col("abs_err") / pl.col("base_mae")).mean(), bias=pl.col("err").mean())
        for r in g.sort(seg).iter_rows(named=True):
            if r["n"] >= 200:
                rows.append((seg, str(r[seg]), r["n"], r["rel"], r["bias"]))
    for seg, val, n, rel, bias in rows:
        flag = " **" if abs(bias) >= 0.75 or rel >= 1.15 else ""
        lines.append(f"| {seg} | {val}{flag} | {n} | {rel:.2f}x | {bias:+.2f} |")
    return lines + ["", "** = bias of 0.75+ points or 15%+ larger errors than usual.", ""]


def calibration(d: pl.DataFrame) -> list[str]:
    lines = ["## 3. Calibration by projection tier (mean projection vs mean actual)", "",
             "| position | tier | n | mean projection | mean actual | gap (actual - proj) | ESPN mean (same players) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    g = d.group_by("position", "tier").agg(n=pl.len(), proj=pl.col("pred").mean(), act=pl.col("actual").mean(),
                                           espn=pl.col("espn_proj").mean()).sort("position", "tier")
    for r in g.iter_rows(named=True):
        lines.append(f"| {r['position']} | {r['tier']} | {r['n']} | {r['proj']:.2f} | {r['act']:.2f} | "
                     f"{r['act'] - r['proj']:+.2f} | {r['espn'] if r['espn'] is None else round(r['espn'], 2)} |")
    return lines + [""]


def vs_espn(d: pl.DataFrame) -> list[str]:
    e = d.drop_nulls("espn_proj").with_columns(espn_abs=(pl.col("espn_proj") - pl.col("actual")).abs())
    lines = ["## 4. Where ESPN is more accurate than us (players both projected, MAE ours - ESPN; + = ESPN better)", "",
             "| segment | value | player-weeks | ours | ESPN | gap |", "| --- | --- | --- | --- | --- | --- |"]
    for seg in ["position", "week_group", "experience", "new_team", "depth", "injury", "game_script", "sample", "tier"]:
        g = e.group_by(seg).agg(n=pl.len(), ours=pl.col("abs_err").mean(), espn=pl.col("espn_abs").mean())
        for r in g.sort(seg).iter_rows(named=True):
            if r["n"] >= 200:
                gap = r["ours"] - r["espn"]
                flag = " **" if gap >= 0.15 else ""
                lines.append(f"| {seg} | {r[seg]}{flag} | {r['n']} | {r['ours']:.2f} | {r['espn']:.2f} | {gap:+.2f} |")
    return lines + ["", "** = ESPN at least 0.15 points/player-week more accurate.", ""]


def components() -> list[str]:
    """One fit on 2020-2023, predict 2024 stat by stat; points error split by stat (PPR weights)."""
    feats = pl.read_parquet(FEATURES_PATH)
    train = feats.filter(pl.col("season").is_between(2020, 2023))
    test = feats.filter((pl.col("season") == 2024) & ~pl.col("left_injured"))
    model = BoostingModel(tuned=TUNED_PATH).fit(train)
    preds = []
    for (week,), rows in test.group_by("week"):
        p = model.predict(rows)
        preds.append(rows.select(*KEY, "position").join(p, on="player_id", how="left"))
    p = pl.concat(preds)
    t = test.select(*KEY, *[c for c in test.columns if c.startswith("y_")])
    j = p.join(t, on=KEY)
    weights = {
        "TDs (6/td, 4/pass td)": [("passing_tds", SCORING.passing_tds), ("rushing_tds", 6), ("receiving_tds", 6)],
        "yards": [("passing_yards", 0.04), ("rushing_yards", 0.1), ("receiving_yards", 0.1)],
        "receptions (PPR)": [("receptions", 1.0)],
        "interceptions/fumbles": [("passing_interceptions", -2), ("fumbles_lost_total", -2)],
    }
    lines = ["## 2. Which stat drives the points error (2024, one model fit on 2020-23)", "",
             "Share of each position's points-error variance from each stat group (errors in points).", "",
             "| position | " + " | ".join(weights) + " |", "| --- |" + " --- |" * len(weights)]
    for pos in ["QB", "RB", "WR", "TE"]:
        g = j.filter(pl.col("position") == pos)
        parts = {}
        for name, comps in weights.items():
            err = np.zeros(g.height)
            for stat, w in comps:
                if stat in g.columns and f"y_{stat}" in g.columns:
                    err += w * (g[stat].fill_null(0).to_numpy() - g[f"y_{stat}"].fill_null(0).to_numpy())
            parts[name] = err.var()
        total = sum(parts.values())
        lines.append(f"| {pos} | " + " | ".join(f"{v / total:.0%}" for v in parts.values()) + " |")
    return lines + ["", "(Shares of summed variance; cross-terms ignored, so read as rough proportions.)", ""]


def main() -> None:
    d = load()
    lines = [f"# Weak points of the weekly model ({date.today()})", "", __doc__.split("\n\n")[1], "",
             f"Player-weeks: {d.height}. Overall MAE {d['abs_err'].mean():.2f}, bias {d['err'].mean():+.2f}.", ""]
    lines += segments(d) + components() + calibration(d) + vs_espn(d)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
