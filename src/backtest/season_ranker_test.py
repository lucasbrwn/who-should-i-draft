"""v2 item 3b: a SEASON ranker for the draft board (Lucas, 2026-11-19: "the ranker should project who
will have the most fantasy points by end of season").

Why: the draft board is purely an ORDER problem, and it is where we lose to the experts (final exam:
board vs ECR close pairs 46.9%). The weekly ranker (src/backtest/ranker_test.py) was rejected the same
day; this tests the idea where it fits best.

Rules (LOCKED 2026-11-19 before any season-ranker result was computed; Lucas: "Go."):
    ranker    LightGBM lambdarank on the season model's own inputs (b_inputs: last season's stats, age,
              team changes, vacated/arrived opportunity, win totals, ADP, option A's per-game number...),
              groups = season x position, label = actual season PPR points rounded, clipped 0..499,
              LINEAR gain. Settings fixed in advance, never tuned (mirror B_PARAMS): n_estimators 200,
              learning_rate 0.05, num_leaves 15, min_child_samples 20, reg_lambda 1.0, seed 0,
              lambdarank_truncation_level 100 (draft pools go 72 RBs / 96 WRs deep, default is 30).
    training  walk-forward: the 2022 board learns from 2021 only, 2023 from 2021-22, 2024 from 2021-23.
              2025 is "seen" (final exam) and not used.
    remap     inside each season-position, the season model's point projections (season_b) are handed
              out in the ranker's order, so VOR / the recommender / every scoring format still get points.
    champion  the current board (season model B), rebuilt here the same way into a scratch folder.
    KEEP the season ranker only if BOTH:
      1. close pairs vs ECR (same position, within 12 ECR spots, src/backtest/vs_consensus.py):
         ranker board's "right when we disagree" is higher than the current board's AND the two 95%
         CIs do not overlap
      2. mock draft (only run if 1 passes): sim drafter's edge with the ranker board minus with the
         current board, paired on the same drafts - REJECT if the 95% CI is entirely below 0
    Otherwise (including any tie) the current board stays.

    python -m src.backtest.season_ranker_test           # boards + rule 1
    python -m src.backtest.season_ranker_test --mock    # rule 2, after the two mock runs (see printout)
"""

import argparse
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

import src.backtest.vs_consensus as vc
from src.backtest.harness import bootstrap
from src.backtest.ranker_test import rank_remap
from src.models.season import POSITIONS, SeasonModel, b_inputs, board, season_table

EVAL_SEASONS = [2022, 2023, 2024]
MAX_LABEL = 499
PARAMS = dict(objective="lambdarank", n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20,
              reg_lambda=1.0, random_state=0, verbose=-1, lambdarank_truncation_level=100,
              label_gain=list(range(MAX_LABEL + 1)))
SCRATCH = Path("data/processed/season_projections/ranker_test")
DIRS = {"current": SCRATCH / "current", "ranker": SCRATCH / "ranker"}
REPORT_DIR = Path("docs/reports")


class SeasonRanker:
    """LambdaRank on season rows; one group per season x position."""

    def fit(self, train: pl.DataFrame) -> "SeasonRanker":
        self.cols = [c for c in b_inputs(train) if train[c].null_count() < train.height]
        assert "actual" not in self.cols, "target leaked into features"
        train = train.sort("season", "position")
        groups = train.group_by("season", "position", maintain_order=True).len()["len"].to_numpy()
        label = train["actual"].round().clip(0, MAX_LABEL).cast(pl.Int32).to_numpy()
        self.model = lgb.LGBMRanker(**PARAMS).fit(self._x(train), label, group=groups)
        return self

    def _x(self, df: pl.DataFrame) -> pd.DataFrame:
        x = df.select([pl.col(c).cast(pl.Float64) if c != "position" else pl.col(c) for c in self.cols]).to_pandas()
        x["position"] = pd.Categorical(x["position"], categories=POSITIONS)
        return x

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        return self.model.predict(self._x(df))


def build_boards() -> None:
    data = season_table()
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    for season in EVAL_SEASONS:
        train, test = data.filter(pl.col("season") < season), data.filter(pl.col("season") == season)
        test = test.with_columns(season_b=pl.Series(SeasonModel().fit(train).predict(test)),
                                 rank_score=pl.Series(SeasonRanker().fit(train).predict(test)))
        board(test).write_parquet(DIRS["current"] / f"board_{season}.parquet")
        remapped = rank_remap(test.with_columns(week=pl.lit(0)), "rank_score", "season_b", "season_rank",
                              group=("season", "position"))
        board(remapped.drop("week", "season_b").rename({"season_rank": "season_b"})).write_parquet(
            DIRS["ranker"] / f"board_{season}.parquet")
        moved = (remapped["season_rank"] != remapped["season_b"]).mean()
        print(f"{season}: boards written ({moved:.0%} of players got a different projection after the remap)", flush=True)


def rule1() -> list[str]:
    res, lines = {}, ["| board | close pairs | we disagree with ECR | **right when we disagree** [95% CI] | by position |",
                      "| --- | --- | --- | --- | --- |"]
    for name, d in DIRS.items():
        vc.OUT_DIR = d  # vs_consensus reads board_{season}.parquet from here
        rows, _, _ = vc.run("ecr", "season_b")
        a = rows[0]
        res[name] = a
        pos = ", ".join(f"{r['group']} {r['win_rate']:.0%}" for r in rows[1:5])
        lines.append(f"| {name} | {a['pairs']} | {a['disagree']} | **{a['win_rate']:.1%}** [{a['lo']:.1%}, {a['hi']:.1%}] | {pos} |")
    n, c = res["ranker"], res["current"]
    passed = n["win_rate"] > c["win_rate"] and n["lo"] > c["hi"]
    lines += ["", f"- Rule 1 (higher, CIs don't overlap): **{'PASS -> run the mock draft (rule 2)' if passed else 'FAIL -> REJECT, current board stays'}**"]
    return lines


def rule2() -> list[str]:
    a = pl.read_parquet(REPORT_DIR / "mock_draft_season_ranker.parquet").filter(pl.col("drafter") == "sim")
    b = pl.read_parquet(REPORT_DIR / "mock_draft_season_current.parquet").filter(pl.col("drafter") == "sim")
    j = a.join(b, on=["season", "slot", "rep"], suffix="_cur")
    diff = (j["edge"] - j["edge_cur"]).to_numpy()
    ci = bootstrap(diff)
    reject = ci.high < 0
    return [f"- Rule 2 sim edge, ranker board minus current board (same drafts): {ci}",
            f"- **Verdict: {'REJECT, current board stays' if reject else 'KEEP the season ranker'}**"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mock", action="store_true", help="score rule 2 from the two finished mock runs")
    args = ap.parse_args()
    path = REPORT_DIR / f"season_ranker_test_{date.today()}.md"
    if args.mock:
        text = path.read_text(encoding="utf-8") + "\n" + "\n".join(rule2()) + "\n"
    else:
        build_boards()
        text = "\n".join([f"# Season ranker (LambdaRank + rank remap) vs current draft board, {EVAL_SEASONS} ({date.today()})",
                          "", __doc__.split("\n\n")[2], "", *rule1()]) + "\n"
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
