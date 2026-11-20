"""v2 item 3: train for ORDER (LightGBM LambdaRank) instead of point accuracy, then put real points back.

Why (2026-11-19): start/sit is a ranking question ("A or B this week?"), but every model so far learns
to get each stat's number close (squared error / Poisson). A ranker is penalized only for putting
players in the wrong order inside a position-week, which is exactly what the expert head-to-head grades.

A ranker's output has no units, so it can't be a projection (no MAE, no blend, no VOR). RANK REMAP:
inside each position-week, keep our current point values but hand them out in the ranker's order
(ranker's #1 gets our highest projection, and so on). Totals stay realistic; only the order changes.

Rules (LOCKED 2026-11-19 before any ranker result was computed; Lucas: no changes to the proposal):
    ranker    LightGBM lambdarank, one per position, same feature columns as the boosting model,
              groups = position-week, label = actual PPR points rounded and clipped to 0..40 with
              LINEAR gain (a 30-pt game matters 30x a 1-pt game, not 2^30x). PPR only (known
              mismatch for half/standard leagues - accepted to save 3x training).
              Settings fixed in advance, never tuned: n_estimators 300, learning_rate 0.05,
              num_leaves 31, min_child_samples 50, reg_lambda 0.1, seed 0.
    training  walk-forward like the boosting model: every week uses only earlier weeks, refit every
              4 weeks and at each season start; games a player left injured are not trained on.
    seasons   test 2022-2024 (same as the blend test). 2025 is "seen" (final exam) - not used.
    remap     expected_rank = (our point projection reassigned in the ranker's order) x chance he plays
    champion  the current product: 1/3 blend of ours + ESPN + FantasyPros
    KEEP the ranker only if BOTH:
      1. blend(ranker) vs FantasyPros "right when we disagree" is higher than blend(current) AND the
         two 95% CIs do not overlap
      2. blend(ranker) MAE is not worse than blend(current) by more than 0.02 points
    Otherwise (including any tie) the simpler current model stays.

    ~/.venvs/whoshouldidraft/Scripts/python -m src.backtest.ranker_test
"""

from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from src.backtest.expert_blend import build, head_to_head
from src.backtest.harness import actual_stats, points
from src.backtest.vs_espn import FEATURES_PATH, our_predictions
from src.models.boosting import usable_columns

SEASONS = [2022, 2023, 2024]
POSITIONS = ["QB", "RB", "WR", "TE"]
REFIT_WEEKS = 4
MAX_LABEL = 40
PARAMS = dict(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
              reg_lambda=0.1, random_state=0, verbose=-1, label_gain=list(range(MAX_LABEL + 1)))
CACHE = Path("data/processed")
REPORT_DIR = Path("docs/reports")
KEY = ["player_id", "season", "week"]


def rank_remap(frame: pl.DataFrame, score: str, values: str, out: str,
               group: tuple[str, ...] = ("season", "week", "position")) -> pl.DataFrame:
    """Inside each group (default season-week-position), give the k-th highest `score` the k-th highest `values`."""
    group = list(group)
    by_score = pl.col(score).rank("ordinal", descending=True).over(group)
    by_value = pl.col(values).rank("ordinal", descending=True).over(group)
    lookup = frame.select(*group, k=by_value, **{out: pl.col(values)})
    return frame.with_columns(k=by_score).join(lookup, on=[*group, "k"], how="left").drop("k")


def _x(df: pl.DataFrame, cols: list[str], cats: dict[str, list[str]]) -> pd.DataFrame:
    text = [c for c in cols if df.schema[c] == pl.String]
    x = df.select([pl.col(c).cast(pl.Float64) for c in cols if c not in text] + text).to_pandas()
    for c in text:
        cats.setdefault(c, sorted(df[c].drop_nulls().unique()))
        x[c] = pd.Categorical(x[c].where(x[c].isin(cats[c])), categories=cats[c])
    return x


def fit_rankers(train: pl.DataFrame) -> dict:
    """One LambdaRank model per position; rows sorted so each position-week group is contiguous."""
    train = train.filter(~pl.col("left_injured"))
    labels = points(actual_stats(train)).round().clip(0, MAX_LABEL).cast(pl.Int32)
    models = {}
    for pos in POSITIONS:
        # pick feature columns BEFORE the label exists, so the answer can never become an input
        cols, cats = usable_columns(train.filter(pl.col("position") == pos)), {}
        assert not any(c == "label" or c.startswith("y_") for c in cols), "target leaked into features"
        rows = train.with_columns(label=labels).filter(pl.col("position") == pos).sort("season", "week")
        groups = rows.group_by("season", "week", maintain_order=True).len()["len"].to_numpy()
        model = lgb.LGBMRanker(**PARAMS).fit(_x(rows, cols, cats), rows["label"].to_numpy(), group=groups)
        models[pos] = (model, cols, cats)
    return models


def ranker_scores(season: int, feats: pl.DataFrame) -> pl.DataFrame:
    path = CACHE / f"ranker_scores_{season}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    out, models = [], None
    for week in sorted(feats.filter(pl.col("season") == season)["week"].unique().to_list()):
        if models is None or (week - 1) % REFIT_WEEKS == 0:
            before = (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week))
            models = fit_rankers(feats.filter(before))
            print(f"{season} wk{week}: refit", flush=True)
        rows = feats.filter((pl.col("season") == season) & (pl.col("week") == week))
        for pos, (model, cols, cats) in models.items():
            r = rows.filter(pl.col("position") == pos)
            if r.height:
                out.append(r.select(KEY).with_columns(rank_score=pl.Series(model.predict(_x(r, cols, cats)))))
    res = pl.concat(out)
    res.write_parquet(path)
    return res


def main() -> None:
    feats = pl.read_parquet(FEATURES_PATH).filter(pl.col("season") <= max(SEASONS))
    scores = pl.concat([ranker_scores(s, feats) for s in SEASONS])
    ours = pl.concat([our_predictions(s).select(*KEY, "position", "pred", "p_play") for s in SEASONS])
    ours = rank_remap(ours.join(scores, on=KEY, how="left").filter(pl.col("rank_score").is_not_null()),
                      "rank_score", "pred", "pred_rank").with_columns(expected_rank=pl.col("pred_rank") * pl.col("p_play"))
    frame = build(SEASONS).join(ours.select(*KEY, "expected_rank"), on=KEY, how="left").with_columns(
        blend_rank=pl.mean_horizontal("expected_rank", "espn_proj", "fp_pts"))

    lines = [f"# Ranking objective (LambdaRank + rank remap) vs current model, {SEASONS} ({date.today()})", "",
             __doc__.split("\n\n")[3], "",
             "| version | vs FantasyPros: disagreements | right [95% CI] | vs ESPN: disagreements | right [95% CI] |",
             "| --- | --- | --- | --- | --- |"]
    res = {}
    for name in ["blend_rank", "blend", "expected_rank", "expected"]:
        fp, es = head_to_head(frame, "fp_rank", name), head_to_head(frame, "espn_rank", name)
        res[name] = fp
        lines.append(f"| {name} | {fp[0]} | **{fp[1]:.1%}** [{fp[2]:.1%}, {fp[3]:.1%}] "
                     f"| {es[0]} | **{es[1]:.1%}** [{es[2]:.1%}, {es[3]:.1%}] |")
    full = frame.drop_nulls(["expected", "expected_rank", "espn_proj", "fp_pts", "actual"])
    mae = {c: (full[c] - full["actual"]).abs().mean() for c in ["blend_rank", "blend", "expected_rank", "expected"]}
    (_, nw, nlo, _), (_, ow, _, ohi) = res["blend_rank"], res["blend"]
    rule1 = nw > ow and nlo > ohi
    rule2 = mae["blend_rank"] - mae["blend"] <= 0.02
    verdict = "KEEP the ranker" if rule1 and rule2 else "REJECT - current model stays"
    lines += ["", f"MAE on {full.height:,} player-weeks with every source: " + ", ".join(f"{k} {v:.3f}" for k, v in mae.items()), "",
              f"- Rule 1 (higher win rate, CIs don't overlap): {'PASS' if rule1 else 'FAIL'}",
              f"- Rule 2 (blend MAE not worse by > 0.02): {'PASS' if rule2 else 'FAIL'} ({mae['blend_rank'] - mae['blend']:+.3f})",
              f"- **Verdict: {verdict}**"]
    path = REPORT_DIR / f"ranker_test_{date.today()}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
