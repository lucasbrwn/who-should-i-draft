"""One final model on top of ours + ESPN + FantasyPros ("stacking"), vs the 1/3 blend.

Why (2026-11-18, Lucas): he wants ONE model that uses the expert data as inputs, not an average of three
predictions, and he worried a weak source drags a strong one down. Putting the FP rank into the
per-stat models failed (it got ignored: one of 185 inputs). A second-stage model sees only a handful
of inputs and can learn WHEN to trust whom (injury report -> ESPN? role change -> ours? rookie -> FP?).

Rules (LOCKED 2026-11-18 before any stacker result was computed; Lucas: "Yes"):
    inputs    our expected points, our raw projection, chance he plays, ESPN projection, FantasyPros
              ECR RANK (raw rank, not the rank->points map, which was fit on other seasons incl. later
              ones), position, injury report status, depth rank, years of experience. Missing = blank.
    model     sklearn HistGradientBoostingRegressor, settings fixed in advance (never tuned):
              max_depth 3, learning_rate 0.05, max_iter 300, min_samples_leaf 50, l2 1.0, seed 0
    training  week-by-week walk-forward: each test week is predicted by a stacker trained only on
              EARLIER weeks (from 2022 week 1). Test seasons 2023-2024 (2022 has too little history).
              2025 untouched.
    champion  the 1/3 blend on the same 2023-24 rows. (Its FP rank->points map was fit leave-one-season-
              out, so for 2023 it saw 2024 outcomes: a small edge FOR the champion, i.e. conservative.)
    judge     same pools / pairs / player bootstrap as expert_blend.py:
      1. stacker vs FantasyPros head-to-head CI entirely above 50%, AND vs ESPN not significantly worse
      2. stacker minus blend MAE (player-weeks with every source): if significantly WORSE (CI > 0) the
         blend stays; if tied or better, the STACKER becomes the product (Lucas prefers one model).
      If 1 fails: the blend stays.
Known bias: ESPN history is its FINAL number (may include post-Friday news), so the stacker may learn to
over-trust ESPN compared with our live Thursday snapshot.

    python -m src.backtest.stacker_test
"""

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from src.backtest.expert_blend import build, head_to_head, paired_mae_diff
from src.backtest.vs_espn import our_predictions

SEASONS = [2022, 2023, 2024]
TEST_SEASONS = [2023, 2024]
REPORT_DIR = Path("docs/reports")
POSITIONS = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
STATUS = {"Questionable": 1, "Doubtful": 2, "Out": 3}
INPUTS = ["expected", "pred", "p_play", "espn_proj", "fp_rank", "pos_code", "status_code", "depth_rank", "years_exp"]
PARAMS = dict(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=50, l2_regularization=1.0, random_state=0)


def stacker_frame() -> pl.DataFrame:
    frame = build(SEASONS)   # expected, espn_proj, fp_rank, fp_pts, blend, espn_rank, actual ...
    extra = pl.concat([our_predictions(s).select("player_id", "season", "week", "pred", "p_play", "report_status",
                                                 "depth_rank", "years_exp") for s in SEASONS])
    return frame.join(extra, on=["player_id", "season", "week"], how="left").with_columns(
        pos_code=pl.col("position").replace_strict(POSITIONS, default=None).cast(pl.Float64),
        status_code=pl.col("report_status").replace_strict(STATUS, default=0).cast(pl.Float64),
        t=pl.col("season") * 100 + pl.col("week"),
    )


def walk_forward(frame: pl.DataFrame) -> pl.DataFrame:
    """Predict every 2023-24 week with a stacker trained only on earlier weeks."""
    frame = frame.filter(pl.col("actual").is_not_null())
    x = lambda df: df.select(INPUTS).cast(pl.Float64).to_numpy()  # noqa: E731
    out = []
    for t in sorted(frame.filter(pl.col("season").is_in(TEST_SEASONS))["t"].unique().to_list()):
        train, test = frame.filter(pl.col("t") < t), frame.filter(pl.col("t") == t)
        model = HistGradientBoostingRegressor(**PARAMS).fit(x(train), train["actual"].to_numpy())
        out.append(test.select("player_id", "season", "week").with_columns(stacker=pl.Series(np.clip(model.predict(x(test)), 0, None))))
    return pl.concat(out)


def main() -> None:
    frame = stacker_frame()
    preds = walk_forward(frame)
    frame = frame.filter(pl.col("season").is_in(TEST_SEASONS)).join(preds, on=["player_id", "season", "week"], how="left")
    lines = [f"# Stacker (one final model on ours + ESPN + FantasyPros) vs the 1/3 blend, {TEST_SEASONS} ({date.today()})", "",
             __doc__.split("\n\n")[2], "",
             "| version | vs FantasyPros: disagreements | right [95% CI] | vs ESPN: disagreements | right [95% CI] |",
             "| --- | --- | --- | --- | --- |"]
    res = {}
    for name in ["stacker", "blend", "experts only", "ours alone"]:
        fp, es = head_to_head(frame, "fp_rank", name), head_to_head(frame, "espn_rank", name)
        res[name] = (fp, es)
        lines.append(f"| {name} | {fp[0]} | **{fp[1]:.1%}** [{fp[2]:.1%}, {fp[3]:.1%}] "
                     f"| {es[0]} | **{es[1]:.1%}** [{es[2]:.1%}, {es[3]:.1%}] |")
    full = frame.drop_nulls(["expected", "espn_proj", "fp_pts", "actual", "stacker"])
    mae = {c: (full[c] - full["actual"]).abs().mean() for c in ["stacker", "blend", "experts only", "expected", "espn_proj"]}
    d, lo, hi = paired_mae_diff(full, "stacker", "blend")
    (_, _, flo, _), (_, _, _, ehi) = res["stacker"]
    if not (flo > 0.5 and ehi > 0.5):
        verdict = "stacker FAILS the head-to-head bar -> blend stays"
    elif lo > 0:
        verdict = "stacker passes but has significantly WORSE MAE than the blend -> blend stays"
    elif hi < 0:
        verdict = "stacker passes and has significantly BETTER MAE -> STACKER becomes the product"
    else:
        verdict = "stacker passes, MAE tie with the blend -> STACKER becomes the product (one model, Lucas's preference)"
    lines += ["", f"MAE on {full.height:,} player-weeks with every source: "
              + ", ".join(f"{k} {v:.3f}" for k, v in mae.items()), "",
              f"Stacker minus blend MAE: {d:+.3f} [{lo:+.3f}, {hi:+.3f}] (negative = stacker better)", "",
              f"**Verdict: {verdict}**"]
    path = REPORT_DIR / f"stacker_test_{date.today()}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
