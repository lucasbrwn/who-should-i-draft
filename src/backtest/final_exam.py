"""FINAL EXAM, part A: the weekly product on the untouched 2025 season. Runs ONCE (SCRUM-66).

Lucas, 2026-11-18: "Let's do the final exam now" (model frozen early, tag `model-freeze`).

Rules (LOCKED 2026-11-18, before any 2025 prediction or outcome was looked at):
    what      the weekly product exactly as it runs live (src/predict/week.py):
                proj_if_plays  boosting, each 2025 week trained only on EARLIER games (2020-24 + earlier
                               2025 weeks: the same walk-forward as live use within a season)
                calibrated     tier calibration fit on 2021-24 walk-forward predictions
                expected       calibrated x chance he plays (rates fit on 2021-24)
                blend          1/3 expected + 1/3 ESPN projection + 1/3 FantasyPros ECR -> points
                               (frozen table models/fp_rank_points.json, fit on 2022-24)
    data      ESPN 2025 weekly projections (cached by src/ingest/exam_benchmarks.py); FantasyPros 2025
              Friday ECR snapshots (weeks 2-17; Thursday games dropped). Same pools / pairs / player
              bootstrap as the development tests (src/backtest/expert_blend.py).
    graded    1. HEADLINE  blend vs FantasyPros, start/sit: "CONFIRMED" if the 95% CI is entirely > 50%
              2.           blend vs ESPN: "BEATS ESPN" if CI > 50%; "MATCHES" if CI includes 50%;
                           "LOSES" if CI < 50%
              3.           our model alone vs both (reported, no pass line)
              4.           MAE of blend / ours / ESPN / FantasyPros on player-weeks with every source
              5.           likely range (25-75) coverage, all players and projection >= 10
    reference 2022-24 development results (docs/reports/expert_blend_2026-11-17.md): blend vs FP 53.0%
              [50.9, 54.9], vs ESPN 52.9% [51.1, 54.6]; MAE blend 3.953, ours 4.099, ESPN 3.993.
              (Dev used uncalibrated "expected"; the exam grades the product, which is calibrated.
               Calibration never reorders players within a position.)
    safety    refuses to run if the results file exists, or if src/ / models/ differ from tag model-freeze.
              Results are reported as-is, good or bad, in the README and slides.

    python -m src.backtest.final_exam
"""

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from src.backtest.expert_blend import head_to_head, paired_mae_diff
from src.backtest.harness import run
from src.backtest.vs_espn import espn_history, pool
from src.backtest.weekly_vs_consensus import game_days, weekly_ecr
from src.features.team import read_schedules
from src.models.boosting import TUNED_PATH, BoostingModel
from src.models.expert_blend import load_map, rank_to_points
from src.models.play_chance import PlayChance, game_day_pool
from src.models.ranges import LIKELY, BucketRanges, load_backtest
from src.models.weekly_calibration import WeeklyCalibration, load_history

EXAM = 2025
HISTORY = [2021, 2022, 2023, 2024]
RESULTS = Path("docs/reports/final_exam_2025_weekly.md")
PREDICTIONS = Path("data/processed/final_exam_2025_weekly_predictions.parquet")
FEATURES_PATH = Path("data/processed/features.parquet")
TAG = "model-freeze"


def check_frozen() -> str:
    if RESULTS.exists():
        raise SystemExit(f"{RESULTS} exists: the final exam already ran. It runs once.")
    tag = subprocess.run(["git", "rev-parse", "--short", TAG], capture_output=True, text=True)
    if tag.returncode != 0:
        raise SystemExit(f"git tag {TAG} not found: freeze first (git tag {TAG})")
    diff = subprocess.run(["git", "diff", "--quiet", TAG, "--", "src", "models"])
    if diff.returncode != 0:
        raise SystemExit(f"src/ or models/ differ from tag {TAG}: the exam must run on the frozen code")
    return tag.stdout.strip()


def product_predictions() -> pl.DataFrame:
    feats = pl.read_parquet(FEATURES_PATH).filter(pl.col("season") <= EXAM)
    res = run(BoostingModel(tuned=TUNED_PATH, name="boosting"), feats, [EXAM], allow_final_exam=True)
    info = feats.filter(pl.col("season") == EXAM).select("player_id", "season", "week", "team", "depth_rank",
                                                         "report_status", "practice_code")
    res = res.join(info, on=["player_id", "season", "week"], how="left")
    cal = WeeklyCalibration().fit(load_history(HISTORY))
    res = res.with_columns(calibrated=pl.Series(cal.apply(res["pred"].to_numpy(), res["position"].to_numpy())))
    chance = PlayChance().fit(game_day_pool(HISTORY, read_schedules()))
    res = chance.rates(res).with_columns(expected=pl.col("calibrated") * pl.col("p_play"))
    ranges = BucketRanges(LIKELY).fit(load_backtest()).predict(res.with_columns(proj=pl.col("calibrated")))
    return ranges.select(pl.exclude("proj"))


def with_experts(res: pl.DataFrame) -> pl.DataFrame:
    espn = espn_history_exam()
    days = game_days([EXAM])
    fp = (weekly_ecr([EXAM]).join(days, on=["player_id", "season", "week"], how="inner")
          .filter(pl.col("gameday") > pl.col("d")).select("player_id", "season", "week", fp_rank="rank"))
    table = load_map()
    out = (res.join(espn, on=["player_id", "season", "week"], how="left")
           .join(fp, on=["player_id", "season", "week"], how="left"))
    out = out.with_columns(fp_pts=pl.struct("position", "fp_rank").map_elements(
        lambda r: rank_to_points(r["position"], r["fp_rank"], table), return_dtype=pl.Float64))
    return out.with_columns(
        blend=pl.mean_horizontal("expected", "espn_proj", "fp_pts"),
        **{"ours alone": pl.col("expected")},
        espn_rank=pl.col("espn_proj").rank("ordinal", descending=True).over("season", "week", "position"),
    )


def espn_history_exam() -> pl.DataFrame:
    """Cached 2025 ESPN projections (src/ingest/exam_benchmarks.py); same shape as vs_espn.espn_history."""
    out = []
    for week in range(1, 19):
        path = Path(f"data/raw/espn/espn_proj_{EXAM}_wk{week:02d}.parquet")
        out.append(pl.read_parquet(path).with_columns(week=pl.lit(week, pl.Int32)))
    return (pl.concat(out).drop_nulls("player_id").unique(["player_id", "week"])
            .with_columns(season=pl.lit(EXAM, pl.Int32)).select("player_id", "season", "week", "espn_proj"))


def verdict_fp(lo: float) -> str:
    return "CONFIRMED (CI above 50%)" if lo > 0.5 else "NOT CONFIRMED (CI includes or is below 50%)"


def verdict_espn(lo: float, hi: float) -> str:
    return "BEATS ESPN" if lo > 0.5 else ("LOSES TO ESPN" if hi < 0.5 else "MATCHES ESPN")


def main() -> None:
    commit = check_frozen()
    started = datetime.now(timezone.utc).isoformat()
    res = product_predictions()
    frame = with_experts(res)
    frame.write_parquet(PREDICTIONS)

    rows, h2h = [], {}
    for name in ["blend", "ours alone"]:
        fp, es = head_to_head(frame, "fp_rank", name), head_to_head(frame, "espn_rank", name)
        h2h[name] = (fp, es)
        rows.append(f"| {name} | {fp[0]} | **{fp[1]:.1%}** [{fp[2]:.1%}, {fp[3]:.1%}] "
                    f"| {es[0]} | **{es[1]:.1%}** [{es[2]:.1%}, {es[3]:.1%}] |")
    full = frame.drop_nulls(["expected", "espn_proj", "fp_pts", "actual"])
    mae = {c: (full[c] - full["actual"]).abs().mean() for c in ["blend", "expected", "espn_proj", "fp_pts"]}
    d, lo, hi = paired_mae_diff(full, "blend", "espn_proj")
    played = frame.filter(pl.col("actual").is_not_null())
    inside = lambda df: ((df["actual"] >= df["bust"]) & (df["actual"] <= df["ceiling"])).mean()  # noqa: E731
    (_, fw, flo, fhi), (_, ew, elo, ehi) = h2h["blend"]

    lines = [f"# FINAL EXAM 2025, part A: weekly product (run once, {started})", "",
             f"Frozen code: tag `{TAG}` = {commit}. 2025 never used for training before this run.", "",
             __doc__.split("\n\n")[2], "",
             "## Head-to-head: when the version disagrees with the benchmark, how often it's right", "",
             "| version | vs FantasyPros: disagreements | right [95% CI] | vs ESPN: disagreements | right [95% CI] |",
             "| --- | --- | --- | --- | --- |", *rows, "",
             f"## Points error (MAE), {full.height:,} player-weeks with every source", "",
             "| version | 2025 MAE |", "| --- | --- |",
             f"| blend | {mae['blend']:.3f} |", f"| ours (expected) | {mae['expected']:.3f} |",
             f"| ESPN | {mae['espn_proj']:.3f} |", f"| FantasyPros (rank -> pts) | {mae['fp_pts']:.3f} |", "",
             f"Blend minus ESPN MAE: {d:+.3f} [{lo:+.3f}, {hi:+.3f}] (negative = blend better)", "",
             "## Likely range (25-75) coverage (target about 50%)", "",
             f"- all players who played: {inside(played):.1%} ({played.height:,} player-weeks)",
             f"- projection >= 10: {inside(played.filter(pl.col('calibrated') >= 10)):.1%}", "",
             "## Verdict (rules locked before the run)", "",
             f"1. **Blend vs FantasyPros: {fw:.1%} [{flo:.1%}, {fhi:.1%}] -> {verdict_fp(flo)}**",
             f"2. **Blend vs ESPN: {ew:.1%} [{elo:.1%}, {ehi:.1%}] -> {verdict_espn(elo, ehi)}**",
             "", "Dev reference (2022-24): blend vs FP 53.0% [50.9, 54.9], vs ESPN 52.9% [51.1, 54.6]."]
    RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {RESULTS} and {PREDICTIONS}")


if __name__ == "__main__":
    main()
