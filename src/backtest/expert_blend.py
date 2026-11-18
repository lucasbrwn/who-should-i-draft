"""Model + experts blend: does averaging our weekly number with ESPN and FantasyPros beat the experts?

Why (2026-11-18): every play-by-play feature since 11/14 has tied; the model looks capped by
information, not by the algorithm. On the draft side, feeding the market (ADP) into our season model
turned a loss into a +58 pts/season win (SCRUM-93). This is the weekly version. Lucas chose to test it
knowing the claim becomes "adds value on top of the experts", not "beats them alone".

Rules (LOCKED 2026-11-18 before any blend result was computed):
    ours      expected points = walk-forward boosting projection x chance he plays (what the product shows)
    ESPN      ESPN's weekly PPR projection (points)
    FP        FantasyPros weekly ECR (Friday snapshot) turned into points: per position, a decreasing
              isotonic fit of actual PPR points on ECR rank, fit on the OTHER test seasons only
              (leave-one-season-out, so a season's own outcomes never set its map). Players whose game
              was before the snapshot (Thursday) get no FP value - the experts may have seen the result.
    blend     plain average of whichever of the three exist. EQUAL WEIGHTS, fixed in advance - no
              fitting, so nothing here can overfit (simple averages are hard to beat in forecasting).
    seasons   2022-2024 (2025 is the final exam). Same pools / pairs / bootstrap as vs_espn.py.

    PRIMARY   blend vs FantasyPros head-to-head (FP's top-24/48/60/24, pairs within 6 FP spots):
              PASS if the 95% CI of "right when we disagree" is entirely above 50%
    AND       blend vs ESPN head-to-head: not significantly worse (CI upper bound above 50%)
    -> PASS: the blend becomes the product's weekly number (our model stays visible next to it).
    -> FAIL: keep our model alone; report the result.

    CLAIM CHECK (decides the wording, not the keep): does OUR model add anything to the experts?
              Paired MAE of the 3-way blend vs the experts-only blend (ESPN + FP), same player-weeks,
              player-level bootstrap. Only if that CI is entirely below 0 may we say "our model adds
              value on top of the experts"; otherwise the honest wording is "averaging sources helps".
    REPORTED  (not decisive) ours+ESPN, ours+FP, experts-only head-to-heads; MAE of every version.

Known bias: ESPN's historical number is its FINAL projection, which can include news after Friday;
live we snapshot ESPN Thursday 2 PM / Saturday 10 AM. So the backtest may flatter any blend using ESPN.

    python -m src.backtest.expert_blend
"""

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from src.backtest.vs_consensus import N_BOOT, bootstrap_rate, score_pairs
from src.backtest.vs_espn import espn_history, our_predictions, pool, weekly_pairs
from src.backtest.weekly_vs_consensus import POOL, game_days, weekly_ecr

SEASONS = [2022, 2023, 2024]
REPORT_DIR = Path("docs/reports")
BLENDS = {"blend": ["expected", "espn_proj", "fp_pts"],          # the one under test
          "ours+ESPN": ["expected", "espn_proj"],
          "ours+FP": ["expected", "fp_pts"],
          "experts only": ["espn_proj", "fp_pts"],
          "ours alone": ["expected"]}


def fp_ranks(seasons: list[int]) -> pl.DataFrame:
    """FP ECR rank for players whose game came AFTER the snapshot."""
    days = game_days(seasons)
    return (weekly_ecr(seasons).join(days, on=["player_id", "season", "week"], how="inner")
            .filter(pl.col("gameday") > pl.col("d")).select("player_id", "season", "week", fp_rank="rank"))


def fp_points(frame: pl.DataFrame) -> pl.DataFrame:
    """Add fp_pts: rank -> points via isotonic fits on the other seasons (needs fp_rank, actual, season, position)."""
    out = []
    for season in frame["season"].unique().to_list():
        rows = frame.filter(pl.col("season") == season)
        fits = {}
        for pos in POOL:
            train = frame.filter((pl.col("season") != season) & (pl.col("position") == pos)
                                 & pl.col("fp_rank").is_not_null())
            if train.is_empty():
                continue
            fits[pos] = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(
                train["fp_rank"].to_numpy(), train["actual"].to_numpy())
        pts = np.full(rows.height, np.nan)
        for pos, iso in fits.items():
            m = ((rows["position"] == pos) & rows["fp_rank"].is_not_null()).to_numpy()
            if m.any():
                pts[m] = iso.predict(rows.filter(pl.Series(m))["fp_rank"].to_numpy())
        out.append(rows.with_columns(fp_pts=pl.Series(pts).fill_nan(None)))
    return pl.concat(out)


def add_blends(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns([pl.mean_horizontal(cols).alias(name) for name, cols in BLENDS.items()])


def head_to_head(frame: pl.DataFrame, market_rank: str, ours: str) -> tuple[int, float, float, float]:
    f = frame.filter(pl.col(market_rank).is_not_null()).with_columns(rank=pl.col(market_rank))
    dis = score_pairs(weekly_pairs(pool(f)), "actual", ours=ours).filter("disagree")
    return (dis.height, *bootstrap_rate(dis, "we_right"))


def paired_mae_diff(frame: pl.DataFrame, a: str, b: str, seed: int = 0) -> tuple[float, float, float]:
    """mean(|a-actual| - |b-actual|) with a player-level bootstrap CI (negative = a better)."""
    d = frame.select("player_id", "season", diff=(pl.col(a) - pl.col("actual")).abs() - (pl.col(b) - pl.col("actual")).abs())
    per = d.group_by("player_id", "season").agg(s=pl.col("diff").sum(), n=pl.len())
    s, n = per["s"].to_numpy(), per["n"].to_numpy()
    rng = np.random.default_rng(seed)
    boots = [(s[i].sum() / n[i].sum()) for i in (rng.integers(0, len(s), len(s)) for _ in range(N_BOOT))]
    return float(s.sum() / n.sum()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def build(seasons: list[int]) -> pl.DataFrame:
    if 2025 in seasons:
        raise ValueError("2025 is the final exam")
    frames = []
    for season in seasons:
        ours = our_predictions(season).select("player_id", "season", "week", "position", "actual", "expected")
        e = espn_history(season).select("player_id", "season", "week", "espn_proj")
        frames.append(ours.join(e, on=["player_id", "season", "week"], how="left"))
        print(f"{season}: loaded", flush=True)
    frame = pl.concat(frames).join(fp_ranks(seasons), on=["player_id", "season", "week"], how="left")
    frame = fp_points(frame)
    return add_blends(frame).with_columns(
        espn_rank=pl.col("espn_proj").rank("ordinal", descending=True).over("season", "week", "position"))


def main() -> None:
    frame = build(SEASONS)
    lines = [f"# Model + experts blend, seasons {SEASONS} ({date.today()})", "",
             __doc__.split("\n\n")[2], "",
             "## Head-to-head: when the version disagrees with the benchmark, how often it's right", "",
             "| version | vs FantasyPros: disagreements | right [95% CI] | vs ESPN: disagreements | right [95% CI] |",
             "| --- | --- | --- | --- | --- |"]
    res = {}
    for name in BLENDS:
        fp = head_to_head(frame, "fp_rank", name)
        es = head_to_head(frame, "espn_rank", name)
        res[name] = (fp, es)
        lines.append(f"| {name} | {fp[0]} | **{fp[1]:.1%}** [{fp[2]:.1%}, {fp[3]:.1%}] "
                     f"| {es[0]} | **{es[1]:.1%}** [{es[2]:.1%}, {es[3]:.1%}] |")
    # MAE on player-weeks that have all three numbers
    full = frame.drop_nulls(["expected", "espn_proj", "fp_pts", "actual"])
    lines += ["", f"## Points error (MAE) on {full.height:,} player-weeks with all three numbers", "",
              "| version | MAE |", "| --- | --- |"]
    lines += [f"| {c} | {(full[c] - full['actual']).abs().mean():.3f} |"
              for c in ["expected", "espn_proj", "fp_pts", *BLENDS]]
    d, lo, hi = paired_mae_diff(full, "blend", "experts only")
    (_, fw, flo, fhi), (_, ew, elo, ehi) = res["blend"]
    passed = flo > 0.5 and ehi > 0.5
    adds = hi < 0
    lines += ["", "## Verdict (rules locked before the run)", "",
              f"- PRIMARY blend vs FantasyPros: {fw:.1%} [{flo:.1%}, {fhi:.1%}] -> {'CI above 50%' if flo > 0.5 else 'CI includes/below 50%'}",
              f"- blend vs ESPN: {ew:.1%} [{elo:.1%}, {ehi:.1%}] -> {'not significantly worse' if ehi > 0.5 else 'SIGNIFICANTLY WORSE'}",
              f"- **{'PASS' if passed else 'FAIL'}**",
              f"- CLAIM CHECK blend minus experts-only MAE: {d:+.3f} [{lo:+.3f}, {hi:+.3f}] -> "
              + ("our model adds value on top of the experts" if adds else "NOT shown that our model adds value; wording = 'averaging sources helps'")]
    path = REPORT_DIR / f"expert_blend_{date.today()}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
