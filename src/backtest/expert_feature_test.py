"""Challenger vs champion: our model WITH the FantasyPros rank as an input, vs the 1/3-1/3-1/3 blend.

Rules (LOCKED 2026-11-18 before the challenger's results were computed; Lucas: "Do it"):
    challenger  walk-forward boosting on our features + fp_ecr (src/features/expert.py), each week trained
                only on earlier weeks; shown as expected points = projection x chance he plays
    champion    the blend that passed on 2026-11-18 (src/backtest/expert_blend.py)
    seasons     2022-2024; 2025 untouched. Same pools, pairs, bootstrap as the blend test.

    1. The challenger must pass the blend's bar: vs FantasyPros head-to-head CI entirely above 50%,
       AND vs ESPN not significantly worse (CI upper bound above 50%).
    2. If both pass: the lower points error (MAE, same player-weeks with all sources) wins IF the paired
       difference is significant (player-level bootstrap CI excludes 0).
    3. If they tie on MAE: the challenger wins (Lucas prefers one model we can call ours).
    4. If the challenger fails 1: the blend stays.

    python -m src.backtest.expert_feature_test
"""

from datetime import date
from pathlib import Path

import polars as pl

from src.backtest.expert_blend import SEASONS, build, head_to_head, paired_mae_diff
from src.backtest.harness import run
from src.features.expert import build_expert_features
from src.features.team import read_schedules
from src.models.boosting import TUNED_PATH, BoostingModel
from src.models.play_chance import PlayChance, game_day_pool

FEATURES_PATH = Path("data/processed/features.parquet")
CACHE = Path("data/processed")
REPORT_DIR = Path("docs/reports")
RATE_POOL = [2021, 2022, 2023, 2024]


def challenger_predictions(season: int) -> pl.DataFrame:
    path = CACHE / f"weekly_predictions_fpinput_{season}.parquet"
    if not path.exists():
        feats = pl.read_parquet(FEATURES_PATH).filter(pl.col("season") <= season)
        feats = feats.join(build_expert_features(sorted(feats["season"].unique().to_list())),
                           on=["player_id", "season", "week"], how="left")
        assert "fp_ecr" in feats.columns and feats["fp_ecr"].is_not_null().sum() > 0, "fp_ecr missing"
        res = run(BoostingModel(tuned=TUNED_PATH, name="boosting_fpinput"), feats, [season])
        res.write_parquet(path)
    res = pl.read_parquet(path)
    info = pl.read_parquet(FEATURES_PATH).select("player_id", "season", "week", "depth_rank", "report_status",
                                                 "practice_code", "years_exp")
    res = res.join(info, on=["player_id", "season", "week"], how="left")
    chance = PlayChance().fit(game_day_pool([s for s in RATE_POOL if s != season], read_schedules()))
    return chance.rates(res).select("player_id", "season", "week", challenger=pl.col("pred") * pl.col("p_play"))


def main() -> None:
    frame = build(SEASONS)
    ch = pl.concat([challenger_predictions(s) for s in SEASONS])
    frame = frame.join(ch, on=["player_id", "season", "week"], how="left")
    lines = [f"# Challenger (our model + FantasyPros rank as input) vs champion (blend), {SEASONS} ({date.today()})", "",
             __doc__.split("\n\n")[1], "",
             "| version | vs FantasyPros: disagreements | right [95% CI] | vs ESPN: disagreements | right [95% CI] |",
             "| --- | --- | --- | --- | --- |"]
    res = {}
    for name in ["challenger", "blend", "ours alone"]:
        fp, es = head_to_head(frame, "fp_rank", name), head_to_head(frame, "espn_rank", name)
        res[name] = (fp, es)
        lines.append(f"| {name} | {fp[0]} | **{fp[1]:.1%}** [{fp[2]:.1%}, {fp[3]:.1%}] "
                     f"| {es[0]} | **{es[1]:.1%}** [{es[2]:.1%}, {es[3]:.1%}] |")
    full = frame.drop_nulls(["expected", "espn_proj", "fp_pts", "actual", "challenger"])
    mae = {c: (full[c] - full["actual"]).abs().mean() for c in ["challenger", "blend", "expected", "espn_proj"]}
    d, lo, hi = paired_mae_diff(full, "challenger", "blend")
    (_, _, flo, _), (_, _, _, ehi) = res["challenger"]
    passes = flo > 0.5 and ehi > 0.5
    if not passes:
        verdict = "challenger FAILS the bar -> blend stays"
    elif hi < 0:
        verdict = "challenger passes and has significantly lower MAE -> CHALLENGER wins"
    elif lo > 0:
        verdict = "challenger passes but blend has significantly lower MAE -> BLEND stays"
    else:
        verdict = "challenger passes, MAE tie -> CHALLENGER wins (rule 3: one model we call ours)"
    lines += ["", f"MAE on {full.height:,} player-weeks with every source: "
              + ", ".join(f"{k} {v:.3f}" for k, v in mae.items()), "",
              f"Challenger minus blend MAE: {d:+.3f} [{lo:+.3f}, {hi:+.3f}] (negative = challenger better)", "",
              f"**Verdict: {verdict}**"]
    path = REPORT_DIR / f"expert_feature_test_{date.today()}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
