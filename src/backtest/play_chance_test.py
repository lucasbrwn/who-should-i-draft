"""Does "chance he plays" improve live predictions? (rule set 2026-11-17, before results)

The normal backtest only scores players who played, so it can't see this. Here every week of 2024 is
predicted the way live predictions are: every QB/RB/WR/TE on the pre-game active roster (ACT + INA)
gets a row, and anyone who didn't play scores 0.

    raw       model projection (= points if he plays)
    expected  raw x chance he plays (depth rank + final injury report), rates fit on 2021-23 only

Rule: at QB, expected must beat raw on BOTH MAE and per-week Spearman (paired week-bootstrap 95% CI
excluding zero). RB/WR/TE: no significant loss on either. Guard: count of QBs listed as backups
(depth rank >= 2) inside the weekly top 12, raw vs expected.

    python -m src.backtest.play_chance_test
"""

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from src.backtest.harness import points
from src.features.assemble import PBP_COLS, build_all
from src.features.team import read_schedules
from src.ingest.pbp import read_pbp
from src.ingest.player_weeks import INJURIES_DIR, read_dir
from src.models.boosting import TUNED_PATH, BoostingModel
from src.models.play_chance import PlayChance, game_day_pool
from src.predict.week import PW_PATH, ROSTERS_DIR, train_rows, upcoming_rows

SEASON, WEEKS, RATE_SEASONS = 2024, range(2, 19), [2021, 2022, 2023]
OUT = Path("data/processed/play_chance_test_2024.parquet")
REPORT = Path(f"docs/reports/play_chance_test_{date.today()}.md")
POSITIONS = ["QB", "RB", "WR", "TE"]


def predict_weeks() -> pl.DataFrame:
    pw_all = pl.read_parquet(PW_PATH)
    seasons = [s for s in sorted(pw_all["season"].unique().to_list()) if s <= SEASON]
    schedules = read_schedules().filter(pl.col("season").is_in(seasons))
    pbp = read_pbp(seasons, columns=PBP_COLS).join(schedules.select("game_id", "season", "week"), on="game_id", how="left")
    rosters, injuries = read_dir(ROSTERS_DIR, "rosters", [SEASON]), read_dir(INJURIES_DIR, "injuries", [SEASON])
    chance = PlayChance().fit(game_day_pool(RATE_SEASONS, schedules))
    actual = pw_all.filter(pl.col("season") == SEASON).select("player_id", "week", actual=pl.col("fantasy_points_ppr"))
    out = []
    for week in WEEKS:
        hist = pw_all.filter((pl.col("season") < SEASON) | ((pl.col("season") == SEASON) & (pl.col("week") < week)))
        feats = build_all(pl.concat([hist, upcoming_rows(rosters, schedules, injuries, SEASON, week)],
                                    how="diagonal_relaxed"), schedules, pbp)
        rows = feats.filter((pl.col("season") == SEASON) & (pl.col("week") == week)).sort("player_id")
        model = BoostingModel(tuned=TUNED_PATH).fit(train_rows(feats, SEASON, week))
        stats = rows.select("player_id", "position").join(model.predict(rows), on="player_id", how="left")
        preds = rows.select("player_id", "player_display_name", "position", "week", "depth_rank",
                            "report_status", "practice_code").with_columns(raw=points(stats.drop("player_id")))
        preds = chance.rates(preds).with_columns(expected=pl.col("raw") * pl.col("p_play"))
        out.append(preds.join(actual.filter(pl.col("week") == week).drop("week"), on="player_id", how="left")
                   .with_columns(played=pl.col("actual").is_not_null(), actual=pl.col("actual").fill_null(0.0)))
        print(f"week {week}: {rows.height} players", flush=True)
    return pl.concat(out)


def weekly(res: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for (pos, week), g in res.group_by("position", "week"):
        a = g["actual"].to_numpy()
        rows.append({"position": pos, "week": week,
                     "mae_raw": np.abs(g["raw"].to_numpy() - a).mean(), "mae_exp": np.abs(g["expected"].to_numpy() - a).mean(),
                     "sp_raw": spearmanr(g["raw"], a).statistic, "sp_exp": spearmanr(g["expected"], a).statistic})
    return pl.DataFrame(rows)


def paired_ci(diff: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), (n, len(diff)))].mean(1)
    return float(diff.mean()), *np.percentile(boots, [2.5, 97.5])


def main() -> None:
    res = predict_weeks()
    res.write_parquet(OUT)
    w = weekly(res)
    lines = [f"# Chance-he-plays test ({date.today()})", "", __doc__.split("\n\n")[1], "",
             "| position | MAE raw | MAE expected | MAE diff [95% CI] | Spearman raw | Spearman expected | Spearman diff [95% CI] |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    verdict = {}
    for pos in POSITIONS:
        g = w.filter(pl.col("position") == pos)
        md, ml, mh = paired_ci((g["mae_exp"] - g["mae_raw"]).to_numpy())
        sd, sl, sh = paired_ci((g["sp_exp"] - g["sp_raw"]).to_numpy())
        verdict[pos] = (mh < 0 and sl > 0) if pos == "QB" else (ml <= 0 and sh >= 0)
        lines.append(f"| {pos} | {g['mae_raw'].mean():.3f} | {g['mae_exp'].mean():.3f} | {md:+.3f} [{ml:+.3f}, {mh:+.3f}] "
                     f"| {g['sp_raw'].mean():.3f} | {g['sp_exp'].mean():.3f} | {sd:+.3f} [{sl:+.3f}, {sh:+.3f}] |")
    guard = {}
    for col in ("raw", "expected"):
        top = res.filter(pl.col("position") == "QB").sort(col, descending=True).group_by("week").head(12)
        guard[col] = top.filter(pl.col("depth_rank") >= 2).height
    lines += ["", f"Guard (QBs listed as backups inside the weekly top 12, {len(WEEKS)} weeks): raw {guard['raw']}, expected {guard['expected']}",
              "", "RULE: " + ("PASS -> use expected points" if all(verdict.values()) else f"FAIL {verdict}")]
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
