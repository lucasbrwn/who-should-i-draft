"""Do scenario projections help? (rule set 2026-11-17, before results)

Live-style 2024 test, like play_chance_test.py: every pre-game active player each week, 0 if he didn't
play, chance-he-plays rates from 2021-23. Two versions of expected points:

    current   proj (if he plays, as if every teammate plays) x chance he plays
    scenario  teammates of key players who might sit (Q/D/Out starters) get
              q x proj(they play) + (1 - q) x proj(they're out), then x chance he plays

Rule: on AFFECTED players (teammates of a key player in doubt), scenario must improve MAE (95% CI
below 0, week bootstrap). On ALL players, no position significantly worse on MAE or Spearman.

    python -m src.backtest.scenario_test
"""

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from src.backtest.play_chance_test import paired_ci
from src.features.assemble import PBP_COLS
from src.features.team import read_schedules
from src.ingest.pbp import read_pbp
from src.ingest.player_weeks import INJURIES_DIR, read_dir
from src.models.play_chance import PlayChance, game_day_pool
from src.predict.week import PW_PATH, ROSTERS_DIR, project_rows, upcoming_rows

SEASON, WEEKS, RATE_SEASONS = 2024, range(2, 19), [2021, 2022, 2023]
OUT = Path("data/processed/scenario_test_2024.parquet")
REPORT = Path(f"docs/reports/scenario_test_{date.today()}.md")
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
        res = project_rows(hist, upcoming_rows(rosters, schedules, injuries, SEASON, week), schedules, pbp,
                           SEASON, week, chance)
        res = chance.rates(res).with_columns(week=pl.lit(week), current=pl.col("proj_with") * pl.col("p_play"),
                                             scenario=pl.col("proj") * pl.col("p_play"))
        out.append(res.join(actual.filter(pl.col("week") == week).drop("week"), on="player_id", how="left")
                   .with_columns(actual=pl.col("actual").fill_null(0.0)))
        print(f"week {week}: {res.height} players, {res.filter(pl.col('absent_chance') > 0).height} affected", flush=True)
    return pl.concat(out)


def by_week(res: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    rows = []
    for (week,), g in res.group_by("week"):
        a = g["actual"].to_numpy()
        rows.append({"week": week, **{f"mae_{c}": np.abs(g[c].to_numpy() - a).mean() for c in cols},
                     **{f"sp_{c}": spearmanr(g[c], a).statistic for c in cols}})
    return pl.DataFrame(rows)


def main() -> None:
    res = predict_weeks()
    res.write_parquet(OUT)
    lines = [f"# Scenario projection test ({date.today()})", "", __doc__.split("\n\n")[1], ""]
    aff = res.filter(pl.col("absent_chance") > 0)
    w = by_week(aff, ["current", "scenario"])
    md, ml, mh = paired_ci((w["mae_scenario"] - w["mae_current"]).to_numpy())
    lines += [f"Affected player-weeks: {aff.height} (of {res.height})", "",
              f"AFFECTED MAE: current {w['mae_current'].mean():.3f}, scenario {w['mae_scenario'].mean():.3f}, "
              f"diff {md:+.3f} [{ml:+.3f}, {mh:+.3f}]", "",
              "| position (all players) | MAE diff [95% CI] | Spearman diff [95% CI] |", "| --- | --- | --- |"]
    ok_all = True
    for pos in POSITIONS:
        g = by_week(res.filter(pl.col("position") == pos), ["current", "scenario"])
        d1, l1, h1 = paired_ci((g["mae_scenario"] - g["mae_current"]).to_numpy())
        d2, l2, h2 = paired_ci((g["sp_scenario"] - g["sp_current"]).to_numpy())
        ok_all &= (l1 <= 0) and (h2 >= 0)
        lines.append(f"| {pos} | {d1:+.3f} [{l1:+.3f}, {h1:+.3f}] | {d2:+.4f} [{l2:+.4f}, {h2:+.4f}] |")
    passed = mh < 0 and ok_all
    lines += ["", "RULE: " + ("PASS -> use scenario projections" if passed else "FAIL -> keep current")]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
