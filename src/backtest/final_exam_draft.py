"""FINAL EXAM, part B: the draft side on the untouched 2025 season. Runs ONCE (SCRUM-66).

Part A (weekly) ran 2026-11-18 on tag model-freeze. This part grades the frozen DRAFT pipeline.

Rules (LOCKED 2026-11-18, before the 2025 board or any 2025 draft result was built or looked at):
    board     src/models/season.py exactly as frozen: option A (weekly model on Week 1 preseason rows,
              trained on 2020-24) + season model B (with ADP inputs, SCRUM-93) trained on 2021-24 seasons.
    tests     1. close pairs vs ECR (src/backtest/vs_consensus.py rules: market top-200, same position,
                 within 12 spots, total PPR season points): our board's win rate when it disagrees.
                 Reported; "CONFIRMED" only if CI > 50%. Dev: 47.1% [40.4, 53.7].
              2. HEADLINE mock draft (src/backtest/mock_draft.py rules, 12 teams, standard PPR, 10 reps x
                 12 slots, opponents draft from ECR + noise): paired edge of the sim drafter over the ADP
                 drafter. "CONFIRMED" if the 95% CI is entirely > 0. Dev: +58 [+35, +82].
                 Also reported: vor - adp, sim - vor, each drafter vs league average.
    safety    refuses to run if the results file exists, or if model code (src/models, src/features,
              src/draft, src/scoring, models/) differs from tag model-freeze. The frozen modules' 2025
              guards are lifted from HERE only, for this run. Results reported as-is.

    python -m src.backtest.final_exam_draft
"""

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

import src.backtest.mock_draft as mock
import src.backtest.vs_consensus as vc
from src.features.team import read_schedules
from src.models import season as sm
from src.scoring.config import PRESETS

EXAM = 2025
TAG = "model-freeze"
FROZEN = ["src/models", "src/features", "src/draft", "src/scoring", "models"]
RESULTS = Path("docs/reports/final_exam_2025_draft.md")
MOCK_PATH = Path("docs/reports/final_exam_2025_mock.parquet")


def check_frozen() -> str:
    if RESULTS.exists():
        raise SystemExit(f"{RESULTS} exists: part B already ran. It runs once.")
    tag = subprocess.run(["git", "rev-parse", "--short", TAG], capture_output=True, text=True)
    if tag.returncode != 0:
        raise SystemExit(f"git tag {TAG} not found")
    if subprocess.run(["git", "diff", "--quiet", TAG, "--", *FROZEN]).returncode != 0:
        raise SystemExit(f"model code differs from tag {TAG}")
    return tag.stdout.strip()


def build_board() -> pl.DataFrame:
    """2025 board from the frozen season pipeline, trained on 2021-24 only."""
    pw, schedules = pl.read_parquet(sm.PLAYER_WEEKS_PATH), read_schedules()
    a = sm.a_projection(EXAM)                       # weekly model trained on seasons < 2025
    test = sm.season_dataset(a, pw, schedules)
    train = sm.season_table().filter(pl.col("season") < EXAM)
    b = sm.board(test.with_columns(season_b=pl.Series(sm.SeasonModel().fit(train).predict(test))))
    b.write_parquet(sm.OUT_DIR / f"board_{EXAM}.parquet")
    return b


def main() -> None:
    commit = check_frozen()
    started = datetime.now(timezone.utc).isoformat()
    b = build_board()
    print(f"2025 board built: {b.height} players. Top 10: "
          + ", ".join(f"{r['player_display_name']} ({r['position']}{r['pos_rank']})" for r in b.head(10).iter_rows(named=True)),
          flush=True)

    vc.SEASONS, vc.EXAM_SEASON = [EXAM], -1        # lift the guard for this run only
    lines = [f"# FINAL EXAM 2025, part B: draft side (run once, {started})", "",
             f"Frozen model code: tag `{TAG}` = {commit}.", "", __doc__.split("\n\n")[2], ""]
    close = {}
    for market, name in [("ecr", "our board vs ECR (primary)"), ("adp", "our board vs ADP")]:
        rows, ppg, dropped = vc.run(market, "season_b")
        close[market] = rows[0]
        lines += [f"## Close pairs: {name}", "", *vc.table(rows[:5]), "", "Points per game view:", "",
                  *vc.table(ppg), "", f"Pairs dropped (not on our board): {dropped}", ""]
    print("close-pair tests done; starting mock drafts", flush=True)

    res = mock.run(10, PRESETS["standard-ppr"], False, upside=False, seasons=[EXAM])
    res.write_parquet(MOCK_PATH)
    lines += ["## Mock drafts", "", mock.report(res).split("\n", 2)[2]]
    wide = res.pivot(on="drafter", index=["season", "slot", "rep"], values="edge")
    from src.backtest.harness import bootstrap
    ci = bootstrap((wide["sim"] - wide["adp"]).to_numpy())
    e = close["ecr"]
    lines += ["## Verdict (rules locked before the run)", "",
              f"1. Board vs ECR close pairs: {e['win_rate']:.1%} [{e['lo']:.1%}, {e['hi']:.1%}] -> "
              + ("CONFIRMED" if e["lo"] > 0.5 else "NOT CONFIRMED") + "  (dev 47.1% [40.4, 53.7])",
              f"2. **HEADLINE sim drafter minus ADP drafter: {ci.mean:+.0f} pts/season [{ci.low:+.0f}, {ci.high:+.0f}] -> "
              + ("CONFIRMED" if ci.low > 0 else "NOT CONFIRMED") + "**  (dev +58 [+35, +82])"]
    RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    main()
