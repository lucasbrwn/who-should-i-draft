"""Predict an upcoming week and save it with a timestamp (SCRUM-76).

    python -m src.predict --season 2026 --week 11
    python -m src.predict --season 2026 --week 11 --check-ranges   # bust/ceiling coverage on 2024

Saves to predictions/<season>_wk<week>_<UTC timestamp>/:
    demo_players.csv   the 10 demo players (projection, bust, ceiling, opponent, injury status)
    all_players.csv    every active QB/RB/WR/TE
    games.csv          this week's games + Vegas spread/total (game predictions: SCRUM-75)
    meta.txt           when it ran, model version, what it trained on
Run it again before kickoff to pick up the latest injury report; every run is kept.
"""

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from src.predict.week import BACKTEST_PATH, demo_table, project_week, range_coverage

OUT_DIR = Path("predictions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--check-ranges", action="store_true")
    args = parser.parse_args()
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(12)
    pl.Config.set_tbl_width_chars(140)

    if args.check_ranges:
        print("Bust/ceiling coverage: ranges fit on 2022-23 backtest, checked on 2024 (target ~80% inside)")
        print(range_coverage(pl.read_parquet(BACKTEST_PATH), [2022, 2023], 2024))

    now = datetime.now(timezone.utc)
    preds, games = project_week(args.season, args.week)
    demo = demo_table(preds, args.season, args.week)

    out = OUT_DIR / f"{args.season}_wk{args.week:02d}_{now:%Y%m%dT%H%M%SZ}"
    out.mkdir(parents=True)
    demo.write_csv(out / "demo_players.csv")
    preds.write_csv(out / "all_players.csv")
    games.write_csv(out / "games.csv")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    (out / "meta.txt").write_text(
        f"generated_utc: {now.isoformat()}\nseason: {args.season}\nweek: {args.week}\n"
        f"model: boosting (tuned, models/boosting_params.json), git {commit}\n"
        f"trained on: every finished game 2020-{args.season} before week {args.week}, except 2025 (final exam holdout)\n"
        f"bust/ceiling: 10th/90th pct of actual points by position x projection bucket, 2022-24 backtest\n"
        f"players: {preds.height} active QB/RB/WR/TE\n"
    )

    print(f"\nDemo players, {args.season} week {args.week} (PPR):")
    print(demo.select("name", "position", "team", "opponent_team", "proj", "bust", "ceiling", "report_status", "roster_status"))
    print("\nGames (spread_line: + = home team favored):")
    print(games)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
