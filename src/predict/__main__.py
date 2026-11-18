"""Predict an upcoming week and save it with a timestamp (SCRUM-76).

    python -m src.predict --season 2026 --week 11
    python -m src.models.ranges                     # bust/ceiling method comparison (SCRUM-52)

Saves to predictions/<season>_wk<week>_<UTC timestamp>/:
    demo_players.csv   the 10 demo players (projection, likely range, opponent, injury status)
    all_players.csv    every active QB/RB/WR/TE
    games.csv          this week's games + Vegas spread/total (game predictions: SCRUM-75)
    espn.csv           ESPN's projections, fetched at the same moment (src/predict/benchmarks.py)
    fantasypros.csv    FantasyPros weekly expert rankings, same moment
    blend.csv          THE weekly number: average of ours, ESPN, FantasyPros (src/models/expert_blend.py)
    meta.txt           when it ran, model version, what it trained on
Run it again before kickoff to pick up the latest injury report; every run is kept.
"""

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from src.models import expert_blend
from src.predict import benchmarks
from src.predict.week import demo_table, project_week

OUT_DIR = Path("predictions")


def write_blend(out: Path, preds: pl.DataFrame, demo: pl.DataFrame) -> None:
    """blend.csv from this run's own predictions + the ESPN / FantasyPros numbers saved at the same moment."""
    espn = pl.read_csv(out / "espn.csv", schema_overrides={"player_id": pl.String})
    fp = pl.read_csv(out / "fantasypros.csv", schema_overrides={"player_id": pl.String})
    b = expert_blend.blend(preds.select("player_id", "player_display_name", "position", "team", "opponent_team",
                                        "report_status", "p_play", "proj"),
                           espn, fp, expert_blend.load_map())
    b.write_csv(out / "blend.csv")
    with open(out / "meta.txt", "a", encoding="utf-8") as f:
        f.write("blend = mean of proj (ours), espn_proj, fp_pts (FantasyPros ECR -> pts, models/fp_rank_points.json); "
                f"{(b['blend_sources'] == 3).sum()} players have all three\n")
    print("\nDemo players, blended (the weekly number):")
    print(b.join(demo.select("player_id"), on="player_id", how="semi")
          .select("player_display_name", "position", "proj", "espn_proj", "fp_ecr", "fp_pts", "blend", "blend_sources"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--no-benchmarks", action="store_true", help="skip the ESPN / FantasyPros snapshot")
    args = parser.parse_args()
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(12)
    pl.Config.set_tbl_width_chars(140)

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
        f"trained on: every finished game 2020-{args.season} before week {args.week} (2025 included since the final exam, v2)\n"
        f"likely range: 25th-75th pct outcome, BucketRanges(LIKELY) (src/models/ranges.py), 2022-24 backtest\n"
        f"calibration: per-position isotonic tier calibration fit on 2021-24 walk-forward predictions\n"
        f"proj = expected points = proj_if_plays x p_play (depth rank + injury report; src/models/play_chance.py)\n"
        f"players: {preds.height} active QB/RB/WR/TE\n"
    )

    print(f"\nDemo players, {args.season} week {args.week} (PPR):")
    print(demo.select("name", "position", "team", "opponent_team", "proj", "proj_if_plays", "p_play", "likely_low", "likely_high", "depth_rank", "report_status", "roster_status"))
    print("\nGames (spread_line: + = home team favored):")
    print(games)
    if not args.no_benchmarks:
        try:  # a benchmark site being down must never block our own locked predictions
            print("\n" + benchmarks.snapshot(args.season, args.week, out))
        except Exception as e:  # noqa: BLE001
            print(f"\nbenchmarks: FAILED ({e}); our predictions are saved. Retry: "
                  f"python -m src.predict.benchmarks --season {args.season} --week {args.week} --out {out}")
        else:
            write_blend(out, preds, demo)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
