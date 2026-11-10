"""One command rebuilds every feature table and the combined features.parquet (SCRUM-39).

    python -m src.features

Order matters: matchup needs usage + defense. Each step runs its own checks and stops on failure.
"""

import polars as pl

from src.features import defense, matchup, team, usage, vacated
from src.features.assemble import OUT_PATH, assemble, check_features, feature_columns

PW_PATH = "data/processed/player_weeks.parquet"


def main() -> None:
    for step in (usage, team, defense, matchup, vacated):
        print(f"--- {step.__name__}")
        step.main()

    print("--- assemble")
    pw = pl.read_parquet(PW_PATH)
    groups = [pl.read_parquet(p) for p in (usage.OUT_PATH, team.OUT_PATH, matchup.OUT_PATH, vacated.OUT_PATH)]
    feats = assemble(pw, groups)
    problems = check_features(feats, pw)
    for p in problems:
        print("CHECK FAILED:", p)
    if problems:
        raise SystemExit(1)
    feats.write_parquet(OUT_PATH)
    print(f"Wrote {OUT_PATH}: {feats.height:,} rows, {len(feature_columns(feats))} features, "
          f"{sum(c.startswith('y_') for c in feats.columns)} targets")


if __name__ == "__main__":
    main()
