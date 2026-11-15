"""Age + years of experience (SCRUM-90).

Lucas (2026-11-14): he'd rather draft a younger RB/WR. The data agrees, through production rather
than injuries: from one season to the next, RBs 29+ lost 3.7 PPG (median -40%) against 1.2 for RBs
under 25. Until now the model had no idea how old anyone was.

    age        years old on game day (from roster birth date)
    years_exp  NFL seasons before this one (0 = rookie), from the player's first roster season

Both are known before kickoff and never depend on results, so they can't leak.

Usage:
    python -m src.features.age
"""

from pathlib import Path

import polars as pl

from src.ingest.in_game_injuries import ROSTERS_DIR

PLAYER_WEEKS_PATH = Path("data/processed/player_weeks.parquet")
OUT_PATH = Path("data/processed/features_age.parquet")
KEY = ["player_id", "season", "week"]
FEATURES = ["age", "years_exp"]


def read_players(rosters_dir: Path = ROSTERS_DIR) -> pl.DataFrame:
    """One row per player: birth date and first NFL season, from every weekly roster file on disk."""
    rosters = pl.concat([pl.read_parquet(p, columns=["gsis_id", "birth_date", "entry_year"])
                         for p in sorted(rosters_dir.glob("rosters_*.parquet"))], how="diagonal_relaxed")
    return players_from_rosters(rosters)


def players_from_rosters(rosters: pl.DataFrame) -> pl.DataFrame:
    return (rosters.drop_nulls("gsis_id").group_by("gsis_id")
            .agg(birth_date=pl.col("birth_date").cast(pl.Date, strict=False).drop_nulls().first(),
                 entry_year=pl.col("entry_year").drop_nulls().min())
            .rename({"gsis_id": "player_id"}))


def build_age_features(pw: pl.DataFrame, players: pl.DataFrame) -> pl.DataFrame:
    return (pw.select(KEY + ["gameday"]).join(players, on="player_id", how="left")
            .select(KEY,
                    age=(pl.col("gameday").cast(pl.String).str.to_date() - pl.col("birth_date")).dt.total_days() / 365.25,
                    years_exp=(pl.col("season") - pl.col("entry_year")).cast(pl.Float64)))


def main() -> None:
    pw = pl.read_parquet(PLAYER_WEEKS_PATH)
    feats = build_age_features(pw, read_players())
    missing = feats["age"].null_count()
    feats.write_parquet(OUT_PATH)
    print(f"Wrote {OUT_PATH}: {feats.height:,} rows x {feats.width} columns ({missing} rows without a birth date)")


if __name__ == "__main__":
    main()
