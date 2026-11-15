"""Offseason vacated opportunity (SCRUM-94): work left behind by players who LEFT the team.

Lucas, 2026-11-14: bench picks should target players who lost teammates and will gain touches.
vacated.py only covers teammates missing in-season; this covers offseason departures.

For each team-season, compare last season's roster to this season's Week 1 roster (published
before kickoff, so known before the season):
    offseason_vacated_tgt_{wr,te,rb,total}  share of last season's team targets that belonged to players
                                            no longer on the team (left, retired, unsigned, or not ACTIVE at Week 1)
    offseason_vacated_carry                 same for carries
    offseason_arrived_tgt / _carry          new players' target / carry share on their OLD teams last
                                            season: competition for the vacated work
Values are team-level and constant all season. The first season (no previous one) is blank.

Usage:
    python -m src.features.offseason
"""

from pathlib import Path

import polars as pl

from src.ingest.in_game_injuries import ROSTERS_DIR

PLAYER_WEEKS_PATH = Path("data/processed/player_weeks.parquet")
OUT_PATH = Path("data/processed/features_offseason.parquet")
KEY = ["player_id", "season", "week"]
POSITIONS = ["WR", "TE", "RB"]
FEATURES = [*[f"offseason_vacated_tgt_{p.lower()}" for p in POSITIONS], "offseason_vacated_tgt_total",
            "offseason_vacated_carry", "offseason_arrived_tgt", "offseason_arrived_carry"]


def read_week1_rosters(rosters_dir: Path = ROSTERS_DIR) -> pl.DataFrame:
    """season, team, player_id for players ACTIVE on a Week 1 regular-season roster."""
    frames = [pl.read_parquet(p, columns=["season", "week", "game_type", "team", "gsis_id", "status"])
              for p in sorted(rosters_dir.glob("rosters_*.parquet"))]
    if not frames:
        return pl.DataFrame(schema={"season": pl.Int32, "team": pl.String, "player_id": pl.String})
    r = pl.concat(frames, how="diagonal_relaxed")
    return (r.filter(pl.col("week") == 1, pl.col("game_type") == "REG", pl.col("status") == "ACT")
            .select(pl.col("season").cast(pl.Int32), "team", player_id="gsis_id").drop_nulls().unique())


def last_season_shares(pw: pl.DataFrame) -> pl.DataFrame:
    """Per player, per team he played for: share of that team's targets and carries in the season."""
    team_tot = pw.group_by("season", "team").agg(team_tgt=pl.col("targets").sum(), team_car=pl.col("carries").sum())
    return (pw.group_by("season", "team", "player_id").agg(pl.col("position").first(), tgt=pl.col("targets").sum(),
                                                            car=pl.col("carries").sum())
            .join(team_tot, on=["season", "team"])
            .with_columns(tgt_share=pl.when(pl.col("team_tgt") > 0).then(pl.col("tgt") / pl.col("team_tgt")).otherwise(0.0),
                          car_share=pl.when(pl.col("team_car") > 0).then(pl.col("car") / pl.col("team_car")).otherwise(0.0)))


def build_offseason_features(pw: pl.DataFrame, week1: pl.DataFrame) -> pl.DataFrame:
    shares = last_season_shares(pw).with_columns(season=(pl.col("season") + 1).cast(pl.Int32))  # keyed to the NEXT season
    week1 = week1.with_columns(pl.col("season").cast(pl.Int32))
    stayed = week1.select("season", "team", "player_id").with_columns(on_team=pl.lit(True))
    left = (shares.join(stayed, on=["season", "team", "player_id"], how="left")
            .filter(pl.col("on_team").is_null())
            # only seasons where we actually have this year's Week 1 roster
            .join(week1.select("season").unique(), on="season", how="semi"))
    vacated = left.group_by("season", "team").agg(
        *[pl.col("tgt_share").filter(pl.col("position") == p).sum().alias(f"offseason_vacated_tgt_{p.lower()}") for p in POSITIONS],
        offseason_vacated_tgt_total=pl.col("tgt_share").sum(), offseason_vacated_carry=pl.col("car_share").sum())
    # Arrivals: on this team's Week 1 roster, but last season's work came for a different team.
    arrived = (week1.join(shares.select("season", "player_id", old_team="team", tgt_share="tgt_share", car_share="car_share"),
                          on=["season", "player_id"], how="inner")
               .filter(pl.col("old_team") != pl.col("team"))
               .group_by("season", "team").agg(offseason_arrived_tgt=pl.col("tgt_share").sum(),
                                                offseason_arrived_carry=pl.col("car_share").sum()))
    seasons_known = week1.select("season").unique().join(shares.select("season").unique(), on="season", how="semi")
    teams = pw.select(pl.col("season").cast(pl.Int32), "team").unique().join(seasons_known, on="season", how="semi")
    team_feats = (teams.join(vacated, on=["season", "team"], how="left").join(arrived, on=["season", "team"], how="left")
                  .with_columns([pl.col(c).fill_null(0.0) for c in FEATURES]))
    return (pw.select(*KEY, "team").with_columns(pl.col("season").cast(pl.Int32))
            .join(team_feats, on=["season", "team"], how="left").select(*KEY, *FEATURES)
            .with_columns(pl.col("season").cast(pw.schema["season"])).sort(KEY))


def main() -> None:
    pw = pl.read_parquet(PLAYER_WEEKS_PATH)
    feats = build_offseason_features(pw, read_week1_rosters())
    feats.write_parquet(OUT_PATH)
    print(f"Wrote {OUT_PATH}: {feats.height:,} rows x {feats.width} columns")


if __name__ == "__main__":
    main()
