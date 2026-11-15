"""Market consensus before each season (SCRUM-55, SCRUM-43): FantasyPros ECR and ADP, matched to gsis ids.

ECR  FantasyPros expert consensus rankings, PPR cheat sheet, the last weekly snapshot before
     Week 1 kickoff (nflverse load_ff_rankings; 5-6 days before kickoff every year, 2021-2026).
     Matched to our players through nflverse's fantasypros_id -> gsis_id table.
ADP  FantasyPros PPR ADP averaged across ESPN/Sleeper/CBS/RTSports/Fantrax, gathered by Lucas
     into data/reference/adp/ (2020-2026). Checked 2026-11-14 to be frozen before Week 1: players
     hurt in Weeks 2-4 kept their preseason ADP. Quirks handled here:
       - "Player (Bye)" shows the player's team TODAY, not that season's, so it's ignored and
         players are matched by name + position against that season's rosters.
       - "Rank" isn't sorted by the average (players on one platform drift down), so AVG is the ADP.
       - nicknames ("Hollywood Brown") need the alias table below.

Usage:
    python -m src.ingest.consensus     # download ECR + id map, report match rates
"""

from pathlib import Path

import nflreadpy
import polars as pl

from src.features.team import read_schedules
from src.ingest.in_game_injuries import rosters_path

RANKINGS_PATH = Path("data/raw/ff_rankings_all.parquet")
IDS_PATH = Path("data/raw/ff_playerids.parquet")
ADP_DIR = Path("data/reference/adp")
ECR_PAGE = "/nfl/rankings/ppr-cheatsheets.php"
POSITIONS = ["QB", "RB", "WR", "TE"]
SUFFIX = r"\s+(jr|sr|ii|iii|iv|v)$"
# FantasyPros name -> name on nflverse rosters (after normalize()).
ALIASES = {
    "hollywood brown": "marquise brown", "robbie chosen": "robby anderson", "robbie anderson": "robby anderson",
    "nyheim miller-hines": "nyheim hines", "joshua palmer": "josh palmer", "kenny gainwell": "kenneth gainwell",
    "gabe davis": "gabriel davis", "chig okonkwo": "chigoziem okonkwo", "william fuller": "will fuller",
    "jeff wilson": "jeffery wilson", "demario douglas": "demario douglas", "aj dillon": "aj dillon",
    "cam ward": "cameron ward",
}


def download() -> None:
    RANKINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    nflreadpy.load_ff_rankings("all").write_parquet(RANKINGS_PATH)
    nflreadpy.load_ff_playerids().write_parquet(IDS_PATH)


def normalize(name: pl.Expr) -> pl.Expr:
    """Lowercase, no periods/apostrophes, no Jr./III suffix: 'D.J. Moore' and 'DJ Moore' match."""
    n = name.str.to_lowercase().str.replace_all(r"[.'’]", "").str.replace_all(r"\s+", " ").str.strip_chars()
    n = n.str.replace(SUFFIX, "")
    return n.replace(ALIASES)


def kickoff(season: int) -> pl.Expr:
    week1 = read_schedules().filter(pl.col("season") == season, pl.col("week") == 1, pl.col("game_type") == "REG")
    return week1["gameday"].str.to_date().min()


def load_ecr(season: int) -> pl.DataFrame:
    """player_id, position, ecr (overall PPR rank) from the last snapshot before Week 1."""
    r = (pl.read_parquet(RANKINGS_PATH).filter(pl.col("fp_page") == ECR_PAGE)
         .with_columns(d=pl.col("scrape_date").cast(pl.Date)))
    before = r.filter(pl.col("d") < kickoff(season), pl.col("d") >= pl.date(season, 7, 1))
    if before.is_empty():
        raise ValueError(f"no preseason ECR snapshot for {season}")
    snap = before.filter(pl.col("d") == before["d"].max(), pl.col("pos").is_in(POSITIONS))
    ids = pl.read_parquet(IDS_PATH).select(pl.col("fantasypros_id").cast(pl.String), "gsis_id").drop_nulls().unique("fantasypros_id")
    return (snap.select(fantasypros_id=pl.col("id").cast(pl.String), ecr_name="player", position="pos",
                        ecr=pl.col("ecr").cast(pl.Float64), ecr_date="d")
            .join(ids, on="fantasypros_id", how="left").rename({"gsis_id": "player_id"}).sort("ecr"))


def load_adp(season: int) -> pl.DataFrame:
    """player_id, position, adp (average pick across platforms), matched by name + position."""
    raw = pl.read_csv(ADP_DIR / f"fantasypros_adp_{season}.csv", infer_schema_length=0)
    adp = (raw.select(adp_name=pl.col("Player (Bye)").str.replace(r"\s+[A-Z]{2,3}\s+\(\d+\)$", ""),
                      position=pl.col("POS").str.extract(r"^([A-Z]+)"), adp=pl.col("AVG").cast(pl.Float64))
           .filter(pl.col("position").is_in(POSITIONS)).with_columns(key=normalize(pl.col("adp_name"))))
    rosters = pl.read_parquet(rosters_path(season), columns=["gsis_id", "full_name", "position"]).drop_nulls()
    names = (rosters.select(player_id="gsis_id", position="position", key=normalize(pl.col("full_name")))
             .unique(["player_id"]).filter(pl.col("key").is_unique().over("position", "key")))  # ambiguous names dropped
    return adp.join(names, on=["key", "position"], how="left").drop("key").sort("adp")


def main() -> None:
    download()
    for season in range(2021, 2027):
        ecr, adp = load_ecr(season), load_adp(season)
        top_e, top_a = ecr.head(200), adp.head(200)
        print(f"{season}: ECR snapshot {ecr['ecr_date'][0]}, top-200 matched {top_e['player_id'].is_not_null().sum()}/200"
              f" | ADP top-200 matched {top_a['player_id'].is_not_null().sum()}/200"
              f" (unmatched: {', '.join(top_a.filter(pl.col('player_id').is_null())['adp_name'].head(5))})")


if __name__ == "__main__":
    main()
