"""Availability model (SCRUM-51): how many games will a player miss to injury this season?

Feeds season projections (SCRUM-53): expected games = team games - expected injury misses.

What counts as an injury miss (Lucas, 2026-11-14: injuries only, not benchings). A week the
player's team played, he didn't play, and either:
    - his roster status was an injury reserve list: R01 injured reserve, R48 IR designated to
      return, R04 PUP, R05 non-football injury. (Suspensions R30/R40 and the COVID lists R59/R62
      are not injuries and don't count.)
    - or the injury report listed him Out or Doubtful.
Benchings, healthy scratches and cuts don't count: role changes are the projection's job (snap
share, backup_qb_starting), and counting them here would punish a benched player twice.

Who it covers: players with 8+ PPR points per game in 4+ games last season who are on a Week 1
roster this season. Everyone else (rookies, no history) gets the position average.

Models, all fit on earlier seasons only:
    position_avg   average injury misses for the position (the bar to beat)
    rates          average by position x last season's misses (0 / 1-3 / 4+)
    poisson        Poisson regression: position, age, misses last season and two seasons ago

Usage:
    python -m src.models.availability     # walk-forward 2022-2024 report (2025 untouched)
"""

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import PoissonRegressor

from src.backtest.harness import bootstrap
from src.features.team import read_schedules
from src.ingest.availability import injuries_path
from src.ingest.in_game_injuries import rosters_path

PLAYER_WEEKS_PATH = Path("data/processed/player_weeks.parquet")
REPORT_DIR = Path("docs/reports")
POSITIONS = ["QB", "RB", "WR", "TE"]
INJURY_CODES = {"R01", "R48", "R04", "R05"}
OUT_STATUSES = {"Out", "Doubtful"}
MIN_PPG, MIN_GAMES = 8.0, 4
FIRST_SEASON = 2020
EVAL_SEASONS = [2022, 2023, 2024]  # 2025 is the final exam


def injury_weeks(rosters: pl.DataFrame, injuries: pl.DataFrame, schedules: pl.DataFrame,
                 played: pl.DataFrame) -> pl.DataFrame:
    """One row per player per week his roster team had a regular-season game: played, injury_miss."""
    reg = schedules.filter(pl.col("game_type") == "REG")
    team_weeks = pl.concat([reg.select("season", "week", team="home_team"),
                            reg.select("season", "week", team="away_team")]).unique()
    ros = (rosters.filter(pl.col("game_type") == "REG", pl.col("position").is_in(POSITIONS))
           .select("season", "week", "team", "position", "birth_date", player_id="gsis_id",
                   reserve_injury=(pl.col("status") == "RES") & pl.col("status_description_abbr").is_in(INJURY_CODES))
           .unique(["player_id", "season", "week"], keep="first", maintain_order=True)
           .join(team_weeks, on=["season", "week", "team"], how="inner"))  # bye weeks drop out
    out = (injuries.filter(pl.col("game_type") == "REG", pl.col("report_status").is_in(OUT_STATUSES))
           .select("season", "week", player_id="gsis_id").unique().with_columns(reported_out=pl.lit(True)))
    return (ros.join(played.select("player_id", "season", "week").unique().with_columns(played=pl.lit(True)),
                     on=["player_id", "season", "week"], how="left")
            .join(out, on=["player_id", "season", "week"], how="left")
            .with_columns(pl.col("played").fill_null(False), pl.col("reported_out").fill_null(False))
            .with_columns(injury_miss=~pl.col("played") & (pl.col("reserve_injury") | pl.col("reported_out"))))


def season_table(weeks: pl.DataFrame, played: pl.DataFrame) -> pl.DataFrame:
    """One row per player-season in the covered population, with the target and preseason inputs."""
    # Sorted by week so a player listed at two positions gets his earliest one, the same every run.
    misses = weeks.sort("season", "week").group_by("player_id", "season", maintain_order=True).agg(
        injury_missed=pl.col("injury_miss").sum().cast(pl.Float64),
        position=pl.col("position").first(), birth_date=pl.col("birth_date").drop_nulls().first())
    week1 = weeks.filter(pl.col("week") == 1).select("player_id", "season").unique()
    form = played.group_by("player_id", "season").agg(gp=pl.len(), ppg=pl.col("fantasy_points_ppr").mean())
    relevant_last = (form.filter(pl.col("ppg") >= MIN_PPG, pl.col("gp") >= MIN_GAMES)
                     .select("player_id", season=pl.col("season") + 1))
    lag = lambda k, name: misses.select("player_id", pl.col("season") + k, pl.col("injury_missed").alias(name))
    return (week1.join(relevant_last, on=["player_id", "season"])
            .join(misses, on=["player_id", "season"])
            .join(lag(1, "prev_missed"), on=["player_id", "season"], how="left")
            .join(lag(2, "prev2_missed"), on=["player_id", "season"], how="left")
            .with_columns(pl.col("prev_missed").fill_null(0.0),  # relevant last season, so on a roster: no row = no misses
                          has_prev2=pl.col("season") - 2 >= FIRST_SEASON,
                          age=(pl.date(pl.col("season"), 9, 1) - pl.col("birth_date").cast(pl.Date, strict=False))
                          .dt.total_days() / 365.25)
            .sort("season", "player_id"))


def bucket(prev_missed: pl.Expr) -> pl.Expr:
    return pl.when(prev_missed == 0).then(pl.lit("0")).when(prev_missed <= 3).then(pl.lit("1-3")).otherwise(pl.lit("4+"))


class PositionAverage:
    name = "position_avg"

    def fit(self, train: pl.DataFrame):
        self.avg = train.group_by("position").agg(pred=pl.col("injury_missed").mean())
        return self

    def predict(self, rows: pl.DataFrame) -> np.ndarray:
        return rows.select("position").join(self.avg, on="position", how="left")["pred"].fill_null(0).to_numpy()


class Rates:
    name = "rates"

    def fit(self, train: pl.DataFrame):
        self.fallback = PositionAverage().fit(train)
        self.avg = train.with_columns(b=bucket(pl.col("prev_missed"))).group_by("position", "b").agg(
            pred=pl.col("injury_missed").mean())
        return self

    def predict(self, rows: pl.DataFrame) -> np.ndarray:
        got = rows.with_columns(b=bucket(pl.col("prev_missed"))).select("position", "b").join(
            self.avg, on=["position", "b"], how="left")["pred"].to_numpy()
        return np.where(np.isnan(got), self.fallback.predict(rows), got)


class Poisson:
    name = "poisson"

    def fit(self, train: pl.DataFrame):
        self.age_mean = train["age"].mean()
        with_prev2 = train.filter("has_prev2")
        # Training on 2021 alone has no season-two-years-ago; last season's average stands in.
        self.prev2_mean = (with_prev2["prev2_missed"].fill_null(0) if with_prev2.height else train["prev_missed"]).mean()
        x = self._x(train)
        # A column with no spread in training (prev2 when training on 2021 alone) gets weight 0:
        # dividing by its ~0 spread would blow up any new value.
        self.mu, sd = x.mean(axis=0), x.std(axis=0)
        self.sd = np.where(sd < 1e-6, np.inf, sd)
        self.model = PoissonRegressor(alpha=0.01, max_iter=1000).fit((x - self.mu) / self.sd, train["injury_missed"].to_numpy())
        return self

    def _x(self, rows: pl.DataFrame) -> np.ndarray:
        r = rows.with_columns(
            pl.col("age").fill_null(self.age_mean),
            # Seasons without a season-two-years-ago in our data (2021) get the training average.
            prev2=pl.when("has_prev2").then(pl.col("prev2_missed").fill_null(0)).otherwise(self.prev2_mean))
        return np.column_stack([*(r["position"] == p for p in POSITIONS[1:]), r["age"], r["prev_missed"], r["prev2"]]).astype(float)

    def predict(self, rows: pl.DataFrame) -> np.ndarray:
        return self.model.predict((self._x(rows) - self.mu) / self.sd)


MODELS = [PositionAverage, Rates, Poisson]
# Lucas, 2026-11-14: position average feeds season projections (SCRUM-53). It tied the best model
# on 2022-24 (docs/reports/availability_2026-11-14.md), has nothing to overfit, and is easy to explain.
CHOSEN = PositionAverage


def walk_forward(table: pl.DataFrame, seasons: list[int] = EVAL_SEASONS) -> pl.DataFrame:
    """Predict each season from earlier seasons only. One row per player-season per model."""
    if 2025 in seasons:
        raise ValueError("2025 is the final exam: run it once, on purpose, not here")
    out = []
    for season in seasons:
        train, test = table.filter(pl.col("season") < season), table.filter(pl.col("season") == season)
        for cls in MODELS:
            model = cls().fit(train)
            out.append(test.select("player_id", "season", "position", "injury_missed").with_columns(
                model=pl.lit(model.name), pred=pl.Series(np.clip(model.predict(test), 0, 17))))
    return pl.concat(out).with_columns(abs_err=(pl.col("pred") - pl.col("injury_missed")).abs())


def report(results: pl.DataFrame) -> str:
    """MAE (games off per player) and MSE (squared games off). We want EXPECTED games for season
    projections, and the mean is what minimizes squared error, so MSE is the main scorecard;
    MAE rewards guessing the median (1 game) instead."""
    results = results.with_columns(sq_err=(pl.col("pred") - pl.col("injury_missed")) ** 2)
    lines = [f"# Availability model backtest ({date.today()})", "",
             f"Walk-forward over {EVAL_SEASONS} (2025 untouched). Target: games missed to injury.",
             "Values: mean [95% CI over players]. Difference vs position_avg: negative = better."]
    for metric, label in [("sq_err", "MSE (main)"), ("abs_err", "MAE")]:
        lines += ["", f"## {label}", "",
                  "| position | players | actual avg | " + " | ".join(m.name for m in MODELS) + " | rates - avg | poisson - avg |",
                  "| --- " * (5 + len(MODELS)) + "|"]
        for pos in [*POSITIONS, "ALL"]:
            r = results if pos == "ALL" else results.filter(pl.col("position") == pos)
            err = {m.name: r.filter(pl.col("model") == m.name).sort("season", "player_id")[metric].to_numpy() for m in MODELS}
            base = r.filter(pl.col("model") == "position_avg")
            cells = [f"{bootstrap(err[m.name])}" for m in MODELS]
            diffs = [f"{bootstrap(err[n] - err['position_avg'])}" for n in ("rates", "poisson")]
            lines.append(f"| {pos} | {base.height} | {base['injury_missed'].mean():.2f} | " + " | ".join(cells + diffs) + " |")
    return "\n".join(lines) + "\n"


def load_inputs(seasons: range) -> tuple[pl.DataFrame, pl.DataFrame]:
    rosters = pl.concat([pl.read_parquet(rosters_path(s)) for s in seasons], how="diagonal_relaxed")
    injuries = pl.concat([pl.read_parquet(injuries_path(s)) for s in seasons], how="diagonal_relaxed")
    played = pl.read_parquet(PLAYER_WEEKS_PATH)
    weeks = injury_weeks(rosters, injuries, read_schedules(), played)
    return weeks, season_table(weeks, played)


def main() -> None:
    _, table = load_inputs(range(FIRST_SEASON, 2026))
    text = report(walk_forward(table))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"availability_{date.today()}.md"
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
