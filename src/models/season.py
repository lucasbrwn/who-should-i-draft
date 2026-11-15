"""Season projections for the draft (SCRUM-53).

Two layers (Lucas, 2026-11-14: "B stacked with A"):

A. Weekly model on preseason rows: projected points per game if he plays.
    1. One row per QB/RB/WR/TE on a Week 1 roster. Players without a real Week 1 game get a
       placeholder row with blank stats, and the normal feature builders fill it from earlier
       games only (the rows match features.parquet exactly for players who did play Week 1).
    2. Opponent/game inputs are set to what a typical Week 1 training row has (neutralize()).
    3. The boosting model, trained on earlier seasons, predicts per-game stats -> PPR points.
   On its own A lost to "last season's PPG" on 2024: player_weeks only holds games played, so the
   weekly model learned points GIVEN he plays (backups looked like starters), small samples weren't
   regressed, and one-game predictions pull elite players down hard.

B. Season model: predicts actual season points (0 if he never played, so backups and lost
   seasons are learned), from preseason facts plus A's per-game projection as one input:
   last season's per-game stats and games played, two seasons ago, age, experience, draft slot,
   team/coach/QB changes, win totals, vacated opportunity, Week 1 implied total, Week 1 starter.
   Trained on earlier seasons only (walk-forward), compared against the naive board
   (last season's PPG x 17) and A x expected games.

2025 is the final exam (SCRUM-66): it's never built, trained on, or scored here. The 2026 board
trains on 2021-2024 until the exam is done.

Usage:
    python -m src.models.season                 # build inputs, backtest 2022-2024, write boards
"""

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from src.backtest.harness import SCORING, bootstrap
from src.features.assemble import PBP_COLS, build_all, feature_columns
from src.features.team import read_schedules
from src.ingest.in_game_injuries import rosters_path
from src.ingest.pbp import read_pbp
from src.models import availability
from src.models.boosting import TUNED_PATH, BoostingModel
from src.scoring.engine import add_fantasy_points

FEATURES_PATH = Path("data/processed/features.parquet")
PLAYER_WEEKS_PATH = Path("data/processed/player_weeks.parquet")
OUT_DIR = Path("data/processed/season_projections")
REPORT_DIR = Path("docs/reports")
POSITIONS = ["QB", "RB", "WR", "TE"]
BOARD_STATUSES = {"ACT", "RES", "INA"}  # on the team (practice squad and cut players aren't drafted)
FIRST_SEASON = 2021  # needs a previous season of data
EVAL_SEASONS = [2022, 2023, 2024]
LIVE_SEASON = 2026
EXAM_SEASON = 2025
# Inputs that describe the Week 1 opponent/game rather than the player or his team.
GAME_SPECIFIC = ["div_game", "temp", "wind", "is_home", "team_spread", "opp_implied_total", "is_dome",
                 "deep_matchup", "short_matchup"]
# Preseason inputs for the season model (B), taken from the Week 1 feature rows.
ROW_INPUTS = ["age", "years_exp", "player_new_team", "no_games_last_season", "win_total", "win_total_change",
              "new_head_coach", "new_starting_qb", "backup_qb_starting", "implied_total",
              "vacated_tgt_share_wr", "vacated_tgt_share_te", "vacated_tgt_share_rb", "vacated_tgt_share_total",
              "vacated_carry_share",
              # offseason departures/arrivals (SCRUM-94)
              "offseason_vacated_tgt_wr", "offseason_vacated_tgt_te", "offseason_vacated_tgt_rb",
              "offseason_vacated_tgt_total", "offseason_vacated_carry", "offseason_arrived_tgt", "offseason_arrived_carry"]
# SCRUM-93 (Lucas, 2026-11-14, triggered by the locked mock-draft rule): the market's pre-draft ADP is
# a season-model input, so the model learns where the crowd knows more (RB roles, camp news) and
# where our stats do. Snapshot is FantasyPros' pre-Week-1 ADP (checked frozen before the season).
# This reverses the September "consensus is only a benchmark" rule; the claim becomes "our model adds
# value on top of the market." USE_ADP=False reproduces the independent model.
USE_ADP = True
B_PARAMS = dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=20,
                l2_regularization=1.0, random_state=0)


# ---------- A: weekly model on preseason rows ----------

def game_specific_columns(cols: list[str]) -> list[str]:
    return [c for c in cols if c in GAME_SPECIFIC or c.startswith("opp_")]


def week1_players(rosters: pl.DataFrame, season: int) -> pl.DataFrame:
    """Everyone the draft board covers: QB/RB/WR/TE on a team's Week 1 roster."""
    return (rosters.filter(pl.col("season") == season, pl.col("week") == 1, pl.col("game_type") == "REG",
                           pl.col("position").is_in(POSITIONS), pl.col("status").is_in(BOARD_STATUSES))
            .select(player_id="gsis_id", player_display_name="full_name", position="position", team="team",
                    draft_number=pl.col("draft_number").cast(pl.Float64))
            .drop_nulls("player_id").unique("player_id", keep="first", maintain_order=True))


def add_placeholder_rows(pw: pl.DataFrame, players: pl.DataFrame, schedules: pl.DataFrame, season: int) -> pl.DataFrame:
    """Give every board player a Week 1 row. Real rows stay as they are; new ones have blank stats."""
    week1 = schedules.filter(pl.col("season") == season, pl.col("week") == 1, pl.col("game_type") == "REG")
    games = pl.concat([
        week1.select("game_id", "gameday", team="home_team", opponent_team="away_team", is_home=pl.lit(True)),
        week1.select("game_id", "gameday", team="away_team", opponent_team="home_team", is_home=pl.lit(False)),
    ])
    have = pw.filter(pl.col("season") == season, pl.col("week") == 1).select("player_id")
    new = (players.join(have, on="player_id", how="anti").join(games, on="team", how="inner")  # bye in Week 1: skipped
           .with_columns(season=pl.lit(season), week=pl.lit(1), left_injured=pl.lit(False)))
    new = new.select([pl.col(c).cast(pw.schema[c]) if c in new.columns else pl.lit(None, dtype=pw.schema[c]).alias(c)
                      for c in pw.columns])
    return pl.concat([pw, new])


def preseason_rows(season: int, players: pl.DataFrame) -> pl.DataFrame:
    """Week 1 feature rows for every board player, built from this season and last season only
    (no feature looks further back; tests/test_season.py checks they match features.parquet)."""
    seasons = [season - 1, season]
    pw = pl.read_parquet(PLAYER_WEEKS_PATH).filter(pl.col("season").is_in(seasons))
    schedules = read_schedules().filter(pl.col("season").is_in(seasons))
    pbp = read_pbp(seasons, columns=PBP_COLS).join(schedules.select("game_id", "season", "week"), on="game_id", how="left")
    pw = add_placeholder_rows(pw, players, schedules, season)
    feats = build_all(pw, schedules, pbp.drop("season", "week"))
    return feats.filter(pl.col("season") == season, pl.col("week") == 1).join(players.select("player_id"), on="player_id")


def neutralize(rows: pl.DataFrame, train: pl.DataFrame) -> pl.DataFrame:
    """Opponent/game inputs -> what a typical Week 1 training row has.

    Averages come from Week 1 training rows only. A column that's blank in Week 1 (e.g. the
    opponent's last-5-games rating: no games yet) stays blank. Filling it with an all-season
    average gave the model a combination it never trained on and cut Josh Allen's 2024
    projection from 16.5 to 11.6 points per game.
    Booleans (is_home, is_dome) become their average as a number (e.g. 0.5 home): the model reads
    every non-text column as a float anyway, and casting 0.5 back to a boolean would make it True.
    """
    week1 = train.filter(pl.col("week") == 1)
    out = []
    for c in game_specific_columns(feature_columns(rows)):
        numeric = rows.schema[c].is_numeric() or rows.schema[c] == pl.Boolean
        if not numeric or week1[c].null_count() > week1.height / 2:
            out.append(pl.lit(None, dtype=rows.schema[c]).alias(c))
        else:
            out.append(pl.lit(week1[c].cast(pl.Float64).mean(), dtype=pl.Float64).alias(c))
    return rows.with_columns(out)


def expected_games(season: int, positions: pl.Series) -> np.ndarray:
    """Team games minus the position-average injury misses from earlier seasons (SCRUM-51)."""
    _, table = availability.load_inputs(range(availability.FIRST_SEASON, min(season, EXAM_SEASON)))
    model = availability.CHOSEN().fit(table.filter(pl.col("season") < min(season, EXAM_SEASON)))
    return 17 - model.predict(pl.DataFrame({"position": positions}))


def a_projection(season: int, scoring=SCORING) -> pl.DataFrame:
    """Option A for one season: preseason rows + per-game points from the weekly model."""
    players = week1_players(pl.read_parquet(rosters_path(season)), season)
    rows = preseason_rows(season, players)
    history = pl.read_parquet(FEATURES_PATH).filter(pl.col("season") < min(season, EXAM_SEASON))
    model = BoostingModel(tuned=TUNED_PATH, name="boosting").fit(history)
    per_game = model.predict(neutralize(rows, history))  # row-aligned with rows
    a_ppg = add_fantasy_points(per_game.drop("player_id").with_columns(position=rows["position"]), scoring)["points"]
    # Per-game stat projections (pg_*) are kept so a league's own scoring can be applied later
    # (src/draft/vor.py). They are NOT season-model inputs (see b_inputs).
    stats = per_game.drop("player_id").rename(lambda c: f"pg_{c}")
    return (rows.join(players.select("player_id", "draft_number"), on="player_id", how="left")
            .with_columns(stats)
            .with_columns(a_ppg=a_ppg, a_exp_games=pl.Series(expected_games(season, rows["position"])))
            .with_columns(a_season=pl.col("a_ppg") * pl.col("a_exp_games")))


# ---------- B: season-level model ----------

def season_results(pw: pl.DataFrame) -> pl.DataFrame:
    """Per player-season: games played and PPR points in the regular season."""
    reg = pw.filter(pl.col("week") <= pl.when(pl.col("season") <= 2020).then(17).otherwise(18))
    return reg.group_by("player_id", "season").agg(games=pl.len(), points=pl.col("fantasy_points_ppr").sum())


def week1_starters(schedules: pl.DataFrame) -> pl.DataFrame:
    w1 = schedules.filter(pl.col("week") == 1, pl.col("game_type") == "REG")
    return pl.concat([w1.select("season", player_id="home_qb_id"), w1.select("season", player_id="away_qb_id")]) \
        .drop_nulls().unique().with_columns(week1_starter=pl.lit(1.0))


def adp_inputs(seasons: list[int]) -> pl.DataFrame:
    """player_id, season, adp (overall average pick), adp_pos_rank (rank within position)."""
    from src.ingest.consensus import ADP_DIR, load_adp
    frames = []
    for s in seasons:
        if not (ADP_DIR / f"fantasypros_adp_{s}.csv").exists():
            continue
        a = load_adp(s).drop_nulls("player_id").unique("player_id", keep="first")
        frames.append(a.select("player_id", "adp", season=pl.lit(s))
                      .with_columns(adp_pos_rank=pl.col("adp").rank("ordinal").over(a["position"])))
    schema = {"player_id": pl.String, "adp": pl.Float64, "season": pl.Int32, "adp_pos_rank": pl.Float64}
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame(schema=schema)


def season_dataset(a_rows: pl.DataFrame, pw: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    """One row per board player-season: B's inputs + the target (actual season points, 0 if none)."""
    res = season_results(pw)
    lag = lambda k, sfx: res.select("player_id", pl.col("season") + k, **{f"games{sfx}": "games",  # noqa: E731
                                                                         f"ppg{sfx}": pl.col("points") / pl.col("games")})
    prev_cols = [c for c in a_rows.columns if c.endswith("_prev") and not c.startswith("opp_")]
    # Inputs added after these rows were cached come in blank until the rows are rebuilt (--rebuild).
    a_rows = a_rows.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in ROW_INPUTS if c not in a_rows.columns])
    return (a_rows.select("player_id", "player_display_name", "position", "team", "season", "draft_number",
                          "a_ppg", "a_exp_games", "a_season", *ROW_INPUTS, *prev_cols,
                          *[c for c in a_rows.columns if c.startswith("pg_")])
            .join(lag(1, "_last"), on=["player_id", "season"], how="left")
            .join(lag(2, "_2ago"), on=["player_id", "season"], how="left")
            .join(week1_starters(schedules), on=["player_id", "season"], how="left")
            .join(res.select("player_id", "season", actual="points"), on=["player_id", "season"], how="left")
            .with_columns(pl.col("games_last", "games_2ago").fill_null(0), pl.col("week1_starter").fill_null(0.0),
                          pl.col("actual").fill_null(0.0),
                          naive=pl.col("ppg_last").fill_null(0) * 17)
            .pipe(lambda d: d.join(adp_inputs(d["season"].unique().to_list())
                                   .with_columns(pl.col("season").cast(d.schema["season"]), pl.col("adp_pos_rank").cast(pl.Float64)),
                                   on=["player_id", "season"], how="left") if USE_ADP else d))


def b_inputs(df: pl.DataFrame) -> list[str]:
    skip = {"player_id", "player_display_name", "team", "season", "actual", "naive", "a_season", "a_exp_games"}
    return [c for c in df.columns if c not in skip and not c.startswith("pg_")]  # pg_*: for league scoring only


class SeasonModel:
    """Gradient boosting on season-level rows; position is a category."""

    def fit(self, train: pl.DataFrame) -> "SeasonModel":
        # Columns with no values in training (two-seasons-ago when training on 2021 alone) are dropped.
        self.cols = [c for c in b_inputs(train) if train[c].null_count() < train.height]
        self.model = HistGradientBoostingRegressor(categorical_features="from_dtype", **B_PARAMS).fit(
            self._x(train), train["actual"].to_numpy())
        return self

    def _x(self, df: pl.DataFrame):
        x = df.select([pl.col(c).cast(pl.Float64) if c != "position" else pl.col(c) for c in self.cols]).to_pandas()
        x["position"] = pd.Categorical(x["position"], categories=POSITIONS)
        return x

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(self._x(df)), 0, None)


# ---------- evaluation + boards ----------

def draft_pool(s: pl.DataFrame, boards: list[str], n: int) -> pl.DataFrame:
    """Players any board ranks in the top 2n at their position: the ones a draft actually decides between."""
    ids = pl.concat([s.sort(b, descending=True).head(2 * n).select("player_id") for b in boards]).unique()
    return s.join(ids, on="player_id")


POOL_SIZE = {"QB": 24, "RB": 36, "WR": 48, "TE": 18}  # ~2 starters' worth in a 12-team league


def evaluate(results: pl.DataFrame, boards: list[str]) -> pl.DataFrame:
    """Spearman with actual season points, within position, on each season's draft pool."""
    out = []
    for (season, pos), s in results.group_by("season", "position", maintain_order=True):
        pool = draft_pool(s, boards, POOL_SIZE[pos])
        out.append({"season": season, "position": pos, "players": pool.height,
                    **{b: pool.select(pl.corr(b, "actual", method="spearman")).item() for b in boards}})
    return pl.DataFrame(out).sort("season", "position")


def report(scores: pl.DataFrame, boards: list[str]) -> str:
    lines = [f"# Season projections backtest ({date.today()})", "",
             "Walk-forward: each season's boards use earlier seasons only. 2025 untouched. Standard PPR.",
             "Spearman correlation with actual season points, within position, on the draft pool",
             "(players any board ranks top 2x starters: QB 48, RB 72, WR 96, TE 36). Higher = better.", "",
             "| position | " + " | ".join(boards) + " | B - naive [95% CI over seasons x positions] |",
             "| --- " * (len(boards) + 2) + "|"]
    for pos in [*POSITIONS, "ALL"]:
        s = scores if pos == "ALL" else scores.filter(pl.col("position") == pos)
        diff = bootstrap((s["season_b"] - s["naive"]).to_numpy())
        lines.append(f"| {pos} | " + " | ".join(f"{s[b].mean():.3f}" for b in boards) + f" | {diff} |")
    lines += ["", "Per season:", "", "| season | position | players | " + " | ".join(boards) + " |",
              "| --- " * (len(boards) + 3) + "|"]
    for r in scores.iter_rows(named=True):
        lines.append(f"| {r['season']} | {r['position']} | {r['players']} | " + " | ".join(f"{r[b]:.3f}" for b in boards) + " |")
    return "\n".join(lines) + "\n"


def board(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort("season_b", descending=True).with_columns(
        overall_rank=pl.int_range(1, pl.len() + 1),
        pos_rank=pl.col("season_b").rank("ordinal", descending=True).over("position").cast(pl.Int64))


def load_a(season: int, rebuild: bool = False) -> pl.DataFrame:
    if season == EXAM_SEASON:
        raise ValueError("2025 is the final exam: project it once, on purpose, in SCRUM-66")
    path = OUT_DIR / f"a_rows_{season}.parquet"
    if rebuild or not path.exists():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        a_projection(season).write_parquet(path)
    return pl.read_parquet(path)


def season_table(rebuild: bool = False) -> pl.DataFrame:
    """Every non-exam season's board rows with inputs + actual results (shared with src/models/upside.py)."""
    pw, schedules = pl.read_parquet(PLAYER_WEEKS_PATH), read_schedules()
    seasons = [s for s in range(FIRST_SEASON, LIVE_SEASON + 1) if s != EXAM_SEASON]
    return pl.concat([season_dataset(load_a(s, rebuild), pw, schedules) for s in seasons], how="diagonal_relaxed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="rebuild the cached option-A rows")
    args = parser.parse_args()
    data = season_table(args.rebuild)

    results = []
    for season in EVAL_SEASONS:
        train, test = data.filter(pl.col("season") < season), data.filter(pl.col("season") == season)
        results.append(test.with_columns(season_b=pl.Series(SeasonModel().fit(train).predict(test))))
    results = pl.concat(results)
    boards = ["naive", "a_season", "season_b"]
    text = report(evaluate(results, boards), boards)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"season_projections_{date.today()}.md").write_text(text, encoding="utf-8")
    print(text)

    for season in [*EVAL_SEASONS, LIVE_SEASON]:  # boards, each from earlier seasons only
        train = data.filter(pl.col("season") < min(season, EXAM_SEASON))
        test = data.filter(pl.col("season") == season)
        b = board(test.with_columns(season_b=pl.Series(SeasonModel().fit(train).predict(test))))
        b.write_parquet(OUT_DIR / f"board_{season}.parquet")
        print(f"{season} board top 10: " + ", ".join(f"{r['player_display_name']} ({r['position']}{r['pos_rank']})"
                                                     for r in b.head(10).iter_rows(named=True)))


if __name__ == "__main__":
    main()
