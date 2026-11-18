"""Live predictions for an upcoming week (SCRUM-73/74/76).

Steps:
    1. upcoming_rows   one blank player-week row per ACTIVE skill player on this week's roster
                       (no stats yet), with that week's opponent, home/away, and injury report
                       if it's out. Players on IR / practice squad get no row.
    2. build_all       the normal feature pipeline over history + those rows. Every feature only
                       looks at earlier games (tests/test_leakage.py), so blank rows are safe.
    3. BoostingModel   tuned weekly model, trained on every finished game before this week.
                       (2025 was held out until the final exam; it ran 2026-11-18, so v2 trains on it.)
    4. Scenarios       teammates of key players who might sit also get a "he's out" projection
                       (proj_without), shown as info. Blending it into proj FAILED its 2024 test
                       (src/backtest/scenario_test.py), so proj itself is unchanged.
    5. Calibration     per-position tier calibration of "if he plays" points (src/models/weekly_calibration.py)
    6. Ranges          likely range = 25th-75th percentile outcome (src/models/ranges.py, BucketRanges
                       with LIKELY), fit on the 2022-24 walk-forward backtest. Inside ~half the time.
"""

from pathlib import Path

import polars as pl

from src.backtest.harness import points
from src.features.assemble import PBP_COLS, build_all
from src.features.team import read_schedules
from src.ingest.pbp import read_pbp
from src.ingest.player_weeks import INJURIES_DIR, attach_injury_report, read_dir
from src.models.boosting import TUNED_PATH, BoostingModel
from src.models.play_chance import PRE_GAME_ACTIVE, PlayChance, apply, game_day_pool
from src.models.ranges import LIKELY, BucketRanges, load_backtest
from src.models.weekly_calibration import WeeklyCalibration, load_history

PW_PATH = Path("data/processed/player_weeks.parquet")
ROSTERS_DIR = Path("data/raw/rosters")
DEMO_PATH = Path("data/reference/demo_players.csv")
POSITIONS = ["QB", "RB", "WR", "TE"]


def week_games(schedules: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """One row per team playing this week: game_id, opponent, home/away, kickoff."""
    games = schedules.filter((pl.col("season") == season) & (pl.col("week") == week))
    cols = ["game_id", "gameday"]
    home = games.select(*cols, team=pl.col("home_team"), opponent_team=pl.col("away_team"), is_home=pl.lit(True))
    away = games.select(*cols, team=pl.col("away_team"), opponent_team=pl.col("home_team"), is_home=pl.lit(False))
    return pl.concat([home, away])


def upcoming_rows(rosters: pl.DataFrame, schedules: pl.DataFrame, injuries: pl.DataFrame,
                  season: int, week: int) -> pl.DataFrame:
    """Blank player-week rows for active skill players whose team plays this week. Past weeks' rosters
    are game-day snapshots (inactives = INA), so INA counts as active: otherwise a backtest of a past
    week would quietly drop the players who ended up sitting."""
    active = rosters.filter(
        (pl.col("season") == season) & (pl.col("week") == week) & pl.col("status").is_in(PRE_GAME_ACTIVE)
        & pl.col("position").is_in(POSITIONS) & pl.col("gsis_id").is_not_null()
    ).select(
        pl.col("gsis_id").alias("player_id"), pl.col("full_name").alias("player_display_name"),
        "position", "team",
    ).unique("player_id")
    rows = active.join(week_games(schedules, season, week), on="team", how="inner").with_columns(
        season=pl.lit(season, pl.Int32), week=pl.lit(week, pl.Int32),
        snaps_only=pl.lit(False), left_injured=pl.lit(False),
    )
    return attach_injury_report(rows, injuries)


def train_rows(feats: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """Every finished game before this week (2025 included since the final exam ran, 2026-11-18)."""
    before = (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week))
    # sorted: boosting bins features from a row sample, so row order must be fixed for repeatable picks
    return feats.filter(before).sort("season", "week", "player_id")


def play_chance_seasons(season: int) -> list[int]:
    """Earlier seasons for the play-chance rates and tier calibration (2025 included since the exam)."""
    return list(range(2021, season))


UNCERTAIN = ["Questionable", "Doubtful", "Out"]
KEY_ROLE_SNAPS = 0.5   # "key player": depth-chart starter or on the field for half the snaps lately


def predict_if_plays(model: BoostingModel, feats: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """player_id -> projected PPR points IF he plays, for every row of that week."""
    rows = feats.filter((pl.col("season") == season) & (pl.col("week") == week)).sort("player_id")
    stats = rows.select("player_id", "position").join(model.predict(rows), on="player_id", how="left")
    return rows.select("player_id").with_columns(proj=points(stats.drop("player_id")))


def key_absences(rows: pl.DataFrame, chance: PlayChance) -> pl.DataFrame:
    """Key players who might sit: on the injury report (Q/D/Out) AND a starter or big-snap player.
    Returns player_id, team, p_play."""
    key = rows.filter(pl.col("report_status").is_in(UNCERTAIN)
                      & ((pl.col("depth_rank") == 1) | (pl.col("offense_pct_l3") >= KEY_ROLE_SNAPS)))
    return chance.rates(key.select("player_id", "team", "position", "depth_rank", "report_status", "practice_code")
                        ).select("player_id", "team", "p_play")


def project_rows(hist: pl.DataFrame, upcoming: pl.DataFrame, schedules: pl.DataFrame, pbp: pl.DataFrame,
                 season: int, week: int, chance: PlayChance, scenarios: bool = True,
                 model: BoostingModel | None = None) -> pl.DataFrame:
    """Projections for every upcoming row: proj (points if he plays) and, with scenarios, the blend
    for teammates of key players who might sit (src docstring of scenario_blend)."""
    feats = build_all(pl.concat([hist, upcoming], how="diagonal_relaxed"), schedules, pbp)
    if model is None:
        model = BoostingModel(tuned=TUNED_PATH, name="boosting").fit(train_rows(feats, season, week))
    rows = feats.filter((pl.col("season") == season) & (pl.col("week") == week)).sort("player_id")
    out = rows.select("player_id", "player_display_name", "position", "team", "opponent_team", "game_id",
                      "depth_rank", "report_status", "practice_code").join(
        predict_if_plays(model, feats, season, week), on="player_id")
    if not scenarios:
        return out.with_columns(proj_with=pl.col("proj"), proj_without=pl.col("proj"), absent_chance=0.0)
    absent = key_absences(rows, chance)
    if absent.is_empty():
        return out.with_columns(proj_with=pl.col("proj"), proj_without=pl.col("proj"), absent_chance=0.0)
    # The "without" world: those players have no row, so teammates' vacated-opportunity features
    # credit them with the missing work (same logic the model learned from real absences).
    feats_wo = build_all(pl.concat([hist, upcoming.join(absent.select("player_id"), on="player_id", how="anti")],
                                   how="diagonal_relaxed"), schedules, pbp)
    without = predict_if_plays(model, feats_wo, season, week).rename({"proj": "proj_without"})
    return scenario_blend(out, without, absent)


def scenario_blend(out: pl.DataFrame, without: pl.DataFrame, absent: pl.DataFrame) -> pl.DataFrame:
    """Teammates of key players who might sit:
        proj = q x proj(they all play) + (1 - q) x proj(they're out)
    q = chance they ALL play (product of their chances; an Out player makes q = 0, which also fixes
    teammates of ruled-out starters, who used to be projected as if he'd play). Treating "not all play"
    as "all out" is an approximation; it's exact when a team has one key player in doubt."""
    q = absent.group_by("team").agg(q=pl.col("p_play").product())
    out = (out.rename({"proj": "proj_with"}).join(without, on="player_id", how="left").join(q, on="team", how="left")
           .with_columns(is_absent=pl.col("player_id").is_in(absent["player_id"].implode())))
    blend = pl.col("q").is_not_null() & ~pl.col("is_absent") & pl.col("proj_without").is_not_null()
    return out.with_columns(
        proj=pl.when(blend).then(pl.col("q") * pl.col("proj_with") + (1 - pl.col("q")) * pl.col("proj_without"))
        .otherwise(pl.col("proj_with")),
        proj_without=pl.when(blend).then(pl.col("proj_without")).otherwise(pl.col("proj_with")),
        absent_chance=pl.when(blend).then(1 - pl.col("q")).otherwise(0.0),
    ).drop("q", "is_absent")


def project_week(season: int, week: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """All active players' projections for the week, plus that week's games with Vegas lines."""
    pw = pl.read_parquet(PW_PATH)
    seasons = sorted(pw["season"].unique().to_list())
    schedules = read_schedules().filter(pl.col("season").is_in(seasons))
    pbp = read_pbp(seasons, columns=PBP_COLS).join(
        schedules.select("game_id", "season", "week"), on="game_id", how="left")
    rosters = read_dir(ROSTERS_DIR, "rosters", [season])
    injuries = read_dir(INJURIES_DIR, "injuries", [season])

    hist = pw.filter((pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week)))
    upcoming = upcoming_rows(rosters, schedules, injuries, season, week)
    chance = PlayChance().fit(game_day_pool(play_chance_seasons(season), schedules))
    # Scenario blending FAILED its test (docs/reports/scenario_test_2026-11-17.md), so the projection
    # stays "as if teammates play" (proj_with). proj_without is kept as display info only:
    # "if his injured starter sits, he projects X".
    preds = project_rows(hist, upcoming, schedules, pbp, season, week, chance)
    # tier calibration (passed its test 2026-11-17): reshape "if he plays" points by position using
    # earlier seasons' mistakes; never reorders players at a position
    cal = WeeklyCalibration().fit(load_history(play_chance_seasons(season)))
    pos = preds["position"].to_numpy()
    preds = preds.with_columns(
        proj=pl.Series(cal.apply(preds["proj_with"].to_numpy(), pos)).round(1),
        proj_without=pl.Series(cal.apply(preds["proj_without"].to_numpy(), pos)).round(1),
        absent_chance=pl.col("absent_chance").round(2)).drop("proj_with")
    # likely range = 25th-75th percentile outcome (he lands inside about half the time)
    preds = BucketRanges(LIKELY).fit(load_backtest()).predict(preds).select(
        pl.exclude("bust", "ceiling"), likely_low=pl.col("bust").round(1), likely_high=pl.col("ceiling").round(1))
    # projection so far = points IF he plays; expected points = that x chance he plays (src/models/play_chance.py)
    preds = apply(preds, chance)

    games = schedules.filter((pl.col("season") == season) & (pl.col("week") == week)).select(
        "game_id", "gameday", "gametime", "away_team", "home_team", "spread_line", "total_line").sort("gameday", "gametime")
    return preds.sort("proj", descending=True), games


def demo_table(preds: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """The 10 demo players. Anyone without a projection gets the reason (roster status, e.g. RES = IR)."""
    demo = pl.read_csv(DEMO_PATH)
    rosters = read_dir(ROSTERS_DIR, "rosters", [season]).filter(pl.col("week") == week)
    status = rosters.select(pl.col("gsis_id").alias("player_id"), pl.col("team").alias("roster_team"),
                            pl.col("status").alias("roster_status")).unique("player_id")
    out = demo.join(preds.drop("position", "team", "player_display_name"), on="player_id", how="left")
    out = out.join(status, on="player_id", how="left")
    team_changed = out.filter(pl.col("roster_team").is_not_null() & (pl.col("roster_team") != pl.col("team")))
    if team_changed.height:
        raise ValueError(f"demo_players.csv team is out of date: {team_changed.select('name', 'team', 'roster_team').rows()}")
    return out.drop("roster_team")
