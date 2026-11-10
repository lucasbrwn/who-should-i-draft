"""Assemble the model's feature table (SCRUM-39): every feature group + the targets, one row per player-week.

    data/processed/features.parquet
        id columns     player_id, player_display_name, position, season, week, game_id, team, opponent_team
        left_injured   keep the row, but leave it out when training (a 3%-snap game teaches nothing)
        y_*            what the model predicts: that week's actual stats (design doc section 1)
        everything else  features, all known before kickoff

build_all() runs every builder from in-memory data; the leakage test in
tests/test_leakage.py scrambles the future and rebuilds through it.

Usage:
    python -m src.features          # rebuild all feature tables + features.parquet
"""

from pathlib import Path

import polars as pl

from src.features.defense import PBP_COLS as DEFENSE_PBP_COLS
from src.features.defense import build_defense_features, defense_plays, fantasy_over_expected_allowed
from src.features.matchup import PBP_COLS as MATCHUP_PBP_COLS
from src.features.matchup import build_matchup_features, target_depths
from src.features.team import build_team_features, join_to_player_weeks, team_plays
from src.features.usage import build_usage_features
from src.features.vacated import build_vacated_features

OUT_PATH = Path("data/processed/features.parquet")

KEY = ["player_id", "season", "week"]
ID_COLS = ["player_id", "player_display_name", "position", "season", "week", "game_id", "team", "opponent_team"]
TEAM_PBP_COLS = ["game_id", "season_type", "posteam", "play_type", "pass_oe"]
PBP_COLS = list(dict.fromkeys(TEAM_PBP_COLS + DEFENSE_PBP_COLS + MATCHUP_PBP_COLS))

# Modeled components + rare components (rates) + opportunities + PPR points for reference.
TARGETS = [
    "passing_yards", "passing_tds", "passing_interceptions", "rushing_yards", "rushing_tds",
    "receptions", "receiving_yards", "receiving_tds",
    "passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions",
    "fumbles_lost_total", "special_teams_tds",
    "targets", "carries", "attempts",
    "fantasy_points_ppr",
]


def assemble(pw: pl.DataFrame, groups: list[pl.DataFrame]) -> pl.DataFrame:
    """Join feature groups onto the player-week ids + targets. Groups must be keyed by player_id/season/week."""
    out = pw.select(ID_COLS + ["left_injured"] + [pl.col(t).alias(f"y_{t}") for t in TARGETS])
    taken = set(out.columns)
    for g in groups:
        cols = [c for c in g.columns if c not in taken]
        out = out.join(g.select(KEY + cols), on=KEY, how="left")
        taken.update(cols)
    return out.sort(KEY)


def build_all(pw: pl.DataFrame, schedules: pl.DataFrame, pbp: pl.DataFrame) -> pl.DataFrame:
    usage = build_usage_features(pw)
    team = join_to_player_weeks(pw, build_team_features(schedules, team_plays(pbp)))
    defense = build_defense_features(schedules, defense_plays(pbp), fantasy_over_expected_allowed(pw))
    matchup = build_matchup_features(pw, usage, defense, target_depths(pbp))  # carries the opponent's ratings
    vacated = build_vacated_features(pw)
    return assemble(pw, [usage, team, matchup, vacated])


def feature_columns(feats: pl.DataFrame) -> list[str]:
    return [c for c in feats.columns if c not in ID_COLS and c != "left_injured" and not c.startswith("y_")]


def check_features(feats: pl.DataFrame, pw: pl.DataFrame) -> list[str]:
    problems = []
    if feats.height != pw.height:
        problems.append(f"{feats.height} rows vs {pw.height} player_weeks rows")
    if feats.group_by(KEY).len().filter(pl.col("len") > 1).height:
        problems.append("duplicate player-weeks")
    # A raw same-week stat sneaking in as a feature (e.g. 'targets' with no window suffix) would be leakage.
    raw_stats = set(pw.columns) - set(ID_COLS) - {"on_injury_report", "report_status", "practice_code", "is_home"}
    leaked = sorted(set(feature_columns(feats)) & raw_stats)
    if leaked:
        problems.append(f"same-week stats used as features: {leaked}")
    return problems
