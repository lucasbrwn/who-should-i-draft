"""Preseason grade + offseason change flags (SCRUM-88). Rules agreed 2026-11-10."""

import polars as pl

from src.features.preseason import build_preseason_features

WT_SCHEMA = {"season": pl.Int32, "team": pl.String, "win_total": pl.Float64}


def game(season, week, home="DET", away="GB", home_coach="Campbell", away_coach="LaFleur",
         home_qb="goff", away_qb="love", home_score=24, away_score=20):
    return {"game_id": f"{season}_{week:02d}_{away}_{home}", "season": season, "week": week,
            "home_team": home, "away_team": away, "home_coach": home_coach, "away_coach": away_coach,
            "home_qb_id": home_qb, "away_qb_id": away_qb, "home_score": home_score, "away_score": away_score}


def pw_row(player, season, week, team="DET", home="DET", away="GB"):
    return {"player_id": player, "season": season, "week": week, "team": team,
            "game_id": f"{season}_{week:02d}_{away}_{home}"}


def build(games, rows, win_totals=None):
    wt = pl.DataFrame(win_totals or [], schema=WT_SCHEMA)
    return build_preseason_features(pl.DataFrame(rows), pl.DataFrame(games), wt)


def det(feats, season, week, player="a"):
    return feats.filter((pl.col("player_id") == player) & (pl.col("season") == season) & (pl.col("week") == week)).row(0, named=True)


def test_new_head_coach_vs_last_seasons_final_coach():
    games = [game(2023, 17, home_coach="Old"), game(2024, 1, home_coach="New")]
    feats = build(games, [pw_row("a", 2023, 17), pw_row("a", 2024, 1)])
    assert det(feats, 2024, 1)["new_head_coach"] is True


def test_same_coach_is_not_new():
    games = [game(2023, 17), game(2024, 1)]
    assert det(build(games, [pw_row("a", 2023, 17), pw_row("a", 2024, 1)]), 2024, 1)["new_head_coach"] is False


def test_first_season_has_no_change_flags():
    feats = build([game(2020, 1)], [pw_row("a", 2020, 1)])
    assert det(feats, 2020, 1)["new_head_coach"] is None and det(feats, 2020, 1)["new_starting_qb"] is None


def test_new_starting_qb_vs_last_seasons_most_used():
    games = [game(2023, w, home_qb="old") for w in (1, 2, 3)] + [game(2023, 4, home_qb="backup"), game(2024, 1, home_qb="rookie")]
    feats = build(games, [pw_row("a", 2023, 1), pw_row("a", 2024, 1)])
    assert det(feats, 2024, 1)["new_starting_qb"] is True


def test_backup_qb_flag_uses_only_earlier_starts():
    games = [game(2023, 1, home_qb="goff"), game(2023, 2, home_qb="goff"), game(2023, 3, home_qb="backup")]
    feats = build(games, [pw_row("a", 2023, w) for w in (1, 2, 3)])
    assert det(feats, 2023, 2)["backup_qb_starting"] is False
    assert det(feats, 2023, 3)["backup_qb_starting"] is True


def test_unknown_starter_leaves_backup_flag_blank():
    games = [game(2023, 1), game(2023, 2, home_qb=None)]
    assert det(build(games, [pw_row("a", 2023, 1), pw_row("a", 2023, 2)]), 2023, 2)["backup_qb_starting"] is None


def test_player_new_team_and_rookie():
    games = [game(2023, 17), game(2024, 1)]
    rows = [pw_row("vet", 2023, 17, team="GB"), pw_row("vet", 2024, 1, team="DET"), pw_row("rookie", 2024, 1)]
    feats = build(games, rows)
    assert det(feats, 2024, 1, "vet")["player_new_team"] is True
    assert det(feats, 2024, 1, "rookie")["no_games_last_season"] is True
    assert det(feats, 2024, 1, "rookie")["player_new_team"] is None


def test_win_total_and_change_from_last_seasons_wins():
    games = [game(2023, w, home_score=30, away_score=10) for w in (1, 2, 3)] + [game(2024, 1)]
    feats = build(games, [pw_row("a", 2023, 1), pw_row("a", 2024, 1)],
                  win_totals=[{"season": 2024, "team": "DET", "win_total": 10.5}])
    week1 = det(feats, 2024, 1)
    assert week1["win_total"] == 10.5 and week1["win_total_change"] == 10.5 - 3


def test_win_totals_missing_leaves_columns_blank():
    feats = build([game(2024, 1)], [pw_row("a", 2024, 1)])
    assert det(feats, 2024, 1)["win_total"] is None
