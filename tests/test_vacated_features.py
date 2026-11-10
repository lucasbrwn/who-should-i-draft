"""Vacated-opportunity features (SCRUM-38). Each test is one rule we agreed on 2026-11-10."""

import polars as pl
import pytest

from src.features.vacated import build_vacated_features, check_vacated_features

STATS = ["targets", "carries", "receptions", "rushing_yards", "receiving_yards", "attempts", "sacks_suffered",
         "target_share"]


def g(week, player, position="WR", team="CIN", season=2022, target_share=0.0, carries=0, left_injured=False):
    row = {"player_id": player, "position": position, "season": season, "week": week,
           "game_id": f"{season}_{week:02d}_{team}", "team": team, "left_injured": left_injured,
           "on_injury_report": False, "report_status": None, "practice_code": None}
    row.update({s: 0 for s in STATS})
    row.update(target_share=target_share, carries=carries)
    return row


def build(rows):
    return build_vacated_features(pl.DataFrame(rows, schema_overrides={"report_status": pl.String, "practice_code": pl.String,
                                                                        "target_share": pl.Float64}))


def at(feats, player, week):
    return feats.filter((pl.col("player_id") == player) & (pl.col("week") == week)).row(0, named=True)


def test_missing_wr1_vacates_his_target_share():
    rows = [g(1, "chase", target_share=0.30), g(1, "higgins", target_share=0.20),
            g(2, "chase", target_share=0.30), g(2, "higgins", target_share=0.20),
            g(3, "higgins", target_share=0.35)]                       # chase out in week 3
    week3 = at(build(rows), "higgins", 3)
    assert week3["vacated_tgt_share_wr"] == pytest.approx(0.30)
    assert week3["teammates_out"] == 1


def test_nothing_vacated_when_everyone_plays():
    rows = [g(w, p, target_share=0.2) for w in (1, 2) for p in ("a", "b")]
    assert at(build(rows), "a", 2)["teammates_out"] == 0


def test_week_1_has_nothing_vacated():
    assert at(build([g(1, "a")]), "a", 1)["vacated_tgt_share_total"] == 0


def test_vacated_split_by_missing_players_position():
    rows = [g(1, "wr", "WR", target_share=0.25), g(1, "te", "TE", target_share=0.15), g(1, "x"),
            g(2, "x")]                                               # WR and TE both out
    week2 = at(build(rows), "x", 2)
    assert week2["vacated_tgt_share_wr"] == pytest.approx(0.25)
    assert week2["vacated_tgt_share_te"] == pytest.approx(0.15)
    assert week2["vacated_tgt_share_total"] == pytest.approx(0.40)


def test_vacated_carries_from_missing_rb():
    rows = [g(1, "rb1", "RB", carries=20), g(1, "rb2", "RB", carries=5), g(2, "rb2", "RB", carries=20)]
    assert at(build(rows), "rb2", 2)["vacated_carry_share"] == pytest.approx(0.8)


def test_only_teammates_from_the_last_3_games_count():
    # "cut" last played week 1; by week 5 the team's last 3 games are weeks 2-4, so he's not vacated anymore.
    rows = [g(1, "cut", target_share=0.1)] + [g(w, "x") for w in range(1, 6)]
    feats = build(rows)
    assert at(feats, "x", 2)["teammates_out"] == 1
    assert at(feats, "x", 5)["teammates_out"] == 0


def test_traded_player_is_vacated_for_old_team_only():
    rows = [g(1, "traded", team="CIN", target_share=0.2), g(1, "x", team="CIN"), g(1, "y", team="KC"),
            g(2, "traded", team="KC"), g(2, "x", team="CIN"), g(2, "y", team="KC")]
    feats = build(rows)
    assert at(feats, "x", 2)["vacated_tgt_share_wr"] == pytest.approx(0.2)
    assert at(feats, "y", 2)["teammates_out"] == 0


def test_vacated_share_ignores_the_missing_players_left_injured_game():
    # chase: normal 0.30 in week 1, left injured in week 2 (0.02); out week 3. Vacated = 0.30, not the average.
    rows = [g(1, "chase", target_share=0.30), g(2, "chase", target_share=0.02, left_injured=True),
            g(1, "x"), g(2, "x"), g(3, "x")]
    assert at(build(rows), "x", 3)["vacated_tgt_share_wr"] == pytest.approx(0.30)


def test_check_passes_on_clean_data():
    rows = [g(w, p, target_share=0.2) for w in (1, 2, 3) for p in ("a", "b")]
    pw = pl.DataFrame(rows, schema_overrides={"report_status": pl.String, "practice_code": pl.String})
    assert check_vacated_features(build_vacated_features(pw), pw) == []
