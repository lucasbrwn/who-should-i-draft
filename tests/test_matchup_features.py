"""Player x defense matchup features (SCRUM-37). Each test is one rule we agreed on 2026-11-10."""

import polars as pl
import pytest

from src.features.matchup import build_matchup_features, target_depths

WINDOWS = ["l5", "std", "prev"]


def pw_row(week, player="A", position="WR", team="DET", opp="GB", season=2023, left_injured=False):
    return {"player_id": player, "season": season, "week": week, "game_id": f"{season}_{week:02d}_{team}",
            "team": team, "opponent_team": opp, "position": position, "left_injured": left_injured}


def defense_row(week, defteam="GB", season=2023, **ratings):
    row = {"game_id": f"{season}_{week:02d}_DET", "defteam": defteam}
    for r in ["epa_per_dropback", "epa_per_rush", "epa_deep", "epa_short", "sack_rate",
              "fpoe_qb", "fpoe_rb", "fpoe_wr", "fpoe_te"]:
        for w in WINDOWS:
            row[f"{r}_{w}"] = ratings.get(f"{r}_{w}")
    return row


def usage_row(week, player="A", season=2023, share_std=None, share_prev=None):
    return {"player_id": player, "season": season, "week": week,
            "target_share_std": share_std, "target_share_prev": share_prev}


def depth_row(week, player="A", deep=0, short=0, season=2023):
    return {"game_id": f"{season}_{week:02d}_DET", "player_id": player, "deep_tgts": deep, "short_tgts": short}


def build(pw, defense, usage=None, depths=None):
    pw = pl.DataFrame(pw)
    usage = pl.DataFrame(usage or [usage_row(r["week"], r["player_id"], r["season"]) for r in pw.to_dicts()],
                         schema_overrides={"target_share_std": pl.Float64, "target_share_prev": pl.Float64})
    depths = pl.DataFrame(depths or [], schema={"game_id": pl.String, "player_id": pl.String,
                                                "deep_tgts": pl.UInt32, "short_tgts": pl.UInt32})
    defense = pl.DataFrame(defense, infer_schema_length=None).with_columns(pl.exclude("game_id", "defteam").cast(pl.Float64))
    return build_matchup_features(pw, usage, defense, depths)


def row(feats, week, player="A"):
    return feats.filter((pl.col("week") == week) & (pl.col("player_id") == player)).row(0, named=True)


def test_fantasy_points_allowed_match_the_players_position():
    d = [defense_row(1, fpoe_wr_std=4.0, fpoe_rb_std=-2.0)]
    feats = build([pw_row(1, "A", "WR"), pw_row(1, "B", "RB")], d)
    assert row(feats, 1, "A")["opp_fpoe_pos_std"] == 4.0
    assert row(feats, 1, "B")["opp_fpoe_pos_std"] == -2.0


def test_deep_share_uses_only_earlier_games():
    pw = [pw_row(1), pw_row(2), pw_row(3)]
    depths = [depth_row(1, deep=1, short=3), depth_row(2, deep=3, short=1), depth_row(3, deep=10, short=0)]
    feats = build(pw, [defense_row(w) for w in (1, 2, 3)], depths=depths)
    assert row(feats, 1)["deep_tgt_share_std"] is None
    assert row(feats, 3)["deep_tgt_share_std"] == pytest.approx(4 / 8)   # weeks 1-2 only


def test_left_injured_game_is_skipped_in_deep_share():
    pw = [pw_row(1), pw_row(2, left_injured=True), pw_row(3)]
    depths = [depth_row(1, deep=1, short=1), depth_row(2, deep=0, short=0)]
    feats = build(pw, [defense_row(w) for w in (1, 2, 3)], depths=depths)
    assert row(feats, 3)["deep_tgt_share_std"] == pytest.approx(0.5)


def test_deep_matchup_multiplies_player_and_defense():
    pw = [pw_row(1), pw_row(2)]
    feats = build(pw, [defense_row(1), defense_row(2, epa_deep_std=0.5, epa_short_std=0.1)],
                  depths=[depth_row(1, deep=1, short=3)])
    week2 = row(feats, 2)
    assert week2["deep_matchup"] == pytest.approx(0.25 * 0.5)
    assert week2["short_matchup"] == pytest.approx(0.75 * 0.1)


def test_week_1_falls_back_to_last_season():
    pw = [pw_row(17, season=2022), pw_row(1, season=2023)]
    d = [defense_row(17, season=2022), defense_row(1, epa_deep_prev=0.4)]
    feats = build(pw, d, depths=[depth_row(17, deep=1, short=1, season=2022)])
    week1 = feats.filter(pl.col("season") == 2023).row(0, named=True)
    assert week1["deep_tgt_share_std"] is None
    assert week1["deep_matchup"] == pytest.approx(0.5 * 0.4)


def test_rb_checkdown_only_for_running_backs():
    d = [defense_row(1, sack_rate_std=0.08)]
    usage = [usage_row(1, "A", share_std=0.15), usage_row(1, "B", share_std=0.25)]
    feats = build([pw_row(1, "A", "RB"), pw_row(1, "B", "WR")], d, usage=usage)
    assert row(feats, 1, "A")["rb_checkdown"] == pytest.approx(0.15 * 0.08)
    assert row(feats, 1, "B")["rb_checkdown"] is None


def test_target_depths_ignore_sacks_and_non_targets():
    pbp = pl.DataFrame([
        {"game_id": "g", "season_type": "REG", "pass_attempt": 1, "sack": 0, "receiver_player_id": "A", "pass_length": "deep"},
        {"game_id": "g", "season_type": "REG", "pass_attempt": 1, "sack": 0, "receiver_player_id": "A", "pass_length": "short"},
        {"game_id": "g", "season_type": "REG", "pass_attempt": 1, "sack": 1, "receiver_player_id": None, "pass_length": None},
        {"game_id": "g", "season_type": "POST", "pass_attempt": 1, "sack": 0, "receiver_player_id": "A", "pass_length": "deep"},
    ])
    out = target_depths(pbp).row(0, named=True)
    assert (out["deep_tgts"], out["short_tgts"]) == (1, 1)
