"""In-game injury parsing (SCRUM-22 step 3) and the injury columns on player_weeks (steps 2-3)."""

import polars as pl

from src.ingest.in_game_injuries import match_players, parse_mentions, per_player_game
from src.ingest.player_weeks import attach_in_game_injuries, attach_injury_report


def plays(*descs, week=1):
    return pl.DataFrame([{"game_id": "2023_01_NYJ_BUF", "play_id": i, "season": 2023, "week": week,
                          "qtr": 1, "desc": d} for i, d in enumerate(descs)])


def roster(*players):
    return pl.DataFrame([{"season": 2023, "week": 1, "team": t, "jersey_number": n, "gsis_id": g,
                          "first_name": f, "last_name": l, "position": p} for t, n, g, f, l, p in players])


# --- Parsing play text ------------------------------------------------------------------------

def test_parses_team_number_and_name():
    m = parse_mentions(plays("(4:12) 8-A.Rodgers sacked. NYJ-8-A.Rodgers was injured during the play."))
    assert m.select("team", "jersey_number", "abbr_name").to_dicts() == [
        {"team": "NYJ", "jersey_number": 8, "abbr_name": "A.Rodgers"}]


def test_two_injuries_on_one_play():
    m = parse_mentions(plays("... BUF-7-Ta.Johnson was injured during the play. ARI-73-J.Williams was injured during the play."))
    assert m["team"].to_list() == ["BUF", "ARI"]


def test_plays_without_injuries_are_ignored():
    assert parse_mentions(plays("1-K.Murray pass short middle to 4-G.Dortch for 8 yards.")).height == 0


def test_two_letter_team_codes():
    # the Rams are "LA" in nflverse
    assert parse_mentions(plays("LA-10-C.Kupp was injured during the play."))["team"].item() == "LA"


# --- Matching to players ----------------------------------------------------------------------

def test_unique_number_matches():
    m = match_players(parse_mentions(plays("NYJ-8-A.Rodgers was injured during the play.")),
                      roster(("NYJ", 8, "rodgers", "Aaron", "Rodgers", "QB")))
    assert m["gsis_id"].to_list() == ["rodgers"]


def test_shared_number_uses_the_name():
    # offense and defense can both wear #7; "Ta.Johnson" must pick Taron Johnson, not the other #7
    m = match_players(parse_mentions(plays("BUF-7-Ta.Johnson was injured during the play.")),
                      roster(("BUF", 7, "taron", "Taron", "Johnson", "CB"), ("BUF", 7, "other", "Mike", "Smith", "WR")))
    assert m["gsis_id"].to_list() == ["taron"]


def test_shared_number_and_no_name_match_stays_unmatched():
    m = match_players(parse_mentions(plays("BUF-7-Z.Nobody was injured during the play.")),
                      roster(("BUF", 7, "a", "Mike", "Smith", "WR"), ("BUF", 7, "b", "Joe", "Jones", "CB")))
    assert m["gsis_id"].to_list() == [None]   # don't guess


def test_per_player_game_counts_mentions_and_first_quarter():
    m = pl.DataFrame({"game_id": ["g", "g"], "gsis_id": ["p", "p"], "qtr": [3, 2], "play_id": [1, 2]})
    assert per_player_game(m).to_dicts() == [{"game_id": "g", "player_id": "p", "injury_mentions": 2, "injury_qtr": 2}]


# --- left_injured flag ------------------------------------------------------------------------

def games(pcts, mentions_in_week=None):
    rows = pl.DataFrame({"player_id": ["p"] * len(pcts), "season": [2023] * len(pcts),
                         "week": list(range(1, len(pcts) + 1)), "game_id": [f"g{w}" for w in range(1, len(pcts) + 1)],
                         "offense_pct": pcts})
    in_game = pl.DataFrame({"player_id": ["p"], "game_id": [f"g{mentions_in_week}"], "injury_mentions": [1],
                            "injury_qtr": [1]}, schema_overrides={"injury_mentions": pl.UInt32})
    return attach_in_game_injuries(rows, in_game if mentions_in_week else in_game.clear())


def test_hurt_and_snaps_collapsed_is_flagged():
    # Rodgers 2023 wk 1 shape: usually 100%, hurt, 7% of snaps
    df = games([1.0, 1.0, 1.0, 0.07], mentions_in_week=4)
    assert df["left_injured"].to_list() == [False, False, False, True]


def test_hurt_but_came_back_is_not_flagged():
    df = games([0.9, 0.9, 0.85], mentions_in_week=3)
    assert df["left_injured"].to_list() == [False, False, False]


def test_low_snaps_without_injury_text_is_not_flagged():
    # benched, blowout, or demoted: not an injury
    assert not games([0.9, 0.9, 0.1]).filter(pl.col("week") == 3)["left_injured"].item()


def test_usual_share_uses_only_earlier_games():
    # week 2's baseline is week 1 only; the later 0.1 weeks must not lower it (no future data)
    df = games([0.8, 0.8, 0.1, 0.1, 0.1])
    assert df.filter(pl.col("week") == 2)["usual_offense_pct"].item() == 0.8
    assert df.filter(pl.col("week") == 1)["usual_offense_pct"].item() is None


def test_first_game_falls_back_to_half_the_snaps():
    assert games([0.3], mentions_in_week=1)["left_injured"].item() is True
    assert games([0.7], mentions_in_week=1)["left_injured"].item() is False


# --- Injury report ----------------------------------------------------------------------------

def test_injury_report_attaches_and_healthy_players_are_false():
    rows = pl.DataFrame({"player_id": ["hurt", "healthy"], "season": [2023, 2023], "week": [1, 1], "team": ["BUF", "BUF"]},
                        schema_overrides={"season": pl.Int32, "week": pl.Int32})
    report = pl.DataFrame([{"gsis_id": "hurt", "season": 2023, "week": 1, "team": "BUF", "game_type": "REG",
                            "report_status": "Questionable", "practice_code": "LP", "report_primary_injury": "Ankle"}])
    out = {r["player_id"]: r for r in attach_injury_report(rows, report).to_dicts()}
    assert out["hurt"]["report_status"] == "Questionable" and out["hurt"]["on_injury_report"] is True
    assert out["healthy"]["on_injury_report"] is False and out["healthy"]["report_status"] is None
