from dataclasses import fields

import pytest

from app import create_app
from src.scoring.config import PRESETS, League, Roster, Scoring, load_league


def form_for(config, **overrides):
    """What a browser submits: every enabled field. Planned (disabled) fields are never sent."""
    data = {"league.name": config.league.name, "league.type": config.league.type}
    for part_name, part_cls in (("league", League), ("roster", Roster), ("scoring", Scoring)):
        for f in fields(part_cls):
            if "min" in f.metadata and not f.metadata["planned"]:
                data[f"{part_name}.{f.name}"] = str(getattr(getattr(config, part_name), f.name))
    data.update(overrides)
    return data


@pytest.fixture
def client(tmp_path):
    app = create_app(leagues_dir=tmp_path, lines_provider=lambda: [])
    app.config["TESTING"] = True
    return app.test_client(), tmp_path


def test_index_lists_presets(client):
    c, _ = client
    page = c.get("/").get_data(as_text=True)
    for preset in PRESETS.values():
        assert preset.league.name in page


def test_new_league_form_prefills_preset(client):
    c, _ = client
    page = c.get("/leagues/new?preset=superflex-half-ppr").get_data(as_text=True)
    assert 'name="roster.SUPERFLEX" value="1"' in page
    assert 'name="scoring.receptions" value="0.5"' in page


def test_save_creates_league_and_edit_renames(client):
    c, leagues = client
    resp = c.post("/leagues/save", data=form_for(PRESETS["half-ppr"], **{"league.name": "Work League", "roster.WR": "3"}))
    assert resp.status_code == 302
    saved = load_league(leagues / "work-league.json")
    assert saved.roster.WR == 3 and saved.scoring.receptions == 0.5

    resp = c.post("/leagues/save", data=form_for(saved, **{"league.name": "Office League", "original_slug": "work-league"}))
    assert resp.status_code == 302
    assert not (leagues / "work-league.json").exists()
    assert (leagues / "office-league.json").exists()


def test_invalid_form_shows_errors_and_keeps_input(client):
    c, leagues = client
    resp = c.post("/leagues/save", data=form_for(PRESETS["standard-ppr"], **{"league.teams": "abc", "roster.QB": "9"}))
    page = resp.get_data(as_text=True)
    assert resp.status_code == 400
    assert "Teams must be a whole number." in page
    assert "QB must be between 0 and 4." in page
    assert 'value="abc"' in page
    assert not list(leagues.glob("*.json"))


def test_planned_fields_are_hidden_and_keep_defaults(client):
    c, leagues = client
    page = c.get("/leagues/new").get_data(as_text=True)
    assert "roster.IDP" not in page
    assert "IDP" not in page

    form = form_for(PRESETS["standard-ppr"], **{"league.name": "No IDP"})
    assert "roster.IDP" not in form
    assert c.post("/leagues/save", data=form).status_code == 302
    assert load_league(leagues / "no-idp.json").roster.IDP == 0


def test_scoreboard_shows_lines_and_survives_errors(tmp_path):
    game = {"week": 4, "away": "NE", "home": "BUF", "favorite": "BUF -5.5", "total": "49.5", "kickoff": "SUN 1:00 PM ET"}
    page = create_app(tmp_path, lines_provider=lambda: [game]).test_client().get("/").get_data(as_text=True)
    assert "Week 4" in page and "BUF -5.5 · O/U 49.5" in page

    def broken():
        raise OSError("snapshot unreadable")
    resp = create_app(tmp_path, lines_provider=broken).test_client().get("/")
    assert resp.status_code == 200 and "scoreboard" not in resp.get_data(as_text=True)


def test_delete(client):
    c, leagues = client
    c.post("/leagues/save", data=form_for(PRESETS["standard-ppr"], **{"league.name": "Temp"}))
    c.post("/leagues/temp/delete")
    assert not (leagues / "temp.json").exists()
