"""Front Office web app (local only).

Run:  flask --app app run --debug     then open http://127.0.0.1:5000
"""

from dataclasses import fields
from pathlib import Path

from flask import Flask, abort, flash, redirect, render_template, request, url_for

from app.lines import upcoming_lines
from src.scoring.config import (
    LEAGUES_DIR, PLANNED_TYPES, PRESETS, SUPPORTED_TYPES, League, LeagueConfig, Roster, Scoring,
    list_leagues, load_league, save_league, slugify,
)

PARTS = (("league", League), ("roster", Roster), ("scoring", Scoring))


def form_sections(config: LeagueConfig, raw: dict | None = None) -> dict:
    """Group every numeric setting for the template. `raw` holds submitted text to redisplay on error.
    Planned settings are left out of the menu until they're supported."""
    raw = raw or {}

    def info(part_name, f):
        key = f"{part_name}.{f.name}"
        value = raw.get(key, getattr(getattr(config, part_name), f.name))
        return {"key": key, "value": value, **f.metadata}

    scoring_groups: dict[str, list] = {}
    for f in fields(Scoring):
        scoring_groups.setdefault(f.metadata["group"], []).append(info("scoring", f))
    return {
        "teams": info("league", next(f for f in fields(League) if f.name == "teams")),
        "roster": [info("roster", f) for f in fields(Roster) if not f.metadata["planned"]],
        "scoring": scoring_groups,
    }


def parse_form(form) -> tuple[LeagueConfig, list[str]]:
    """Build a config from submitted form fields. Unparseable numbers become errors and fall back
    to the default so the rest of the form can still be validated. Planned settings aren't in the
    menu, so they always keep their default."""
    errors, data = [], {"league": {}, "roster": {}, "scoring": {}}
    data["league"]["name"] = form.get("league.name", "").strip()
    data["league"]["type"] = form.get("league.type", "redraft")

    for part_name, part_cls in PARTS:
        for f in fields(part_cls):
            if "min" not in f.metadata or f.metadata["planned"]:
                continue
            text = form.get(f"{part_name}.{f.name}", "").strip()
            number = float if part_name == "scoring" else int
            try:
                data[part_name][f.name] = number(text)
            except ValueError:
                errors.append(f"{f.metadata['label']} must be a {'number' if number is float else 'whole number'}.")

    config = LeagueConfig.from_dict(data)
    return config, errors + config.validate()


def create_app(leagues_dir: Path = LEAGUES_DIR, lines_provider=upcoming_lines) -> Flask:
    app = Flask(__name__)
    app.secret_key = "local-dev-only"

    @app.context_processor
    def scoreboard():
        try:
            return {"lines": lines_provider()}
        except Exception:  # the strip is decoration; never break a page over it
            return {"lines": []}

    def render_form(config, slug=None, errors=(), raw=None, status=200):
        return render_template(
            "league_form.html", config=config, slug=slug, errors=errors,
            sections=form_sections(config, raw), supported=SUPPORTED_TYPES, planned=PLANNED_TYPES,
            presets=PRESETS,
        ), status

    @app.get("/")
    def index():
        return render_template("index.html", leagues=list_leagues(leagues_dir), presets=PRESETS)

    @app.get("/leagues/new")
    def new_league():
        preset = PRESETS.get(request.args.get("preset", "standard-ppr"), PRESETS["standard-ppr"])
        config = LeagueConfig.from_dict(preset.to_dict())
        config.league.name = "My League"
        return render_form(config)

    @app.get("/leagues/<slug>")
    def edit_league(slug):
        path = leagues_dir / f"{slug}.json"
        if not path.exists():
            abort(404)
        return render_form(load_league(path), slug=slug)

    @app.post("/leagues/save")
    def save():
        original = request.form.get("original_slug") or None
        config, errors = parse_form(request.form)
        if errors:
            return render_form(config, slug=original, errors=errors, raw=request.form, status=400)

        new_slug = slugify(config.league.name)
        if new_slug != original and (leagues_dir / f"{new_slug}.json").exists():
            return render_form(config, slug=original, raw=request.form, status=400,
                               errors=[f"A league named '{config.league.name}' already exists."])
        save_league(config, leagues_dir)
        if original and original != new_slug:
            (leagues_dir / f"{original}.json").unlink(missing_ok=True)
        flash(f"Saved {config.league.name} ({config.summary()}).")
        return redirect(url_for("index"))

    @app.post("/leagues/<slug>/delete")
    def delete(slug):
        path = leagues_dir / f"{slug}.json"
        if path.exists():
            name = load_league(path).league.name
            path.unlink()
            flash(f"Deleted {name}.")
        return redirect(url_for("index"))

    return app
