"""League configuration: league type, roster slots, and scoring rules.

Every module (scoring engine, VOR, draft simulation, frontend) reads this one definition.
Each setting's label, range, and grouping live in field metadata, so the settings menu is
generated from it and a new setting only needs to be added here.

Scoring keys match nflverse weekly stat columns, so the scoring engine can apply them directly.
"""

import json
import math
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEAGUES_DIR = PROJECT_ROOT / "data" / "leagues"

SUPPORTED_TYPES = {"redraft": "Redraft", "bestball": "Best Ball"}
PLANNED_TYPES = {"chop": "Chop / Guillotine", "dynasty": "Dynasty"}

SLOT_ELIGIBILITY = {
    "QB": {"QB"},
    "RB": {"RB"},
    "WR": {"WR"},
    "TE": {"TE"},
    "FLEX": {"RB", "WR", "TE"},
    "SUPERFLEX": {"QB", "RB", "WR", "TE"},
    "K": {"K"},
    "DST": {"DST"},
    "IDP": {"DL", "LB", "DB"},
    "BENCH": set(),
}


def setting(default, label, lo, hi, step=1, group=None, help=""):
    return field(default=default, metadata={
        "label": label, "min": lo, "max": hi, "step": step, "group": group, "help": help,
    })


@dataclass
class League:
    name: str = "My League"
    type: str = "redraft"
    teams: int = setting(12, "Teams", 4, 32)


@dataclass
class Roster:
    QB: int = setting(1, "QB", 0, 4)
    RB: int = setting(2, "RB", 0, 6)
    WR: int = setting(2, "WR", 0, 6)
    TE: int = setting(1, "TE", 0, 4)
    FLEX: int = setting(1, "Flex", 0, 4, help="RB / WR / TE")
    SUPERFLEX: int = setting(0, "Superflex", 0, 3, help="QB / RB / WR / TE")
    K: int = setting(1, "Kicker", 0, 2)
    DST: int = setting(1, "Defense / ST", 0, 2)
    IDP: int = setting(0, "IDP", 0, 0, help="Individual defensive players: planned for a later version")
    BENCH: int = setting(6, "Bench", 0, 30)

    @property
    def starters(self) -> int:
        return sum(getattr(self, f.name) for f in fields(self) if f.name != "BENCH")

    @property
    def size(self) -> int:
        return self.starters + self.BENCH


@dataclass
class Scoring:
    passing_yards: float = setting(0.04, "Points per passing yard", 0, 1, 0.01, "Passing", "0.04 = 1 pt per 25 yds")
    passing_tds: float = setting(4.0, "Passing TD", 0, 12, 0.5, "Passing")
    passing_interceptions: float = setting(-2.0, "Interception thrown", -10, 0, 0.5, "Passing")
    passing_2pt_conversions: float = setting(2.0, "2-pt conversion (pass)", 0, 6, 0.5, "Passing")
    bonus_pass_300: float = setting(0.0, "Bonus: 300+ passing yds", 0, 10, 0.5, "Passing")

    rushing_yards: float = setting(0.1, "Points per rushing yard", 0, 1, 0.01, "Rushing", "0.1 = 1 pt per 10 yds")
    rushing_tds: float = setting(6.0, "Rushing TD", 0, 12, 0.5, "Rushing")
    rushing_2pt_conversions: float = setting(2.0, "2-pt conversion (rush)", 0, 6, 0.5, "Rushing")
    bonus_rush_100: float = setting(0.0, "Bonus: 100+ rushing yds", 0, 10, 0.5, "Rushing")

    receptions: float = setting(1.0, "Points per reception", 0, 2, 0.25, "Receiving", "1 = PPR, 0.5 = Half-PPR, 0 = Standard")
    te_reception_bonus: float = setting(0.0, "TE premium (extra per TE catch)", 0, 2, 0.25, "Receiving")
    receiving_yards: float = setting(0.1, "Points per receiving yard", 0, 1, 0.01, "Receiving")
    receiving_tds: float = setting(6.0, "Receiving TD", 0, 12, 0.5, "Receiving")
    receiving_2pt_conversions: float = setting(2.0, "2-pt conversion (catch)", 0, 6, 0.5, "Receiving")
    bonus_rec_100: float = setting(0.0, "Bonus: 100+ receiving yds", 0, 10, 0.5, "Receiving")

    fumbles_lost: float = setting(-2.0, "Fumble lost", -10, 0, 0.5, "Other")
    special_teams_tds: float = setting(6.0, "Return / special teams TD", 0, 12, 0.5, "Other")


@dataclass
class LeagueConfig:
    league: League = field(default_factory=League)
    roster: Roster = field(default_factory=Roster)
    scoring: Scoring = field(default_factory=Scoring)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LeagueConfig":
        """Missing values fall back to defaults; unknown keys raise so typos aren't silently ignored."""
        parts = {}
        for name, part_cls in (("league", League), ("roster", Roster), ("scoring", Scoring)):
            values = data.get(name, {})
            known = {f.name for f in fields(part_cls)}
            unknown = set(values) - known
            if unknown:
                raise ValueError(f"unknown {name} setting(s): {', '.join(sorted(unknown))}")
            parts[name] = part_cls(**values)
        return cls(**parts)

    def validate(self) -> list[str]:
        errors = []
        name = self.league.name.strip()
        if not name:
            errors.append("League name is required.")
        elif len(name) > 60:
            errors.append("League name must be 60 characters or fewer.")

        if self.league.type in PLANNED_TYPES:
            errors.append(f"{PLANNED_TYPES[self.league.type]} leagues are planned but not supported yet.")
        elif self.league.type not in SUPPORTED_TYPES:
            errors.append(f"Unknown league type '{self.league.type}'.")

        for part in (self.league, self.roster, self.scoring):
            for f in fields(part):
                if "min" not in f.metadata:
                    continue
                value = getattr(part, f.name)
                label = f.metadata["label"]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    errors.append(f"{label} must be a number.")
                elif isinstance(part, (League, Roster)) and value != int(value):
                    errors.append(f"{label} must be a whole number.")
                elif not f.metadata["min"] <= value <= f.metadata["max"]:
                    if part is self.roster and f.name == "IDP":
                        errors.append("IDP slots are planned for a later version.")
                    else:
                        errors.append(f"{label} must be between {f.metadata['min']} and {f.metadata['max']}.")

        if self.roster.starters == 0:
            errors.append("The roster needs at least one starting slot.")
        return errors

    def summary(self) -> str:
        """e.g. '12-team · Half-PPR · Superflex · Redraft'"""
        rec = self.scoring.receptions
        ppr = {1.0: "PPR", 0.5: "Half-PPR", 0.0: "Standard"}.get(rec, f"{rec:g} PPR")
        parts = [f"{self.league.teams}-team", ppr]
        if self.roster.SUPERFLEX:
            parts.append("Superflex")
        elif self.roster.QB >= 2:
            parts.append("2QB")
        if self.scoring.passing_tds == 6:
            parts.append("6-pt pass TD")
        if self.scoring.te_reception_bonus:
            parts.append("TE premium")
        parts.append(SUPPORTED_TYPES.get(self.league.type, self.league.type.title()))
        return " · ".join(parts)


def _preset(name, league_type="redraft", roster=None, scoring=None, teams=12) -> LeagueConfig:
    return LeagueConfig(
        league=League(name=name, type=league_type, teams=teams),
        roster=Roster(**(roster or {})),
        scoring=Scoring(**(scoring or {})),
    )


PRESETS: dict[str, LeagueConfig] = {
    "standard-ppr": _preset("Standard PPR"),
    "half-ppr": _preset("Half-PPR", scoring={"receptions": 0.5}),
    "standard": _preset("Standard (no PPR)", scoring={"receptions": 0.0}),
    "superflex-half-ppr": _preset(
        "Superflex Half-PPR", roster={"SUPERFLEX": 1, "BENCH": 7}, scoring={"receptions": 0.5},
    ),
    "bestball-half-ppr": _preset(
        "Best Ball Half-PPR", league_type="bestball",
        roster={"WR": 3, "K": 0, "DST": 0, "BENCH": 10}, scoring={"receptions": 0.5},
    ),
}


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "league"


def save_league(config: LeagueConfig, directory: Path = LEAGUES_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{slugify(config.league.name)}.json"
    path.write_text(json.dumps(config.to_dict(), indent=2))
    return path


def load_league(path: Path) -> LeagueConfig:
    return LeagueConfig.from_dict(json.loads(path.read_text()))


def list_leagues(directory: Path = LEAGUES_DIR) -> dict[str, LeagueConfig]:
    """Saved leagues keyed by slug. Files that fail to load are skipped."""
    leagues = {}
    for path in sorted(directory.glob("*.json")):
        try:
            leagues[path.stem] = load_league(path)
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    return leagues
