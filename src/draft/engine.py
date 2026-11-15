"""Draft state engine (SCRUM-58): who picks when, who's taken, every team's roster.

Snake order: round 1 goes 1 -> N, round 2 goes N -> 1, and so on. Optional third-round reversal
(3RR): round 3 repeats round 2's direction (N -> 1), then the snake continues from there, so the
team with the last pick of round 1 gets the 1st pick of round 4 instead of round 3.

Picks are entered by hand (v1). Undo removes the last pick. A draft saves to
data/drafts/<league-slug>/<draft-id>.json and loads back to exactly the same state.

Kickers and defenses (Lucas, 2026-11-14): we don't project them. They're recorded like any pick
(id "K-<name>" / "DST-<team>") so rosters stay accurate, and `must_fill_kdst` tells the
recommender when a team's remaining picks are only just enough to fill its K/DST slots.
That's a roster rule, not a strategy.

Keepers and auctions are out of scope for v1.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from src.scoring.config import PROJECT_ROOT, LeagueConfig

DRAFTS_DIR = PROJECT_ROOT / "data" / "drafts"
POSITIONS = {"QB", "RB", "WR", "TE", "K", "DST"}
KDST = ("K", "DST")


@dataclass
class Pick:
    overall: int          # 1-based pick number in the whole draft
    round: int            # 1-based
    team: int             # 0-based team index (draft slot - 1)
    player_id: str
    name: str
    position: str


@dataclass
class Draft:
    league_slug: str
    teams: int
    rounds: int
    user_slot: int                      # 1-based draft slot of the user's team
    third_round_reversal: bool = False
    team_names: list[str] = field(default_factory=list)
    picks: list[Pick] = field(default_factory=list)
    draft_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S"))

    def __post_init__(self):
        if not 1 <= self.user_slot <= self.teams:
            raise ValueError(f"user slot must be 1-{self.teams}")
        if not self.team_names:
            self.team_names = [("You" if i == self.user_slot - 1 else f"Team {i + 1}") for i in range(self.teams)]

    @classmethod
    def for_league(cls, slug: str, config: LeagueConfig, user_slot: int, third_round_reversal: bool = False) -> "Draft":
        return cls(league_slug=slug, teams=config.league.teams, rounds=config.roster.size,
                   user_slot=user_slot, third_round_reversal=third_round_reversal)

    # ---------- order ----------
    @property
    def total_picks(self) -> int:
        return self.teams * self.rounds

    @property
    def is_complete(self) -> bool:
        return len(self.picks) >= self.total_picks

    @property
    def next_overall(self) -> int:
        return len(self.picks) + 1

    def round_of(self, overall: int) -> int:
        return (overall - 1) // self.teams + 1

    def team_for(self, overall: int) -> int:
        """0-based team index on the clock at this overall pick."""
        if not 1 <= overall <= self.total_picks:
            raise ValueError(f"pick {overall} is outside the draft (1-{self.total_picks})")
        rnd = self.round_of(overall)
        i = (overall - 1) % self.teams
        reverse = rnd % 2 == 0
        if self.third_round_reversal and rnd >= 3:
            reverse = not reverse   # from round 3 on, every direction flips
        return self.teams - 1 - i if reverse else i

    @property
    def on_clock(self) -> int | None:
        return None if self.is_complete else self.team_for(self.next_overall)

    def team_pick_numbers(self, team: int) -> list[int]:
        return [o for o in range(1, self.total_picks + 1) if self.team_for(o) == team]

    def next_pick_for(self, team: int, after: int | None = None) -> int | None:
        """The team's next overall pick after `after` (default: after the current pick)."""
        after = self.next_overall if after is None else after
        return next((o for o in self.team_pick_numbers(team) if o > after), None)

    # ---------- picks ----------
    def taken(self) -> set[str]:
        return {p.player_id for p in self.picks}

    def make_pick(self, player_id: str, name: str, position: str) -> Pick:
        if self.is_complete:
            raise ValueError("the draft is complete")
        if position not in POSITIONS:
            raise ValueError(f"unknown position {position!r}")
        if player_id in self.taken():
            raise ValueError(f"{name} is already drafted")
        o = self.next_overall
        pick = Pick(overall=o, round=self.round_of(o), team=self.team_for(o), player_id=player_id, name=name,
                    position=position)
        self.picks.append(pick)
        return pick

    def undo(self) -> Pick | None:
        return self.picks.pop() if self.picks else None

    def roster(self, team: int) -> list[Pick]:
        return [p for p in self.picks if p.team == team]

    def picks_left(self, team: int) -> int:
        return self.rounds - len(self.roster(team))

    def must_fill_kdst(self, team: int, config: LeagueConfig) -> list[str]:
        """K/DST slots still empty when the team has no spare picks left for anything else."""
        have = [p.position for p in self.roster(team)]
        open_slots = [pos for pos in KDST for _ in range(max(0, getattr(config.roster, pos) - have.count(pos)))]
        return open_slots if self.picks_left(team) <= len(open_slots) else []

    # ---------- save / load ----------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Draft":
        picks = [Pick(**p) for p in data.get("picks", [])]
        return cls(**{**{k: v for k, v in data.items() if k != "picks"}, "picks": picks})

    def path(self, directory: Path = DRAFTS_DIR) -> Path:
        return directory / self.league_slug / f"{self.draft_id}.json"

    def save(self, directory: Path = DRAFTS_DIR) -> Path:
        p = self.path(directory)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p


def load_draft(path: Path) -> Draft:
    return Draft.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def list_drafts(league_slug: str, directory: Path = DRAFTS_DIR) -> list[Path]:
    return sorted((directory / league_slug).glob("*.json"))
