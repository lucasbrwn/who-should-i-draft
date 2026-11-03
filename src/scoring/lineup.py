"""Pick the highest-scoring legal starting lineup from a roster.

Used by best ball scoring (the lineup is set after the games), replacement-level math, and draft
simulation in M6.

Why greedy is optimal here: slots are filled from strictest to loosest (QB, RB, WR, TE, K, DST,
then FLEX, then SUPERFLEX), always taking the best eligible player left. Every slot's eligible
positions either contain or don't overlap the earlier slots' positions (RB is inside FLEX, which
is inside SUPERFLEX). So a player used in a strict slot could never have done more good in a
looser one: anyone who fits the loose slot later is still available for it. The tests check
this against brute force.
"""

from dataclasses import dataclass, fields

from src.scoring.config import SLOT_ELIGIBILITY, Roster

# Strictest first. IDP and BENCH are not starting slots we fill (IDP isn't supported yet).
FILL_ORDER = ["QB", "RB", "WR", "TE", "K", "DST", "FLEX", "SUPERFLEX"]


@dataclass(frozen=True)
class Player:
    id: str
    position: str
    points: float


def optimal_lineup(players: list[Player], roster: Roster) -> dict[str, list[Player]]:
    """Slot name -> starters in that slot, best first. Slots may be short if the roster is thin."""
    if roster.IDP:
        raise ValueError("IDP slots are not supported yet")
    unknown = {f.name for f in fields(roster)} - set(FILL_ORDER) - {"IDP", "BENCH"}
    assert not unknown, f"FILL_ORDER is missing slots: {unknown}"

    remaining = sorted(players, key=lambda p: p.points, reverse=True)
    lineup = {}
    for slot in FILL_ORDER:
        eligible = SLOT_ELIGIBILITY[slot]
        # positions in `remaining` of the best eligible players for this slot
        picks = [i for i, p in enumerate(remaining) if p.position in eligible][:getattr(roster, slot)]
        lineup[slot] = [remaining[i] for i in picks]
        remaining = [p for i, p in enumerate(remaining) if i not in picks]
    return lineup


def lineup_points(players: list[Player], roster: Roster) -> float:
    return sum(p.points for slot in optimal_lineup(players, roster).values() for p in slot)
