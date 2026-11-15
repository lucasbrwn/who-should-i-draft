"""Monte Carlo pick recommender (SCRUM-60). Decisions: Lucas, 2026-11-14.

At the user's pick, for each of the top candidates (by VOR, filtered by the user's strategy preset):
    1. draft the candidate,
    2. simulate the rest of the draft many times:
         opponents  pick the first ELIGIBLE player in a noisy ADP order (ADP + noise, one fresh noise
                    draw per simulated draft). Eligible = under the position cap (no 3rd QB in a
                    1-QB league) and not a K/DST until the team has to fill those slots.
         our team   keeps picking best VOR (the same rollout every candidate gets, so the comparison is fair).
    3. score our final roster: best legal starting lineup (league slots) + BENCH_WEIGHT x bench points.
Recommend the candidate with the highest average final score.

Why ADP for opponents: ADP is where real drafters actually took players (Lucas's FantasyPros files);
experts' rankings describe opinion, not drafting.
Why the bench credit: backups play when starters miss games. Fantasy-relevant players miss about
2.3 of 17 games to injury (SCRUM-51), so a bench player is worth ~15% of his projection.

K and DST aren't projected; they're placeholders worth 0 points, picked only when must_fill_kdst says so.
Speed target: < 3 seconds per pick (8 candidates x 200 simulations by default).
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import polars as pl

from src.draft.engine import KDST, Draft
from src.scoring.config import LeagueConfig
from src.scoring.lineup import Player, lineup_points, optimal_lineup

BENCH_WEIGHT = 0.15
N_CANDIDATES = 8
N_SIMS = 200
ADP_NOISE = lambda adp: 2.0 + 0.10 * adp  # noqa: E731  (sd in picks: early picks are tighter)
MISSING_ADP = 300.0                       # undrafted in ADP: late, ordered by our VOR

# allowed(position, round, roster_positions) -> bool. Strategy presets (SCRUM-92) plug in here.
Allowed = Callable[[str, int, list[str]], bool]


@dataclass
class Pool:
    ids: list[str]
    names: list[str]
    pos: list[str]
    points: np.ndarray
    vor: np.ndarray
    adp: np.ndarray
    vor_order: np.ndarray   # indices, best VOR first

    @classmethod
    def build(cls, vor_board: pl.DataFrame, adp: pl.DataFrame | None) -> "Pool":
        b = vor_board.sort("vor", descending=True)
        if adp is not None:
            b = b.join(adp.drop_nulls("player_id").unique("player_id").select("player_id", "adp"),
                       on="player_id", how="left")
        else:
            b = b.with_columns(adp=pl.lit(None, dtype=pl.Float64))
        b = b.with_columns(adp=pl.col("adp").fill_null(MISSING_ADP + pl.int_range(pl.len()).cast(pl.Float64)))
        return cls(ids=b["player_id"].to_list(), names=b["player_display_name"].to_list(), pos=b["position"].to_list(),
                   points=b["points"].to_numpy(), vor=b["vor"].to_numpy(), adp=b["adp"].to_numpy(),
                   vor_order=np.arange(b.height))


def position_caps(config: LeagueConfig) -> dict[str, int]:
    """Most of each position a sensible drafter rosters: starters + flex share + one backup for QB/TE."""
    r = config.roster
    return {"QB": r.QB + r.SUPERFLEX + 1, "TE": r.TE + 1,
            "RB": r.size, "WR": r.size, "K": r.K, "DST": r.DST}


def score_roster(players: list[Player], config: LeagueConfig) -> float:
    lineup = optimal_lineup(players, config.roster)
    starters = {p.id for slot in lineup.values() for p in slot}
    bench = sum(p.points for p in players if p.id not in starters)
    return lineup_points(players, config.roster) + BENCH_WEIGHT * bench


def _eligible(position: str, counts: dict[str, int], caps: dict[str, int], kdst_needed: list[str]) -> bool:
    if kdst_needed:
        return position in kdst_needed
    if position in KDST:
        return False
    return counts.get(position, 0) < caps[position]


class Simulator:
    def __init__(self, draft: Draft, pool: Pool, config: LeagueConfig, allowed: Allowed | None = None, seed: int = 0):
        self.draft, self.pool, self.config, self.allowed = draft, pool, config, allowed
        self.caps = position_caps(config)
        self.rng = np.random.default_rng(seed)
        self.index = {pid: i for i, pid in enumerate(pool.ids)}

    def _rosters(self) -> list[list[str]]:
        rosters = [[] for _ in range(self.draft.teams)]
        for p in self.draft.picks:
            rosters[p.team].append(p.player_id)
        return rosters

    def _kdst_needed(self, team: int, positions: list[str], picks_left: int) -> list[str]:
        open_slots = [k for k in KDST for _ in range(max(0, getattr(self.config.roster, k) - positions.count(k)))]
        return open_slots if picks_left <= len(open_slots) else []

    def run_once(self, first_pick: str, me: int) -> float:
        """Draft `first_pick` for team `me` now, simulate to the end, return our roster's score."""
        d, pool = self.draft, self.pool
        taken = d.taken() | {first_pick}
        rosters = self._rosters()
        rosters[me].append(first_pick)
        positions = [[self._pos(pid) for pid in r] for r in rosters]
        noisy = pool.adp + self.rng.normal(0, 1, len(pool.adp)) * ADP_NOISE(pool.adp)
        adp_order = np.argsort(noisy)
        for overall in range(d.next_overall + 1, d.total_picks + 1):
            team = d.team_for(overall)
            left = d.rounds - len(rosters[team])
            need = self._kdst_needed(team, positions[team], left)
            counts = {p: positions[team].count(p) for p in set(positions[team])}
            if need:
                pid, pos = f"{need[0]}-sim-{team}-{overall}", need[0]
            else:
                order = pool.vor_order if team == me else adp_order
                rnd = d.round_of(overall)
                pid = pos = None
                for i in order:
                    cand = pool.ids[i]
                    if cand in taken:
                        continue
                    p = pool.pos[i]
                    if not _eligible(p, counts, self.caps, []):
                        continue
                    if team == me and self.allowed and not self.allowed(p, rnd, positions[me]):
                        continue
                    pid, pos = cand, p
                    break
                if pid is None:
                    pid, pos = f"empty-{team}-{overall}", "WR"
            taken.add(pid)
            rosters[team].append(pid)
            positions[team].append(pos)
        return score_roster(self._players(rosters[me]), self.config)

    def _pos(self, pid: str) -> str:
        i = self.index.get(pid)
        if i is not None:
            return self.pool.pos[i]
        return next((p.position for p in self.draft.picks if p.player_id == pid), pid.split("-")[0])

    def _players(self, pids: list[str]) -> list[Player]:
        out = []
        for pid in pids:
            i = self.index.get(pid)
            if i is not None:
                out.append(Player(pid, self.pool.pos[i], float(self.pool.points[i])))
        return out


def candidates(draft: Draft, pool: Pool, config: LeagueConfig, team: int, allowed: Allowed | None,
               n: int = N_CANDIDATES) -> list[int]:
    """Top-n available players by VOR that the team may draft now."""
    taken = draft.taken()
    roster_pos = [p.position for p in draft.roster(team)]
    counts = {p: roster_pos.count(p) for p in set(roster_pos)}
    caps = position_caps(config)
    rnd = draft.round_of(draft.next_overall)
    out = []
    for i in pool.vor_order:
        if pool.ids[i] in taken or not _eligible(pool.pos[i], counts, caps, []):
            continue
        if allowed and not allowed(pool.pos[i], rnd, roster_pos):
            continue
        out.append(int(i))
        if len(out) == n:
            break
    return out


def recommend(draft: Draft, pool: Pool, config: LeagueConfig, allowed: Allowed | None = None,
              n_candidates: int = N_CANDIDATES, n_sims: int = N_SIMS, seed: int = 0) -> list[dict]:
    """Ranked recommendations for the team on the clock (usually the user)."""
    team = draft.on_clock
    if team is None:
        return []
    need = draft.must_fill_kdst(team, config)
    if need:
        return [{"player_id": None, "name": f"Your {need[0]}", "position": need[0], "expected": None,
                 "reason": f"Only {draft.picks_left(team)} pick(s) left: fill your {' and '.join(need)} now."}]
    sim = Simulator(draft, pool, config, allowed, seed)
    out = []
    for i in candidates(draft, pool, config, team, allowed, n_candidates):
        scores = np.array([sim.run_once(pool.ids[i], team) for _ in range(n_sims)])
        out.append({"player_id": pool.ids[i], "name": pool.names[i], "position": pool.pos[i],
                    "points": float(pool.points[i]), "vor": float(pool.vor[i]), "adp": float(pool.adp[i]),
                    "expected": float(scores.mean()), "se": float(scores.std(ddof=1) / np.sqrt(n_sims))})
    return sorted(out, key=lambda r: r["expected"], reverse=True)


def best_vor(draft: Draft, pool: Pool, config: LeagueConfig, allowed: Allowed | None = None) -> str | None:
    """The plain-VOR pick (the benchmark the simulator must beat, SCRUM-59)."""
    team = draft.on_clock
    c = candidates(draft, pool, config, team, allowed, 1) if team is not None else []
    return pool.ids[c[0]] if c else None
