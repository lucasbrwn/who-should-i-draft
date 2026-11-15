"""Monte Carlo pick recommender (SCRUM-60). Decisions: Lucas, 2026-11-14.

Every player is valued by what he adds to THIS roster ("marginal value"): the change in roster score
(best legal starting lineup + BENCH_WEIGHT x bench) if he's added. A 2nd QB in a 1-QB league only
adds backup value; in superflex he adds a starter's. No position rules: it falls out of the league's
lineup slots. (Plain VOR ignored this and took two TEs, or two QBs, in the first five rounds:
mock drafts 2026-11-14.)

At the user's pick, for the top candidates by marginal value (filtered by the strategy preset):
    1. draft the candidate,
    2. simulate the rest of the draft many times:
         opponents  pick the first ELIGIBLE player in a noisy ADP order (ADP + noise, one fresh draw
                    per simulated draft); eligible = under the position cap, and K/DST only when the
                    team must fill those slots.
         our team   picks the best marginal value each time (the same rollout for every candidate).
    3. score our final roster.
Recommend the candidate with the highest average final score.

Why ADP for opponents: ADP is where real drafters took players. Why the bench credit: backups play
when starters miss games, about 2.3 of 17 (SCRUM-51), so a bench player is worth ~15% of what he
beats a waiver pickup by.
K and DST aren't projected: 0-point placeholders, picked only when must_fill_kdst says so.
Search budget (Lucas, 2026-11-14: accuracy over speed; real draft clocks give 60-90 s):
    stage 1  N_CANDIDATES candidates x N_SIMS simulations
    stage 2  STAGE2_SIMS more simulations for the top STAGE2_TOP, where the close calls are
The mock-draft test uses a trimmed budget (it tests the method; more simulations only cut noise).
"""

import bisect
from dataclasses import dataclass
from typing import Callable

import numpy as np
import polars as pl

from src.draft.engine import KDST, Draft
from src.scoring.config import LeagueConfig
from src.scoring.lineup import Player

BENCH_WEIGHT = 0.15
# Upside-valued bench (SCRUM-94 #3, Lucas): a bench player also earns
#   UPSIDE_WEIGHT x breakout chance x (ceiling - waiver level),
# the value of him becoming a starter discounted by how likely that is. Off (0) until the
# mock-draft test says it helps; the pre-registered test value is 0.5.
UPSIDE_WEIGHT = 0.0
N_CANDIDATES = 10
N_SIMS = 150
STAGE2_TOP, STAGE2_SIMS = 3, 600           # live budget: ~8-10 s per pick
PER_POSITION = 3                          # candidates considered per position before ranking by marginal value
TIE_POINTS = 1.0                          # marginal values this close count as a tie, broken by ADP (crowd's upside view)
ADP_NOISE = lambda adp: 2.0 + 0.10 * adp  # noqa: E731  (sd in picks: early picks are tighter)
MISSING_ADP = 300.0                       # not in ADP: late, ordered by our VOR
SKILL = ("QB", "RB", "WR", "TE")

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
    by_pos: dict             # position -> indices sorted by points, best first
    replacement: dict        # position -> replacement-level season points (from the VOR board)
    upside: np.ndarray       # breakout chance x (ceiling - replacement); 0 without an upside board

    @classmethod
    def build(cls, vor_board: pl.DataFrame, adp: pl.DataFrame | None, upside: pl.DataFrame | None = None) -> "Pool":
        b = vor_board.sort("vor", descending=True)
        if adp is not None:
            b = b.join(adp.drop_nulls("player_id").unique("player_id").select("player_id", "adp"),
                       on="player_id", how="left")
        else:
            b = b.with_columns(adp=pl.lit(None, dtype=pl.Float64))
        b = b.with_columns(adp=pl.col("adp").fill_null(MISSING_ADP + pl.int_range(pl.len()).cast(pl.Float64)))
        pos, pts = b["position"].to_list(), b["points"].to_numpy()
        by_pos = {p: sorted((i for i in range(b.height) if pos[i] == p), key=lambda i: -pts[i]) for p in SKILL}
        repl = dict(b.group_by("position").agg(pl.col("replacement").first()).rows()) if "replacement" in b.columns else {}
        if upside is not None:
            b = b.join(upside.select("player_id", "ceiling", "breakout_p"), on="player_id", how="left")
            gain = (pl.col("ceiling").fill_null(0.0) - pl.col("position").replace_strict(repl, default=0.0)).clip(lower_bound=0.0)
            up_values = b.select(pl.col("breakout_p").fill_null(0.0) * gain).to_series().to_numpy()
        else:
            up_values = np.zeros(b.height)
        return cls(ids=b["player_id"].to_list(), names=b["player_display_name"].to_list(), pos=pos,
                   points=pts, vor=b["vor"].to_numpy(), adp=b["adp"].to_numpy(),
                   vor_order=np.arange(b.height), by_pos=by_pos, replacement={p: float(repl.get(p, 0.0)) for p in SKILL},
                   upside=up_values)


def position_caps(config: LeagueConfig) -> dict[str, int]:
    """Most of each position a sensible drafter rosters (used for OPPONENTS): starters + flex share + a backup for QB/TE."""
    r = config.roster
    return {"QB": r.QB + r.SUPERFLEX + 1, "TE": r.TE + 1, "RB": r.size, "WR": r.size, "K": r.K, "DST": r.DST}


class Roster:
    """Per-position point lists (best first) with a fast score: same greedy fill as src/scoring/lineup.py.

    Waiver-level stand-ins: every starting slot can be filled by a stand-in worth that position's
    replacement level (you can always find one on waivers or late). Stand-ins can start but never
    count as bench, so a real player only adds value when he beats one. That stops "QB in round 1
    of a 1-QB league": an empty QB slot isn't worth 0, it's worth a waiver QB.
    """

    def __init__(self, config: LeagueConfig, points_by_pos: dict[str, list[float]] | None = None,
                 replacement: dict[str, float] | None = None):
        self.r = config.roster
        # (points, upside) per player, best points first
        self.lists = {p: sorted(((x, 0.0) if not isinstance(x, tuple) else x for x in (points_by_pos or {}).get(p, [])),
                                reverse=True) for p in SKILL}
        repl = replacement or {}
        self.repl = repl
        r = self.r
        n_ghosts = {"QB": r.QB + r.SUPERFLEX, "RB": r.RB + r.FLEX + r.SUPERFLEX, "WR": r.WR + r.FLEX + r.SUPERFLEX,
                    "TE": r.TE + r.FLEX + r.SUPERFLEX}
        self.ghosts = {p: [repl[p]] * n_ghosts[p] for p in SKILL if repl.get(p, 0) > 0}

    def copy(self) -> "Roster":
        new = Roster.__new__(Roster)
        new.r, new.lists, new.ghosts, new.repl = self.r, {p: list(v) for p, v in self.lists.items()}, self.ghosts, self.repl
        return new

    def add(self, pos: str, pts: float, up: float = 0.0) -> None:
        if pos in self.lists:
            bisect.insort(self.lists[pos], (pts, up), key=lambda t: -t[0])

    def score(self, extra: tuple | None = None) -> float:
        """extra: (position, points[, upside]) scored as if added to the roster."""
        lists = self.lists
        if extra is not None and extra[0] in lists:
            item = (extra[1], extra[2] if len(extra) > 2 else 0.0)
            lists = {**lists, extra[0]: sorted(lists[extra[0]] + [item], reverse=True)}
        # (points, is_real, position, upside): stand-ins are (repl, False, pos, 0); best first, real wins ties
        full = {p: sorted([(x, True, p, u) for x, u in lists[p]] + [(g, False, p, 0.0) for g in self.ghosts.get(p, [])],
                          reverse=True) for p in SKILL}
        starters, left = 0.0, {}
        for p in SKILL:
            n = getattr(self.r, p)
            starters += sum(e[0] for e in full[p][:n])
            left[p] = full[p][n:]
        flex_pool = sorted(left["RB"] + left["WR"] + left["TE"], reverse=True)
        starters += sum(e[0] for e in flex_pool[: self.r.FLEX])
        sf_pool = sorted(flex_pool[self.r.FLEX:] + left["QB"], reverse=True)
        starters += sum(e[0] for e in sf_pool[: self.r.SUPERFLEX])
        # A backup is worth BENCH_WEIGHT x what he beats a waiver pickup by (not his raw points:
        # raw points made a 3rd QB look like the best bench add).
        bench_rows = [e for e in sf_pool[self.r.SUPERFLEX:] if e[1]]
        bench = sum(max(0.0, e[0] - self.repl.get(e[2], 0.0)) for e in bench_rows)
        upside = sum(e[3] for e in bench_rows)
        return starters + BENCH_WEIGHT * bench + UPSIDE_WEIGHT * upside


def score_roster(players: list[Player], config: LeagueConfig, replacement: dict[str, float] | None = None) -> float:
    by_pos: dict[str, list[float]] = {}
    for p in players:
        by_pos.setdefault(p.position, []).append(p.points)
    return Roster(config, by_pos, replacement).score()


def _eligible(position: str, counts: dict[str, int], caps: dict[str, int], kdst_needed: list[str]) -> bool:
    if kdst_needed:
        return position in kdst_needed
    if position in KDST:
        return False
    return counts.get(position, 0) < caps[position]


def best_by_marginal(pool: Pool, roster: Roster, taken: set[str], k: int, rnd: int,
                     roster_positions: list[str], allowed: Allowed | None, per_position: int = PER_POSITION,
                     caps: dict[str, int] | None = None) -> list[tuple[float, int]]:
    """Top-k (marginal value, index) among the best `per_position` available players at each position.

    Our team follows the same roster caps as simulated opponents (no 3rd QB in a 1-QB league).
    Near-ties (within TIE_POINTS) go to the earlier ADP: late in drafts every option adds ~0
    projected points, and where the crowd drafts a player reflects upside a mean projection misses.
    """
    base = roster.score()
    scored = []
    counts = {p: roster_positions.count(p) for p in SKILL}
    for p in SKILL:
        if allowed and not allowed(p, rnd, roster_positions):
            continue
        if caps and counts[p] >= caps[p]:
            continue
        found = 0
        for i in pool.by_pos[p]:
            if pool.ids[i] in taken:
                continue
            scored.append((roster.score((p, float(pool.points[i]), float(pool.upside[i]))) - base, i))
            found += 1
            if found == per_position:
                break
    scored.sort(key=lambda t: (round(t[0] / TIE_POINTS), -pool.adp[t[1]]), reverse=True)
    return scored[:k]


class Simulator:
    def __init__(self, draft: Draft, pool: Pool, config: LeagueConfig, allowed: Allowed | None = None, seed: int = 0):
        self.draft, self.pool, self.config, self.allowed = draft, pool, config, allowed
        self.caps = position_caps(config)
        self.rng = np.random.default_rng(seed)
        self.index = {pid: i for i, pid in enumerate(pool.ids)}

    def team_state(self, team: int) -> tuple[list[str], list[str]]:
        picks = [p for p in self.draft.picks if p.team == team]
        return [p.player_id for p in picks], [p.position for p in picks]

    def roster_for(self, pids: list[str]) -> Roster:
        r = Roster(self.config, replacement=self.pool.replacement)
        for pid in pids:
            i = self.index.get(pid)
            if i is not None:
                r.add(self.pool.pos[i], float(self.pool.points[i]), float(self.pool.upside[i]))
        return r

    def run_once(self, first_pick: int, me: int) -> float:
        """Draft pool index `first_pick` for team `me` now, simulate to the end, return our roster's score."""
        d, pool = self.draft, self.pool
        taken = d.taken() | {pool.ids[first_pick]}
        positions = [[] for _ in range(d.teams)]
        for p in d.picks:
            positions[p.team].append(p.position)
        positions[me].append(pool.pos[first_pick])
        mine = self.roster_for([pid for pid, t in ((p.player_id, p.team) for p in d.picks) if t == me])
        mine.add(pool.pos[first_pick], float(pool.points[first_pick]), float(pool.upside[first_pick]))
        noisy = pool.adp + self.rng.normal(0, 1, len(pool.adp)) * ADP_NOISE(pool.adp)
        adp_order = np.argsort(noisy)
        ptr = 0
        for overall in range(d.next_overall + 1, d.total_picks + 1):
            team = d.team_for(overall)
            left = d.rounds - len(positions[team])
            need = _kdst_needed(self.config, positions[team], left)
            if need:
                taken.add(f"{need[0]}-sim-{team}-{overall}")
                positions[team].append(need[0])
                continue
            if team == me:
                best = best_by_marginal(pool, mine, taken, 1, d.round_of(overall), positions[me], self.allowed,
                                        caps=self.caps)
                if not best:
                    positions[me].append("WR")
                    continue
                i = best[0][1]
                mine.add(pool.pos[i], float(pool.points[i]), float(pool.upside[i]))
            else:
                counts = {p: positions[team].count(p) for p in set(positions[team])}
                i = None
                for j in adp_order:
                    if pool.ids[j] not in taken and _eligible(pool.pos[j], counts, self.caps, []):
                        i = int(j)
                        break
                if i is None:
                    positions[team].append("WR")
                    continue
            taken.add(pool.ids[i])
            positions[team].append(pool.pos[i])
        return mine.score()


def _kdst_needed(config: LeagueConfig, positions: list[str], picks_left: int) -> list[str]:
    open_slots = [k for k in KDST for _ in range(max(0, getattr(config.roster, k) - positions.count(k)))]
    return open_slots if picks_left <= len(open_slots) else []


def candidates(draft: Draft, pool: Pool, config: LeagueConfig, team: int, allowed: Allowed | None,
               n: int = N_CANDIDATES) -> list[int]:
    """Top-n available players by marginal value to this team's roster."""
    sim = Simulator(draft, pool, config, allowed)
    pids, positions = sim.team_state(team)
    rnd = draft.round_of(draft.next_overall)
    return [i for _, i in best_by_marginal(pool, sim.roster_for(pids), draft.taken(), n, rnd, positions, allowed,
                                           caps=sim.caps)]


def recommend(draft: Draft, pool: Pool, config: LeagueConfig, allowed: Allowed | None = None,
              n_candidates: int = N_CANDIDATES, n_sims: int = N_SIMS, seed: int = 0,
              stage2_top: int = STAGE2_TOP, stage2_sims: int = STAGE2_SIMS) -> list[dict]:
    """Ranked recommendations for the team on the clock (usually the user)."""
    team = draft.on_clock
    if team is None:
        return []
    need = draft.must_fill_kdst(team, config)
    if need:
        return [{"player_id": None, "name": f"Your {need[0]}", "position": need[0], "expected": None,
                 "reason": f"Only {draft.picks_left(team)} pick(s) left: fill your {' and '.join(need)} now."}]
    sim = Simulator(draft, pool, config, allowed, seed)
    pids, positions = sim.team_state(team)
    base = sim.roster_for(pids)
    runs = {i: [sim.run_once(i, team) for _ in range(n_sims)]
            for i in candidates(draft, pool, config, team, allowed, n_candidates)}
    if stage2_sims and stage2_top:  # spend the extra simulations on the close calls at the top
        top = sorted(runs, key=lambda i: np.mean(runs[i]), reverse=True)[:stage2_top]
        for i in top:
            runs[i] += [sim.run_once(i, team) for _ in range(stage2_sims)]
    out = []
    for i, sc in runs.items():
        scores = np.array(sc)
        out.append({"player_id": pool.ids[i], "name": pool.names[i], "position": pool.pos[i],
                    "points": float(pool.points[i]), "vor": float(pool.vor[i]), "adp": float(pool.adp[i]),
                    "marginal": base.score((pool.pos[i], float(pool.points[i]), float(pool.upside[i]))) - base.score(),
                    "upside": float(pool.upside[i]),
                    "expected": float(scores.mean()), "se": float(scores.std(ddof=1) / np.sqrt(len(scores))),
                    "sims": len(scores)})
    return sorted(out, key=lambda r: r["expected"], reverse=True)


def best_marginal(draft: Draft, pool: Pool, config: LeagueConfig, allowed: Allowed | None = None) -> str | None:
    """Roster-aware VOR pick: the most marginal value right now, no simulation (the benchmark, SCRUM-59)."""
    team = draft.on_clock
    c = candidates(draft, pool, config, team, allowed, 1) if team is not None else []
    return pool.ids[c[0]] if c else None


best_vor = best_marginal  # kept for callers that predate the roster-aware fix
