"""Mock-draft test (SCRUM-59): does the recommender build better teams? Rules set 2026-11-14 before results.

    seasons    2022-2024 (2025 is the final exam)
    league     standard PPR, 12 teams, our standard roster
    our team   three drafters on the SAME drafts (same opponent randomness, paired by seed):
                 sim  the Monte Carlo recommender (SCRUM-60), trimmed to 6 candidates x 50 simulations
                 vor  plain best-VOR
                 adp  best available by ADP with the same noise opponents get ("an average human")
    opponents  11 teams draft from the experts' preseason ranking (ECR) + noise, NOT ADP: the
               recommender assumes ADP opponents internally, so testing against ADP opponents would
               let it grade its own homework. Same eligibility rules (position caps, K/DST at the end).
    scoring    each team's ACTUAL season: best legal lineup every week from real weekly PPR points,
               summed over the regular season. Hindsight lineups apply to all 12 teams equally.
    headline   our points minus the league's average team, and how often we finish 1st in points;
               paired differences between drafters with 95% bootstrap CIs over drafts
    success    sim beats vor (CI on the paired edge excludes 0); otherwise the site recommends by VOR

Usage:
    python -m src.backtest.mock_draft [--reps 10]
"""

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from src.backtest.harness import bootstrap
from src.draft.engine import KDST, Draft
from src.draft.recommender import ADP_NOISE, MISSING_ADP, Pool, _eligible, best_vor, position_caps, recommend
from src.draft.vor import vor_board
from src.ingest.consensus import load_adp, load_ecr
from src.models import calibration
from src.models.season import OUT_DIR, PLAYER_WEEKS_PATH
from src.scoring.config import PRESETS, LeagueConfig
from src.scoring.lineup import Player, lineup_points

SEASONS = [2022, 2023, 2024]
DRAFTERS = ["sim", "vor", "adp"]
SIM_CANDIDATES, SIM_SIMS = 6, 50
REPORT_DIR = Path("docs/reports")


def weekly_points(season: int) -> dict[str, np.ndarray]:
    pw = pl.read_parquet(PLAYER_WEEKS_PATH, columns=["player_id", "season", "week", "fantasy_points_ppr"])
    pw = pw.filter(pl.col("season") == season, pl.col("week") <= 18)
    out: dict[str, np.ndarray] = {}
    for pid, week, pts in pw.select("player_id", "week", "fantasy_points_ppr").rows():
        out.setdefault(pid, np.zeros(18))[week - 1] += pts or 0.0
    return out


def season_score(roster: list[tuple[str, str]], weeks: dict[str, np.ndarray], config: LeagueConfig) -> float:
    total = 0.0
    for w in range(18):
        players = [Player(pid, pos, float(weeks[pid][w])) for pid, pos in roster if pid in weeks]
        total += lineup_points(players, config.roster)
    return total


def kdst_needed(config: LeagueConfig, positions: list[str], picks_left: int) -> list[str]:
    open_slots = [k for k in KDST for _ in range(max(0, getattr(config.roster, k) - positions.count(k)))]
    return open_slots if picks_left <= len(open_slots) else []


def first_eligible(order, pool: Pool, taken: set[str], positions: list[str], caps: dict[str, int]) -> int | None:
    counts = {p: positions.count(p) for p in set(positions)}
    for i in order:
        if pool.ids[i] not in taken and _eligible(pool.pos[i], counts, caps, []):
            return int(i)
    return None


def run_draft(pool: Pool, ecr_rank: np.ndarray, config: LeagueConfig, slot: int, drafter: str, seed: int) -> list[list[tuple[str, str]]]:
    """One full draft. Returns every team's roster as (player_id, position)."""
    rng = np.random.default_rng(seed)
    opp_order = np.argsort(ecr_rank + rng.normal(0, 1, len(ecr_rank)) * ADP_NOISE(ecr_rank))
    adp_order = np.argsort(pool.adp + rng.normal(0, 1, len(pool.adp)) * ADP_NOISE(pool.adp))
    d = Draft.for_league("mock", config, user_slot=slot)
    me, caps = slot - 1, position_caps(config)
    rosters: list[list[tuple[str, str]]] = [[] for _ in range(d.teams)]
    while not d.is_complete:
        team = d.on_clock
        positions = [p for _, p in rosters[team]]
        need = kdst_needed(config, positions, d.picks_left(team))
        if need:
            pid, pos = f"{need[0]}-{team}-{d.next_overall}", need[0]
        else:
            if team != me:
                i = first_eligible(opp_order, pool, d.taken(), positions, caps)
            elif drafter == "adp":
                i = first_eligible(adp_order, pool, d.taken(), positions, caps)
            elif drafter == "vor":
                pid = best_vor(d, pool, config)
                i = pool.ids.index(pid) if pid else None
            else:
                recs = recommend(d, pool, config, n_candidates=SIM_CANDIDATES, n_sims=SIM_SIMS, seed=seed + d.next_overall)
                i = pool.ids.index(recs[0]["player_id"]) if recs and recs[0]["player_id"] else None
            pid, pos = (pool.ids[i], pool.pos[i]) if i is not None else (f"empty-{team}-{d.next_overall}", "WR")
        d.make_pick(pid, pid, pos if pos in ("QB", "RB", "WR", "TE", "K", "DST") else "WR")
        rosters[team].append((pid, pos))
    return rosters


def run(reps: int, config: LeagueConfig, calibrated: bool = False) -> pl.DataFrame:
    rows = []
    for season in SEASONS:
        board = pl.read_parquet(OUT_DIR / f"board_{season}.parquet")
        if calibrated:  # fit on earlier dev seasons only (2022 stays uncalibrated)
            board = calibration.apply(board, calibration.fit([s for s in SEASONS if s < season]))
        vb, _ = vor_board(board, config)
        pool = Pool.build(vb, load_adp(season))
        ecr = load_ecr(season).drop_nulls("player_id").unique("player_id").select("player_id", "ecr")
        ecr_map = dict(ecr.rows())
        ecr_rank = np.array([ecr_map.get(pid, MISSING_ADP + i) for i, pid in enumerate(pool.ids)], dtype=float)
        weeks = weekly_points(season)
        for slot in range(1, 13):
            for rep in range(reps):
                seed = season * 1000 + slot * 37 + rep
                for drafter in DRAFTERS:
                    rosters = run_draft(pool, ecr_rank, config, slot, drafter, seed)
                    scores = np.array([season_score(r, weeks, config) for r in rosters])
                    me = slot - 1
                    others = np.delete(scores, me)
                    rows.append({"season": season, "slot": slot, "rep": rep, "drafter": drafter,
                                 "points": scores[me], "edge": scores[me] - others.mean(),
                                 "rank": int((scores > scores[me]).sum() + 1)})
            print(f"  {season} slot {slot} done", flush=True)
    return pl.DataFrame(rows)


def report(res: pl.DataFrame) -> str:
    lines = [f"# Mock-draft test ({date.today()})", "",
             "Standard PPR, 12 teams, 2022-24 (2025 untouched). Opponents draft from preseason ECR + noise. "
             "Scored on each team's actual season (best lineup every week). Rules in src/backtest/mock_draft.py.", "",
             "| drafter | drafts | avg points | edge vs league avg [95% CI] | finishes 1st | avg rank |",
             "| --- | --- | --- | --- | --- | --- |"]
    for dr in DRAFTERS:
        r = res.filter(pl.col("drafter") == dr)
        e = bootstrap(r["edge"].to_numpy())
        lines.append(f"| {dr} | {r.height} | {r['points'].mean():.0f} | {e} | {(r['rank'] == 1).mean():.1%} | {r['rank'].mean():.2f} |")
    wide = res.pivot(on="drafter", index=["season", "slot", "rep"], values="edge")
    lines += ["", "Paired differences (same drafts):", "", "| comparison | edge difference [95% CI] |", "| --- | --- |"]
    for a, b in [("sim", "vor"), ("sim", "adp"), ("vor", "adp")]:
        lines.append(f"| {a} - {b} | {bootstrap((wide[a] - wide[b]).to_numpy())} |")
    lines += ["", "By season (edge vs league average):", "", "| season | sim | vor | adp |", "| --- | --- | --- | --- |"]
    for s in SEASONS:
        w = res.filter(pl.col("season") == s).group_by("drafter").agg(pl.col("edge").mean())
        m = dict(w.rows())
        lines.append(f"| {s} | {m['sim']:.0f} | {m['vor']:.0f} | {m['adp']:.0f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--calibrated", action="store_true", help="per-position calibration (src/models/calibration.py)")
    args = ap.parse_args()
    res = run(args.reps, PRESETS["standard-ppr"], args.calibrated)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"mock_draft_{date.today()}" + ("_calibrated" if args.calibrated else "")
    res.write_parquet(REPORT_DIR / f"{tag}.parquet")
    text = report(res)
    (REPORT_DIR / f"{tag}.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
