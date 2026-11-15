"""Value over replacement (VOR) and tiers for a league (SCRUM-57). Decisions: Lucas, 2026-11-14.

1. League points. The season model (B, src/models/season.py) predicts standard-PPR season points.
   Other scoring (half PPR, TE premium, 6-pt pass TDs) is applied with a per-player factor from the
   weekly model's per-game stat projections (pg_*):
        league points = B's PPR points x (per-game points under the league's scoring / per-game PPR points)
   B keeps its accuracy; the stat mix decides how the league's rules move each player.
   Limitation: yardage bonuses (+3 at 100 yds) on an AVERAGE stat line are only approximate.
2. Replacement level = starter cutoff. Fill every team's starting lineup league-wide (teams x each
   slot, FLEX and SUPERFLEX included) with the optimal-lineup code. At each position the 3 best
   players who DON'T start, averaged, are replacement level. Superflex therefore pushes QBs up.
3. VOR = league points - replacement level at his position. The board ranks by VOR.
4. Tiers by gaps: within a position, a new tier starts where the drop to the next player is at
   least TIER_GAP x the typical gap, the MEAN drop among that position's draftable players. (The
   median made every top-5 WR his own tier: near the top every gap is big, so the mean, which the
   big gaps pull up, is the fairer yardstick.)
5. Best ball uses the mean for now; weighting the ceiling waits for floor/ceiling (SCRUM-52).

Usage:
    python -m src.draft.vor --league standard-ppr --season 2026
    python -m src.draft.vor --league superflex-half-ppr
"""

import argparse
from dataclasses import replace
from pathlib import Path

import polars as pl

from src.backtest.harness import SCORING as PPR
from src.models.season import OUT_DIR
from src.scoring.config import PRESETS, LeagueConfig, list_leagues
from src.scoring.engine import add_fantasy_points
from src.scoring.lineup import Player, optimal_lineup

POSITIONS = ["QB", "RB", "WR", "TE"]
REPLACEMENT_PLAYERS = 3   # average of the best 3 non-starters
TIER_GAP = 2.0            # x the mean gap among draftable players
MIN_PPR_PPG = 0.5         # below this the scoring factor is unreliable; use 1.0


def league_points(board: pl.DataFrame, config: LeagueConfig) -> pl.DataFrame:
    """Add `points`: projected season points under this league's scoring."""
    stats = board.select("position", *[pl.col(c).alias(c[3:]) for c in board.columns if c.startswith("pg_")])
    if stats.width == 1:  # no per-game stats on this board: PPR only
        return board.with_columns(points=pl.col("season_b"))
    league = add_fantasy_points(stats, config.scoring)["points"]
    ppr = add_fantasy_points(stats, PPR)["points"]
    factor = pl.when(ppr > MIN_PPR_PPG).then(league / ppr).otherwise(1.0)
    return board.with_columns(points=pl.col("season_b") * factor)


def league_roster(config: LeagueConfig):
    """Every team's starting slots added together (no K/DST: they aren't on our board)."""
    r = config.roster
    teams = config.league.teams
    return replace(r, **{s: getattr(r, s) * teams for s in [*POSITIONS, "FLEX", "SUPERFLEX"]}, K=0, DST=0, IDP=0, BENCH=0)


def replacement_levels(board: pl.DataFrame, config: LeagueConfig) -> dict[str, float]:
    players = [Player(pid, pos, pts) for pid, pos, pts in board.select("player_id", "position", "points").rows()]
    lineup = optimal_lineup(players, league_roster(config))
    starters = {p.id for slot in lineup.values() for p in slot}
    out = {}
    for pos in POSITIONS:
        bench = sorted((p.points for p in players if p.position == pos and p.id not in starters), reverse=True)
        out[pos] = sum(bench[:REPLACEMENT_PLAYERS]) / max(1, len(bench[:REPLACEMENT_PLAYERS])) if bench else 0.0
    return out


def tiers(board: pl.DataFrame, draftable: dict[str, int]) -> pl.DataFrame:
    """Tier number within position (1 = best), breaking at unusually big drops."""
    out = []
    for pos in POSITIONS:
        s = board.filter(pl.col("position") == pos).sort("points", descending=True)
        pts = s["points"].to_list()
        gaps = [a - b for a, b in zip(pts, pts[1:])]
        top = gaps[: max(1, draftable.get(pos, 12))]
        typical = sum(top) / len(top) if top else 0.0
        tier, labels = 1, []
        for i in range(len(pts)):
            labels.append(tier)
            if i < len(gaps) and typical > 0 and gaps[i] >= TIER_GAP * typical:
                tier += 1
        out.append(s.select("player_id").with_columns(tier=pl.Series(labels, dtype=pl.Int64)))
    return board.join(pl.concat(out), on="player_id", how="left")


def vor_board(board: pl.DataFrame, config: LeagueConfig) -> tuple[pl.DataFrame, dict[str, float]]:
    b = league_points(board, config)
    repl = replacement_levels(b, config)
    lr = league_roster(config)
    # draftable depth per position for tier spacing: starters at the position + a share of flex spots
    draftable = {p: getattr(lr, p) + (lr.FLEX if p != "QB" else 0) // 3 + lr.SUPERFLEX // 4 for p in POSITIONS}
    b = b.with_columns(replacement=pl.col("position").replace_strict(repl, default=0.0),
                       ).with_columns(vor=pl.col("points") - pl.col("replacement"))
    b = tiers(b, draftable).sort("vor", descending=True).with_columns(
        rank=pl.int_range(1, pl.len() + 1),
        pos_rank=pl.col("points").rank("ordinal", descending=True).over("position").cast(pl.Int64))
    return b, repl


def load_config(name: str) -> LeagueConfig:
    if name in PRESETS:
        return PRESETS[name]
    saved = list_leagues()
    if name in saved:
        return saved[name]
    raise SystemExit(f"unknown league '{name}': presets {list(PRESETS)}, saved {list(saved)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--league", default="standard-ppr")
    ap.add_argument("--season", type=int, default=2026)
    args = ap.parse_args()
    config = load_config(args.league)
    board = pl.read_parquet(OUT_DIR / f"board_{args.season}.parquet")
    b, repl = vor_board(board, config)
    print(f"{config.league.name} ({config.summary()}), {args.season}")
    print("replacement level (season points):", {k: round(v, 1) for k, v in repl.items()})
    for r in b.head(30).iter_rows(named=True):
        print(f"  {r['rank']:>3} {r['position']}{r['pos_rank']:<3} T{r['tier']:<2} {r['player_display_name']:24s} "
              f"{r['points']:6.1f} pts  VOR {r['vor']:6.1f}")


if __name__ == "__main__":
    main()
