"""Decision accuracy vs market consensus (SCRUM-55).

Rules (Lucas, 2026-11-14, set before seeing any results):
    outcome   total PPR points in the regular season (what a pick gives you; 0 if he never played).
              Points per game (both players 6+ games) is reported as a secondary view.
    pairs     same position, within 12 overall spots of each other in the market's ranking
              (about one draft round): "RB14 or RB16 at this pick?" Cross-position value is VOR's job.
    headline  on pairs where our board and the market DISAGREE, how often are we the one who's
              right? 50% = coin flip against the market. Pairs where both agree score the same for
              both, so they can't separate us.
    CI        95% player-level bootstrap: players are resampled (pairs share players, so pairs
              aren't independent), each pair weighted by how often both its players were drawn.
    success   above 50% with the CI excluding 50%.
    markets   ECR (FantasyPros PPR expert consensus, last snapshot before Week 1) is primary; ADP
              (where real drafters took players) is secondary.
    seasons   2022-2024 (development). 2025 is the final exam (SCRUM-66).

Players the market ranked but who weren't on any Week 1 roster (unsigned, retired) have no row on
our board; their pairs are dropped and counted, which can only cost us wins, never add them.

Usage:
    python -m src.backtest.vs_consensus
"""

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from src.ingest.consensus import load_adp, load_ecr
from src.models.season import EXAM_SEASON, OUT_DIR, PLAYER_WEEKS_PATH, season_results

SEASONS = [2022, 2023, 2024]
POOL = 200          # market's top-200 skill players: the draftable pool
WINDOW = 12         # "within one round"
MIN_GAMES_PPG = 6
N_BOOT = 2000
REPORT_DIR = Path("docs/reports")


def pairs(players: pl.DataFrame, market_col: str) -> pl.DataFrame:
    """All same-position pairs within WINDOW market spots. One row per unordered pair (a ranked ahead of b)."""
    p = players.with_columns(idx=pl.int_range(pl.len()))
    j = p.join(p, on=["season", "position"], suffix="_b").filter(
        pl.col(market_col) < pl.col(f"{market_col}_b"),
        (pl.col(f"{market_col}_b") - pl.col(market_col)) <= WINDOW)
    return j


def score_pairs(j: pl.DataFrame, outcome: str, ours: str = "season_b") -> pl.DataFrame:
    """Market says a > b (a ranked ahead). We say a > b if our score (`ours`, higher = better) is
    higher; a tie in our score goes to the market's order. Keep pairs with a real winner (no tie in the outcome)."""
    return (j.filter(pl.col(outcome) != pl.col(f"{outcome}_b"))
            .with_columns(a_won=pl.col(outcome) > pl.col(f"{outcome}_b"),
                          we_say_a=pl.col(ours) >= pl.col(f"{ours}_b"))
            .with_columns(market_right=pl.col("a_won"), we_right=pl.col("we_say_a") == pl.col("a_won"),
                          disagree=~pl.col("we_say_a")))


def bootstrap_rate(scored: pl.DataFrame, value: str, seed: int = 0) -> tuple[float, float, float]:
    """Mean of `value` over pairs, with a player-level cluster bootstrap CI."""
    if scored.is_empty():
        return float("nan"), float("nan"), float("nan")
    keys = pl.concat([scored.select(k=pl.format("{}|{}", "season", "player_id")),
                      scored.select(k=pl.format("{}|{}", "season", "player_id_b"))]).unique()["k"].to_list()
    index = {k: i for i, k in enumerate(keys)}
    a = np.array([index[f"{s}|{p}"] for s, p in scored.select("season", "player_id").rows()])
    b = np.array([index[f"{s}|{p}"] for s, p in scored.select("season", "player_id_b").rows()])
    v = scored[value].cast(pl.Float64).to_numpy()
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(N_BOOT):
        m = np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys)).astype(float)
        w = m[a] * m[b]
        if w.sum() > 0:
            rates.append((w * v).sum() / w.sum())
    return float(v.mean()), float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def build_players(market: str) -> tuple[pl.DataFrame, dict]:
    """Market top-POOL per season, joined to our board and the season's actual results."""
    if EXAM_SEASON in SEASONS:
        raise ValueError("2025 is the final exam")
    res = season_results(pl.read_parquet(PLAYER_WEEKS_PATH))
    out, dropped = [], {}
    for season in SEASONS:
        m = (load_ecr(season).rename({"ecr": "rank"}) if market == "ecr" else load_adp(season).rename({"adp": "rank"}))
        m = m.drop_nulls("player_id").unique("player_id", keep="first").sort("rank").head(POOL)
        board = pl.read_parquet(OUT_DIR / f"board_{season}.parquet").select("player_id", "season_b")
        j = m.select("player_id", "position", "rank").join(board, on="player_id", how="left")
        dropped[season] = j["season_b"].null_count()
        j = j.drop_nulls("season_b").with_columns(season=pl.lit(season))
        r = res.filter(pl.col("season") == season).select("player_id", "games", "points")
        j = j.join(r, on="player_id", how="left").with_columns(pl.col("games", "points").fill_null(0))
        out.append(j.with_columns(ppg=pl.when(pl.col("games") >= MIN_GAMES_PPG)
                                  .then(pl.col("points") / pl.col("games"))))
    return pl.concat(out), dropped


def summarize(scored: pl.DataFrame, label: str) -> dict:
    dis = scored.filter("disagree")
    win, lo, hi = bootstrap_rate(dis, "we_right")
    return {"group": label, "pairs": scored.height, "disagree": dis.height, "win_rate": win, "lo": lo, "hi": hi,
            "ours_acc": scored["we_right"].mean(), "market_acc": scored["market_right"].mean()}


def add_blend(players: pl.DataFrame) -> pl.DataFrame:
    """Blend (set 2026-11-14 before looking): average of our position rank and the market's position
    rank, 50/50, within each season's pool. `blend` is negated so higher = better, like season_b."""
    over = ["season", "position"]
    return players.with_columns(
        blend=-(pl.col("season_b").rank("average", descending=True).over(over)
                + pl.col("rank").rank("average").over(over)) / 2)


def run(market: str, ours: str = "season_b") -> tuple[list[dict], list[dict], dict]:
    players, dropped = build_players(market)
    j = pairs(add_blend(players), "rank")
    main = score_pairs(j, "points", ours)
    ppg = score_pairs(j.filter(pl.col("ppg").is_not_null(), pl.col("ppg_b").is_not_null()), "ppg", ours)
    rows = [summarize(main, "ALL")]
    rows += [summarize(main.filter(pl.col("position") == p), p) for p in ["QB", "RB", "WR", "TE"]]
    rows += [summarize(main.filter(pl.col("season") == s), str(s)) for s in SEASONS]
    return rows, [summarize(ppg, "ALL (PPG, 6+ games)")], dropped


def table(rows: list[dict]) -> list[str]:
    out = ["| group | pairs | we disagree | **we're right when we disagree** [95% CI] | our accuracy | market accuracy |",
           "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        out.append(f"| {r['group']} | {r['pairs']} | {r['disagree']} | **{r['win_rate']:.1%}** [{r['lo']:.1%}, {r['hi']:.1%}] "
                   f"| {r['ours_acc']:.1%} | {r['market_acc']:.1%} |")
    return out


def main() -> None:
    lines = [f"# Decision accuracy vs market consensus ({date.today()})", "",
             f"Seasons {SEASONS} (2025 untouched). Market top-{POOL} QB/RB/WR/TE; same-position pairs within "
             f"{WINDOW} market spots; outcome = total PPR season points. Headline: on pairs where we and the market "
             "disagree, how often we're right (50% = coin flip). CI: player-level bootstrap.", ""]
    for market, ours, name in [("ecr", "season_b", "our board vs ECR (FantasyPros expert consensus, primary)"),
                               ("adp", "season_b", "our board vs ADP (where drafters took players)"),
                               ("ecr", "blend", "50/50 blend of our board + ECR vs ECR alone")]:
        rows, ppg, dropped = run(market, ours)
        lines += [f"## {name}", "", *table(rows), "", "Secondary view, points per game:", "", *table(ppg), "",
                  f"Market top-{POOL} players not on our board (no Week 1 roster spot), pairs dropped: "
                  + ", ".join(f"{s}: {n}" for s, n in dropped.items()), ""]
    text = "\n".join(lines) + "\n"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"vs_consensus_{date.today()}.md"
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
