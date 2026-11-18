"""Head-to-head vs ESPN projections and FantasyPros experts, plus WHERE we lose.

Seasons: 2026 weeks 1-10 (first run), and 2022-2024 (added 2026-11-17 for a bigger sample, same rules;
chance-he-plays rates for each test season come from the OTHER seasons only). 2025 is the final exam.

Rules (set 2026-11-17 before results; same as weekly_vs_consensus.py so results compare to its 49.8%):
    ours      walk-forward weekly boosting (each week trained only on earlier games). Headline uses
              what the product shows: expected points = projection x chance he plays
              (src/models/play_chance.py, rates from 2021-24). Raw projection reported too.
    ESPN      ESPN's weekly PPR projection for that week (public data; ESPN's final number, which may
              include news after our Friday-report lock, so if anything it favors ESPN)
    experts   FantasyPros weekly ECR, Friday snapshot; Thursday games dropped
    pool      the benchmark's weekly top 24 QB / 48 RB / 60 WR / 24 TE, players who played
    pairs     same week + position, within 6 of the benchmark's ranks; outcome = actual PPR points
    headline  when we disagree with the benchmark, how often we're right (player-level bootstrap CI)
    also      points error (MAE) vs ESPN on the same players
    breakdown each disagreement is tagged by the player WE favored (rookie, injury report, backup on
              the depth chart, teammate starter out, returning from a missed game, position)

    python -m src.backtest.vs_espn
"""

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from src.backtest.harness import run
from src.backtest.vs_consensus import bootstrap_rate, score_pairs
from src.backtest.weekly_vs_consensus import POOL, WINDOW, game_days, weekly_ecr
from src.features.team import read_schedules
from src.models.boosting import TUNED_PATH, BoostingModel
from src.models.play_chance import PlayChance, game_day_pool
from src.predict.benchmarks import espn_projections

WEEKS = {2022: list(range(1, 19)), 2023: list(range(1, 19)), 2024: list(range(1, 19)), 2026: list(range(1, 11))}
RATE_POOL = [2021, 2022, 2023, 2024]
FEATURES_PATH = Path("data/processed/features.parquet")
CACHE = Path("data/raw/espn")
OURS_DIR = Path("data/processed")
REPORT_DIR = Path("docs/reports")
TAG_COLS = ["years_exp", "report_status", "depth_rank", "vacated_carry_share", "vacated_tgt_share_total", "missed_last"]


def espn_history(season: int) -> pl.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    out = []
    for week in WEEKS[season]:
        path = CACHE / f"espn_proj_{season}_wk{week:02d}.parquet"
        if not path.exists():
            espn_projections(season, week).write_parquet(path)
        out.append(pl.read_parquet(path).with_columns(week=pl.lit(week, pl.Int32)))
    return pl.concat(out).drop_nulls("player_id").unique(["player_id", "week"]).with_columns(season=pl.lit(season, pl.Int32))


def missed_last_game(pw: pl.DataFrame) -> pl.DataFrame:
    """True if his team played the week before and he didn't (returning from an absence)."""
    team_weeks = pw.select("season", "team", "week").unique()
    prev = team_weeks.sort("week").with_columns(prev_week=pl.col("week").shift(1).over("season", "team"))
    rows = pw.select("player_id", "season", "week", "team").join(prev, on=["season", "team", "week"], how="left")
    played = pw.select("player_id", "season", prev_week="week").with_columns(played_prev=pl.lit(True))
    return rows.join(played, on=["player_id", "season", "prev_week"], how="left").select(
        "player_id", "season", "week",
        missed_last=pl.col("prev_week").is_not_null() & pl.col("played_prev").is_null())


def our_predictions(season: int) -> pl.DataFrame:
    feats = pl.read_parquet(FEATURES_PATH)
    path = OURS_DIR / f"weekly_predictions_{season}.parquet"
    if not path.exists():
        res = run(BoostingModel(tuned=TUNED_PATH, name="boosting"), feats.filter(pl.col("season") <= season), [season])
        res.write_parquet(path)
    res = pl.read_parquet(path).filter(pl.col("week").is_in(WEEKS[season]))
    info = feats.select("player_id", "season", "week", "depth_rank", "report_status", "practice_code", "years_exp",
                        "vacated_carry_share", "vacated_tgt_share_total")
    res = res.join(info, on=["player_id", "season", "week"], how="left")
    chance = PlayChance().fit(game_day_pool([s for s in RATE_POOL if s != season], read_schedules()))
    res = chance.rates(res).with_columns(expected=pl.col("pred") * pl.col("p_play"))
    pw = pl.read_parquet("data/processed/player_weeks.parquet").filter(pl.col("season") == season)
    return res.join(missed_last_game(pw), on=["player_id", "season", "week"], how="left")


def pool(players: pl.DataFrame) -> pl.DataFrame:
    return players.filter(pl.col("rank").rank("ordinal").over("season", "week", "position")
                          <= pl.col("position").replace_strict(POOL, default=0))


def weekly_pairs(p: pl.DataFrame) -> pl.DataFrame:
    return p.join(p, on=["season", "week", "position"], suffix="_b").filter(
        pl.col("rank") < pl.col("rank_b"), (pl.col("rank_b") - pl.col("rank")) <= WINDOW)


def tag_pick(scored: pl.DataFrame) -> pl.DataFrame:
    """Situation of the player WE favored in each pair (a if we said a, else b)."""
    pick = lambda c: pl.when(pl.col("we_say_a")).then(pl.col(c)).otherwise(pl.col(f"{c}_b"))  # noqa: E731
    return scored.with_columns(
        rookie=pick("years_exp") == 0,
        on_report=pick("report_status").is_in(["Questionable", "Doubtful"]).fill_null(False),
        backup=(pick("depth_rank") >= 2).fill_null(False),
        starter_out=((pick("vacated_carry_share") > 0.2) | (pick("vacated_tgt_share_total") > 0.2)).fill_null(False),
        returning=pick("missed_last").fill_null(False),
    )


def summary(scored: pl.DataFrame, label: str) -> str:
    dis = scored.filter("disagree")
    win, lo, hi = bootstrap_rate(dis, "we_right")
    return (f"| {label} | {scored.height} | {dis.height} | **{win:.1%}** [{lo:.1%}, {hi:.1%}] "
            f"| {scored['we_right'].mean():.1%} | {scored['market_right'].mean():.1%} |")


def report(name: str, p: pl.DataFrame, ours: str) -> list[str]:
    scored = tag_pick(score_pairs(weekly_pairs(pool(p)), "actual", ours=ours))
    lines = [f"### vs {name} (ours = {ours})", "",
             "| group | pairs | we disagree | **we're right when we disagree** [95% CI] | our accuracy | their accuracy |",
             "| --- | --- | --- | --- | --- | --- |", summary(scored, "ALL")]
    lines += [summary(scored.filter(pl.col("position") == pos), pos) for pos in POOL]
    lines += ["", "Where we disagree, by the situation of the player WE favored:", "",
              "| our pick is... | disagreements | we're right [95% CI] |", "| --- | --- | --- |"]
    dis = scored.filter("disagree")
    for tag in ["rookie", "on_report", "backup", "starter_out", "returning"]:
        for val in (True, False):
            g = dis.filter(pl.col(tag) == val)
            if g.height >= 20:
                w, lo, hi = bootstrap_rate(g, "we_right")
                lines.append(f"| {tag} = {val} | {g.height} | {w:.1%} [{lo:.1%}, {hi:.1%}] |")
    return lines + [""]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", default=[2026])
    args = parser.parse_args()
    if 2025 in args.seasons:
        raise ValueError("2025 is the final exam")
    keep = ["player_id", "season", "week", "position", "actual", "pred", "expected", *TAG_COLS]
    es, fs = [], []
    for season in args.seasons:
        ours = our_predictions(season)
        e = ours.select(keep).join(espn_history(season).select("player_id", "season", "week", "espn_proj"),
                                   on=["player_id", "season", "week"])
        es.append(e.with_columns(rank=pl.col("espn_proj").rank("ordinal", descending=True).over("season", "week", "position")))
        days = game_days([season])
        fp = (weekly_ecr([season]).join(days, on=["player_id", "season", "week"], how="inner")
              .filter(pl.col("gameday") > pl.col("d")).select("player_id", "season", "week", "rank"))
        fs.append(ours.select(keep).join(fp, on=["player_id", "season", "week"]))
        print(f"{season}: done", flush=True)
    e, f = pl.concat(es), pl.concat(fs)

    pe = pool(e)
    mae = {c: np.abs(pe[c].to_numpy() - pe["actual"].to_numpy()).mean() for c in ["expected", "pred", "espn_proj"]}
    by_pos = pe.group_by("position").agg(*[(pl.col(c) - pl.col("actual")).abs().mean().round(2).alias(c)
                                          for c in ["expected", "pred", "espn_proj"]]).sort("position")
    label = "+".join(str(x) for x in args.seasons)
    lines = [f"# Head-to-head vs ESPN and FantasyPros, seasons {label} ({date.today()})", "",
             __doc__.split("\n\n")[2], "",
             f"Points error on ESPN's pool ({pe.height} player-weeks): ours (expected) {mae['expected']:.2f}, "
             f"ours (raw) {mae['pred']:.2f}, ESPN {mae['espn_proj']:.2f}", "",
             "| position | ours expected | ours raw | ESPN |", "| --- | --- | --- | --- |",
             *[f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in by_pos.rows()], ""]
    if len(args.seasons) > 1:
        lines += ["By season (vs ESPN, ours = expected):", "",
                  "| group | pairs | we disagree | **we're right when we disagree** [95% CI] | our accuracy | their accuracy |",
                  "| --- | --- | --- | --- | --- | --- |"]
        sc = tag_pick(score_pairs(weekly_pairs(pool(e)), "actual", ours="expected"))
        lines += [summary(sc.filter(pl.col("season") == y), str(y)) for y in args.seasons] + [""]
    lines += report("ESPN", e, "expected") + report("ESPN", e, "pred")
    lines += report("FantasyPros experts", f, "expected") + report("FantasyPros experts", f, "pred")
    path = REPORT_DIR / f"vs_espn_{label}_{date.today()}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
