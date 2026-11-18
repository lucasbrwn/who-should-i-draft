"""Routes-run features (SCRUM-87): how often a player runs a route, and what he does with it.

Routes run = team dropbacks the player was on the field for (nflverse participation lists all 11
offensive players on every play). Checked 2026-11-14 against SumerSports' true routes run for the
top-25 WRs of 2023, 2024 and 2025 (75 player-seasons): average miss 0.5-0.8%, correlation 0.999.
For TEs and RBs it overcounts a little (a pass-blocking snap counts as a route) and is unvalidated.

    routes_*                avg routes run per game
    route_participation_*   routes / team dropbacks while he played (the role signal)
    targets_per_route_*     targets / routes (TPRR)
    yards_per_route_*       receiving yards / routes (YPRR, "the gold standard" receiver efficiency)

Windows match usage.py: l3, l5, std (this season, games before this week) and prev (last season).
Games where he left injured are left out. Spikes don't count as dropbacks.

Participation data stops at 2025 (nflverse publishes it after the season). For games without it
(2026), routes are ESTIMATED (estimate_routes): offense snaps x the team's dropbacks / plays x a
per-position scale fit on games that do have participation. Checked 2026-11-17 (scale fit on 2024,
scored on 2025): correlation with real routes WR .98, TE .96, RB .96; typical miss 2-3 routes a game.
Tested 2026-11-17 (src/backtest/route_estimate_test.py, 2024 replayed with routes missing): estimate vs
blank WR Spearman +.004 [.001,.008], nothing worse -> ON by default. Ceiling (real routes vs blank)
is small: WR MAE -0.021, TE -0.034, so paying for charted 2026 routes isn't worth it.

Usage:
    python -m src.features.routes
"""

from pathlib import Path

import polars as pl

from src.features.usage import GROUP, ID_COLS, KEY, WINDOWS

PLAYER_WEEKS_PATH = Path("data/processed/player_weeks.parquet")
PARTICIPATION_DIR = Path("data/raw/participation")
OUT_PATH = Path("data/processed/features_routes.parquet")
PBP_COLS = ["game_id", "play_id", "season_type", "posteam", "qb_dropback", "qb_spike", "play_type"]

MEANS = ["routes"]
RATIOS = {
    "route_participation": ("routes", "team_dropbacks"),
    "targets_per_route": ("targets", "routes"),
    "yards_per_route": ("receiving_yards", "routes"),
}
SUMS = list(dict.fromkeys(MEANS + [c for pair in RATIOS.values() for c in pair]))
FEATURES = [f"{name}_{w}" for name in [*MEANS, *RATIOS] for w in [*WINDOWS, "std", "prev"]]


def read_participation(seasons: list[int]) -> pl.DataFrame:
    paths = [PARTICIPATION_DIR / f"participation_{s}.parquet" for s in seasons]
    # play_id is a float in some seasons' files and an int in others: cast each file before stacking.
    frames = [pl.read_parquet(p, columns=["nflverse_game_id", "play_id", "offense_players"])
              .select(game_id="nflverse_game_id", play_id=pl.col("play_id").cast(pl.Int64), offense_players="offense_players")
              for p in paths if p.exists()]
    if not frames:
        return pl.DataFrame(schema={"game_id": pl.String, "play_id": pl.Int64, "offense_players": pl.String})
    return pl.concat(frames)


def routes_per_game(participation: pl.DataFrame, pbp: pl.DataFrame) -> pl.DataFrame:
    """game_id, team, player_id, routes + the team's dropbacks in that game."""
    dropbacks = (pbp.filter(pl.col("season_type") == "REG", pl.col("qb_dropback") == 1, pl.col("qb_spike") != 1)
                 .select("game_id", pl.col("play_id").cast(pl.Int64), team="posteam"))
    plays = dropbacks.join(participation, on=["game_id", "play_id"], how="inner")
    team = plays.group_by("game_id", "team").agg(team_dropbacks=pl.len())
    players = (plays.with_columns(player_id=pl.col("offense_players").str.split(";")).explode("player_id", empty_as_null=True)
               .filter(pl.col("player_id").str.len_chars() > 0)
               .group_by("game_id", "team", "player_id").agg(routes=pl.len()))
    return team.join(players, on=["game_id", "team"], how="left")


def add_route_columns(pw: pl.DataFrame, per_game: pl.DataFrame) -> pl.DataFrame:
    """Per-game routes and team dropbacks on each player-week. A game with participation data but
    no routes for him counts as 0 routes; a game with no participation data at all stays blank."""
    team = per_game.select("game_id", "team", "team_dropbacks").unique()
    out = (pw.join(team, on=["game_id", "team"], how="left")
           .join(per_game.select("game_id", "player_id", "routes"), on=["game_id", "player_id"], how="left"))
    return out.with_columns(routes=pl.when(pl.col("team_dropbacks").is_not_null()).then(pl.col("routes").fill_null(0)))


def team_play_counts(pbp: pl.DataFrame) -> pl.DataFrame:
    """Every team-game's dropbacks (spikes excluded, as in routes_per_game) and pass + run plays."""
    reg = pbp.filter(pl.col("season_type") == "REG")
    return reg.group_by("game_id", team="posteam").agg(
        est_dropbacks=((pl.col("qb_dropback") == 1) & (pl.col("qb_spike") != 1)).sum(),
        est_plays=pl.col("play_type").is_in(["pass", "run"]).sum(),
    ).filter(pl.col("est_plays") > 0)


def estimate_routes(pw: pl.DataFrame, pbp: pl.DataFrame) -> pl.DataFrame:
    """Fill routes / team_dropbacks for games with no participation data:
    routes ~ offense snaps x team dropbacks / team plays x position scale (scale = real routes / raw
    estimate over the games that DO have participation, so it never uses the games being filled)."""
    counts = team_play_counts(pbp)
    pw = pw.join(counts, on=["game_id", "team"], how="left").with_columns(
        raw_est=pl.col("offense_snaps") * pl.col("est_dropbacks") / pl.col("est_plays"))
    known = pw.filter(pl.col("team_dropbacks").is_not_null() & pl.col("raw_est").is_not_null() & (pl.col("raw_est") > 0))
    scale = known.group_by("position").agg(scale=pl.col("routes").sum() / pl.col("raw_est").sum())
    missing = pl.col("team_dropbacks").is_null() & pl.col("est_dropbacks").is_not_null()
    return pw.join(scale, on="position", how="left").with_columns(
        routes=pl.when(missing).then((pl.col("raw_est") * pl.col("scale")).round(1)).otherwise(pl.col("routes")),
        team_dropbacks=pl.when(missing).then(pl.col("est_dropbacks")).otherwise(pl.col("team_dropbacks")),
    ).drop("est_dropbacks", "est_plays", "raw_est", "scale")


def finish(frame: pl.DataFrame, suffix: str, games: str) -> pl.DataFrame:
    s = lambda c: pl.col(f"{c}__sum")  # noqa: E731
    exprs = [(s(m) / pl.col(games)).alias(f"{m}_{suffix}") for m in MEANS]
    exprs += [pl.when(s(den) > 0).then(s(num) / s(den)).alias(f"{n}_{suffix}") for n, (num, den) in RATIOS.items()]
    return frame.with_columns([e.round(9) for e in exprs]).drop([f"{c}__sum" for c in SUMS] + [games])


def through_each_game(usable: pl.DataFrame) -> pl.DataFrame:
    """Window values INCLUDING each game; build_route_features attaches each week to the latest
    game strictly before it, which keeps the current game out (same as usage.py)."""
    usable = usable.sort(KEY)
    out = usable.select(KEY)
    for suffix, w in {**WINDOWS, "std": None}.items():
        roll = (lambda e: e.cum_sum().over(GROUP)) if w is None else (lambda e, w=w: e.rolling_sum(w, min_samples=1).over(GROUP))
        games = roll(pl.col("routes").is_not_null().cast(pl.Int32)).alias(f"g_{suffix}")
        frame = usable.select(KEY + [roll(pl.col(c).fill_null(0)).alias(f"{c}__sum") for c in SUMS] + [games])
        frame = finish(frame.with_columns(pl.when(pl.col(f"g_{suffix}") > 0).then(pl.col(f"g_{suffix}")).alias(f"g_{suffix}")),
                       suffix, f"g_{suffix}")
        out = out.join(frame, on=KEY)
    return out


def previous_season(usable: pl.DataFrame) -> pl.DataFrame:
    has = usable.filter(pl.col("routes").is_not_null())
    totals = has.group_by(GROUP).agg([pl.col(c).sum().alias(f"{c}__sum") for c in SUMS] + [pl.len().alias("g_prev")])
    return finish(totals, "prev", "g_prev").with_columns(pl.col("season") + 1)


def build_route_features(pw: pl.DataFrame, participation: pl.DataFrame, pbp: pl.DataFrame,
                         estimate: bool = True) -> pl.DataFrame:
    pw = add_route_columns(pw, routes_per_game(participation, pbp))
    if estimate:
        pw = estimate_routes(pw, pbp)
    usable = pw.filter(~pl.col("left_injured"))
    rows = pw.select(ID_COLS).sort("week")
    feats = rows.join_asof(through_each_game(usable).sort("week"), on="week", by=GROUP, strategy="backward",
                           allow_exact_matches=False, check_sortedness=False)
    return feats.join(previous_season(usable), on=GROUP, how="left").select(*KEY, *FEATURES).sort(KEY)


def main() -> None:
    from src.ingest.pbp import read_pbp

    pw = pl.read_parquet(PLAYER_WEEKS_PATH)
    seasons = sorted(pw["season"].unique().to_list())
    feats = build_route_features(pw, read_participation(seasons), read_pbp(seasons, columns=PBP_COLS))
    feats.write_parquet(OUT_PATH)
    filled = feats.group_by("season").agg(pl.col("route_participation_l3").is_not_null().mean().round(2)).sort("season")
    print(f"Wrote {OUT_PATH}: {feats.height:,} rows x {feats.width} columns")
    print("share of rows with route_participation_l3:", dict(filled.rows()))


if __name__ == "__main__":
    main()
