"""Player x defense matchup features (SCRUM-37).

One row per player_weeks row. Pairs what the player does with what this week's opponent allows.
Everything comes from earlier games (the player's and the defense's), same rule as SCRUM-34/36.

    opp_fpoe_pos_*        opponent's fantasy points above expected allowed to THIS player's position
    opp_pass_epa_*, opp_rush_epa_*, opp_epa_deep_*, opp_epa_short_*, opp_sack_rate_*
                          opponent's ratings from SCRUM-36 (windows _l5, _std, _prev)
    deep_tgt_share_*      share of the player's targets that were deep passes
    deep_matchup          deep_tgt_share x opponent deep EPA allowed   (deep threat vs leaky deep defense)
    short_matchup         short share x opponent short EPA allowed
    rb_checkdown          RB target share x opponent sack rate         (pass rush forces dump-offs; RBs only)

The *_matchup products use season-to-date values, falling back to last season when there are no
games yet this season. Ridge needs these products spelled out; boosting could find them alone.

Usage:
    python -m src.features.matchup
"""

from pathlib import Path

import polars as pl

from src.ingest.pbp import read_pbp

PLAYER_WEEKS_PATH = Path("data/processed/player_weeks.parquet")
USAGE_PATH = Path("data/processed/features_usage.parquet")
DEFENSE_PATH = Path("data/processed/features_defense.parquet")
OUT_PATH = Path("data/processed/features_matchup.parquet")

KEY = ["player_id", "season", "week"]
GROUP = ["player_id", "season"]
WINDOWS = ["l5", "std", "prev"]
# our column prefix -> SCRUM-36 rating name
OPP_RATINGS = {"opp_pass_epa": "epa_per_dropback", "opp_rush_epa": "epa_per_rush",
               "opp_epa_deep": "epa_deep", "opp_epa_short": "epa_short", "opp_sack_rate": "sack_rate"}
PBP_COLS = ["game_id", "season_type", "pass_attempt", "sack", "receiver_player_id", "pass_length"]


def target_depths(pbp: pl.DataFrame) -> pl.DataFrame:
    """Deep and short targets per player-game."""
    targets = pbp.filter(
        (pl.col("season_type") == "REG") & (pl.col("pass_attempt") == 1) & (pl.col("sack") != 1)
        & pl.col("receiver_player_id").is_not_null() & pl.col("pass_length").is_in(["deep", "short"])
    )
    return targets.group_by("game_id", pl.col("receiver_player_id").alias("player_id")).agg(
        deep_tgts=(pl.col("pass_length") == "deep").sum(), short_tgts=(pl.col("pass_length") == "short").sum(),
    )


def deep_share_features(pw: pl.DataFrame, depths: pl.DataFrame) -> pl.DataFrame:
    """deep_tgt_share_l5/_std/_prev from the player's earlier games (left-injured games skipped)."""
    games = (pw.filter(~pl.col("left_injured")).select(KEY + ["game_id"])
             .join(depths, on=["game_id", "player_id"], how="left")
             .with_columns(pl.col("deep_tgts", "short_tgts").fill_null(0)).sort(KEY))

    def share(deep: pl.Expr, short: pl.Expr) -> pl.Expr:
        return pl.when(deep + short > 0).then(deep / (deep + short))

    rolled = games.select(
        KEY,
        deep_tgt_share_l5=share(pl.col("deep_tgts").rolling_sum(5, min_samples=1).over(GROUP),
                                pl.col("short_tgts").rolling_sum(5, min_samples=1).over(GROUP)),
        deep_tgt_share_std=share(pl.col("deep_tgts").cum_sum().over(GROUP), pl.col("short_tgts").cum_sum().over(GROUP)),
    )
    current = pw.select(KEY).sort("week").join_asof(
        rolled.sort("week"), on="week", by=GROUP, strategy="backward", allow_exact_matches=False, check_sortedness=False,
    )
    prev = games.group_by(GROUP).agg(pl.sum("deep_tgts"), pl.sum("short_tgts")).select(
        "player_id", (pl.col("season") + 1).alias("season"),
        deep_tgt_share_prev=share(pl.col("deep_tgts"), pl.col("short_tgts")),
    )
    return current.join(prev, on=GROUP, how="left")


def opponent_ratings(pw: pl.DataFrame, defense: pl.DataFrame) -> pl.DataFrame:
    """This week's opponent's ratings, with fantasy points allowed matched to the player's position."""
    keep = [f"{r}_{w}" for r in OPP_RATINGS.values() for w in WINDOWS]
    keep += [f"fpoe_{p}_{w}" for p in ["qb", "rb", "wr", "te"] for w in WINDOWS]
    joined = pw.select(KEY + ["game_id", "opponent_team", "position"]).join(
        defense.select(["game_id", "defteam"] + keep),
        left_on=["game_id", "opponent_team"], right_on=["game_id", "defteam"], how="left",
    )
    pos = pl.col("position").str.to_lowercase()
    fpoe = [
        pl.coalesce([pl.when(pos == p).then(pl.col(f"fpoe_{p}_{w}")) for p in ["qb", "rb", "wr", "te"]]).alias(f"opp_fpoe_pos_{w}")
        for w in WINDOWS
    ]
    renamed = [pl.col(f"{r}_{w}").alias(f"{ours}_{w}") for ours, r in OPP_RATINGS.items() for w in WINDOWS]
    return joined.select(KEY + ["position"] + fpoe + renamed)


def latest(prefix: str) -> pl.Expr:
    """Season-to-date value, or last season's when there are no games yet."""
    return pl.coalesce(pl.col(f"{prefix}_std"), pl.col(f"{prefix}_prev"))


def build_matchup_features(pw: pl.DataFrame, usage: pl.DataFrame, defense: pl.DataFrame, depths: pl.DataFrame) -> pl.DataFrame:
    feats = (
        opponent_ratings(pw, defense)
        .join(deep_share_features(pw, depths), on=KEY, how="left")
        .join(usage.select(KEY + ["target_share_std", "target_share_prev"]), on=KEY, how="left")
    )
    deep = latest("deep_tgt_share")
    return feats.with_columns(
        deep_matchup=deep * latest("opp_epa_deep"),
        short_matchup=(1 - deep) * latest("opp_epa_short"),
        rb_checkdown=pl.when(pl.col("position") == "RB").then(latest("target_share") * latest("opp_sack_rate")),
    ).drop("target_share_std", "target_share_prev").sort(KEY)


def check_matchup_features(feats: pl.DataFrame, pw: pl.DataFrame) -> list[str]:
    problems = []
    if feats.height != pw.height:
        problems.append(f"{feats.height} rows vs {pw.height} player_weeks rows")
    if feats.group_by(KEY).len().filter(pl.col("len") > 1).height:
        problems.append("duplicate player-weeks")
    shares = [f"deep_tgt_share_{w}" for w in WINDOWS]
    if feats.filter(pl.any_horizontal([(pl.col(c) < 0) | (pl.col(c) > 1) for c in shares])).height:
        problems.append("deep target share outside 0-1")
    return problems


def main() -> None:
    pw = pl.read_parquet(PLAYER_WEEKS_PATH)
    feats = build_matchup_features(pw, pl.read_parquet(USAGE_PATH), pl.read_parquet(DEFENSE_PATH),
                                   target_depths(read_pbp(columns=PBP_COLS)))
    problems = check_matchup_features(feats, pw)
    for p in problems:
        print("CHECK FAILED:", p)
    if problems:
        raise SystemExit(1)
    feats.write_parquet(OUT_PATH)
    print(f"Wrote {OUT_PATH}: {feats.height:,} rows x {feats.width} columns")


if __name__ == "__main__":
    main()
