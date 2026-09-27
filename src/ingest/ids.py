"""Player ID translation. Weekly stats and injuries use the NFL's gsis_id ("00-0039521");
snap counts use Pro Football Reference IDs ("BrowSp00")."""

import nflreadpy
import polars as pl


def normalize_name(name: pl.Expr) -> pl.Expr:
    """'Brian Thomas Jr.' and 'C.J. Stroud' -> 'brian thomas', 'cj stroud'."""
    return (
        name.str.to_lowercase()
        .str.replace_all(r"[.'\-]", "")
        .str.replace(r"\s+(jr|sr|ii|iii|iv|v)$", "")
        .str.strip_chars()
    )


def pfr_to_gsis() -> pl.DataFrame:
    """One row per PFR ID with its gsis_id, from nflverse's player directory."""
    return (
        nflreadpy.load_players()
        .select(pl.col("pfr_id").alias("pfr_player_id"), "gsis_id")
        .drop_nulls()
        .unique("pfr_player_id")
    )


def attach_gsis_id(snaps: pl.DataFrame, id_map: pl.DataFrame, player_stats: pl.DataFrame) -> pl.DataFrame:
    """Add gsis_id to snap counts: directory lookup first, then an exact name match within the
    same game and team for anyone left. `id_source` records which method matched."""
    mapped = snaps.join(id_map, on="pfr_player_id", how="left").with_columns(
        pl.when(pl.col("gsis_id").is_not_null()).then(pl.lit("directory")).alias("id_source")
    )

    candidates = (
        player_stats.select(
            "game_id", "team", pl.col("player_id").alias("name_gsis_id"),
            normalize_name(pl.col("player_display_name")).alias("name_key"),
        )
        # a name that appears twice on one team in one game is ambiguous; don't guess
        .filter(pl.len().over("game_id", "team", "name_key") == 1)
    )
    return (
        mapped.with_columns(normalize_name(pl.col("player")).alias("name_key"))
        .join(candidates, on=["game_id", "team", "name_key"], how="left")
        .with_columns(
            pl.coalesce("gsis_id", "name_gsis_id").alias("gsis_id"),
            pl.when(pl.col("id_source").is_null() & pl.col("name_gsis_id").is_not_null())
            .then(pl.lit("name_match"))
            .otherwise(pl.col("id_source"))
            .alias("id_source"),
        )
        .drop("name_key", "name_gsis_id")
    )
