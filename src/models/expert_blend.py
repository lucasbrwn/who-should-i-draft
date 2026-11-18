"""The weekly number the product shows: our expected points averaged with ESPN and FantasyPros.

Passed its locked test 2026-11-18 (src/backtest/expert_blend.py, docs/reports/expert_blend_2026-11-17.md):
2022-24, vs FantasyPros 53.0% [50.9, 54.9], vs ESPN 52.9% [51.1, 54.6]; beats the experts-only average
on points error by 0.027 [0.014, 0.039], so our model adds value on top of the experts.

    blend = plain average of whichever exist: ours (proj), espn_proj, fp_pts.  Equal weights, never tuned.
    fp_pts = FantasyPros position ECR turned into points with a per-position decreasing isotonic fit of
             actual PPR points on ECR, fit on 2022-24 (the backtest's seasons; never 2025).
             Saved as a lookup table in models/fp_rank_points.json so every run uses the same map.

    python -m src.models.expert_blend      # (re)fit and save the FantasyPros rank -> points table
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from src.backtest.weekly_vs_consensus import POOL, game_days, weekly_ecr

MAP_PATH = Path("models/fp_rank_points.json")
FIT_SEASONS = [2022, 2023, 2024]
SOURCES = ["proj", "espn_proj", "fp_pts"]


def fit_rank_points(seasons: list[int] = FIT_SEASONS) -> dict:
    """{position: {"rank": [...], "points": [...]}}: isotonic step points to interpolate between."""
    from sklearn.isotonic import IsotonicRegression

    if 2025 in seasons:
        raise ValueError("2025 is the final exam")
    pw = pl.read_parquet("data/processed/player_weeks.parquet", columns=["player_id", "season", "week", "fantasy_points_ppr"])
    rows = (weekly_ecr(seasons).join(game_days(seasons), on=["player_id", "season", "week"], how="inner")
            .filter(pl.col("gameday") > pl.col("d"))
            .join(pw, on=["player_id", "season", "week"], how="inner").drop_nulls(["rank", "fantasy_points_ppr"]))
    out = {}
    for pos in POOL:
        p = rows.filter(pl.col("position") == pos)
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(
            p["rank"].to_numpy(), p["fantasy_points_ppr"].to_numpy())
        out[pos] = {"rank": [round(float(x), 3) for x in iso.X_thresholds_],
                    "points": [round(float(y), 3) for y in iso.y_thresholds_], "n": p.height}
    return out


def load_map(path: Path = MAP_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rank_to_points(position: str, ecr: float | None, table: dict) -> float | None:
    if ecr is None or position not in table:
        return None
    t = table[position]
    return float(np.interp(ecr, t["rank"], t["points"]))   # flat beyond the ends, like out_of_bounds="clip"


def blend(preds: pl.DataFrame, espn: pl.DataFrame, fp: pl.DataFrame, table: dict) -> pl.DataFrame:
    """preds needs player_id, position, proj; espn: player_id, espn_proj; fp: player_id, fp_ecr."""
    espn = espn.drop_nulls("player_id").unique("player_id").select("player_id", "espn_proj")
    fp = fp.drop_nulls("player_id").unique("player_id").select("player_id", "fp_ecr")
    out = preds.join(espn, on="player_id", how="left").join(fp, on="player_id", how="left")
    out = out.with_columns(fp_pts=pl.struct("position", "fp_ecr").map_elements(
        lambda r: rank_to_points(r["position"], r["fp_ecr"], table), return_dtype=pl.Float64).round(1))
    return out.with_columns(
        blend=pl.mean_horizontal(SOURCES).round(1),
        blend_sources=pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int8) for c in SOURCES]),
    ).sort("blend", descending=True)


def main() -> None:
    table = fit_rank_points()
    MAP_PATH.write_text(json.dumps(table, indent=1), encoding="utf-8")
    for pos in POOL:
        print(pos, {r: round(rank_to_points(pos, r, table), 1) for r in (1, 6, 12, 24, 36)}, f"n={table[pos]['n']}")
    print(f"Wrote {MAP_PATH}")


if __name__ == "__main__":
    main()
