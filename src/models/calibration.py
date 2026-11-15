"""Per-position calibration of season projections (SCRUM-59 follow-up, 2026-11-14).

The mock drafts showed our drafters under-taking RBs. This tests whether a scale fix helps:
for each position, fit   actual = a + b x projected   on the draft-relevant players (our top
24 QB / 48 RB / 60 WR / 24 TE by projection) from EARLIER seasons only, then apply it.
The first season with a board (2022) has no earlier season, so it stays uncalibrated.

Preview before the test: the fits are noisy (RB slope 1.30 from 2022 data, 0.91 from 2022-23),
which suggests the RB gap is about WHICH backs (ranking), not the scale. The mock draft decides.
"""

import numpy as np
import polars as pl

from src.models.season import OUT_DIR, PLAYER_WEEKS_PATH, season_results

POOL = {"QB": 24, "RB": 48, "WR": 60, "TE": 24}
SLOPE_BOUNDS = (0.5, 2.0)


def board_with_actuals(season: int, results: pl.DataFrame) -> pl.DataFrame:
    b = pl.read_parquet(OUT_DIR / f"board_{season}.parquet").select("player_id", "position", "season_b")
    j = b.join(results.filter(pl.col("season") == season).select("player_id", "points"), on="player_id", how="left")
    j = j.with_columns(pl.col("points").fill_null(0.0))
    rank = pl.col("season_b").rank(descending=True).over("position")
    return j.filter(rank <= pl.col("position").replace_strict(POOL, default=0))


def fit(training_seasons: list[int]) -> dict[str, tuple[float, float]]:
    """position -> (a, b). Empty if there are no training seasons."""
    if not training_seasons:
        return {}
    res = season_results(pl.read_parquet(PLAYER_WEEKS_PATH))
    data = pl.concat([board_with_actuals(s, res) for s in training_seasons])
    out = {}
    for pos in POOL:
        d = data.filter(pl.col("position") == pos)
        if d.height < 10:
            continue
        b, a = np.polyfit(d["season_b"].to_numpy(), d["points"].to_numpy(), 1)
        out[pos] = (float(a), float(np.clip(b, *SLOPE_BOUNDS)))
    return out


def apply(board: pl.DataFrame, params: dict[str, tuple[float, float]]) -> pl.DataFrame:
    """Replace season_b with the calibrated projection (never below 0)."""
    if not params:
        return board
    a = pl.col("position").replace_strict({p: v[0] for p, v in params.items()}, default=0.0)
    b = pl.col("position").replace_strict({p: v[1] for p, v in params.items()}, default=1.0)
    return board.with_columns(season_b=(a + b * pl.col("season_b")).clip(lower_bound=0.0))
