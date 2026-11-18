"""Tier calibration of weekly projections (weak-points finding, 2026-11-17).

The 2022-24 diagnosis showed a bend: the top 5% of projections ran high (RB -0.84, WR -0.66 points
actual minus projected) while the 80-95% tier ran low (+0.35 to +0.41). A straight-line fix can't
undo a bend, so per position we fit an ISOTONIC curve: actual points as a non-decreasing function of
the projection, learned from earlier seasons' walk-forward predictions (the model's own past
mistakes). Non-decreasing means it can reshape tiers but never reorder two players.

Above the 99th percentile of training projections, the curve is thin and isotonic "clip" would cap
the very best projections at one value. So there we keep the projection and just shift it by the
curve's correction at that point (the same idea as the top-bucket fix in ranges.py).

RESULT 2026-11-17 (src/backtest/calibration_test.py, rule set before results): MAE 2022-24 QB -0.055,
RB -0.027, WR -0.028 (CIs below 0), TE -0.010 (n.s.), rankings unchanged; top-5% bias RB -0.84 ->
-0.14, WR -0.66 -> -0.29. 2026 check: WR -0.031 (significant), QB/RB lean better, TE flat. PASS ->
live projections are calibrated (src/predict/week.py). Caveat: it overcorrects some QB/TE tiers.
"""

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

POSITIONS = ["QB", "RB", "WR", "TE"]
PRED_PATH = "data/processed/weekly_predictions_{}.parquet"   # walk-forward predictions per season


def load_history(seasons: list[int]) -> pl.DataFrame:
    """Earlier seasons' walk-forward predictions (written by src/backtest/calibration_test.py / vs_espn.py)."""
    missing = [s for s in seasons if not Path(PRED_PATH.format(s)).exists()]
    if missing:
        raise SystemExit(f"Missing walk-forward predictions for {missing}: run python -m src.backtest.calibration_test")
    return pl.concat([pl.read_parquet(PRED_PATH.format(s)).select("position", "pred", "actual") for s in seasons])


class WeeklyCalibration:
    def fit(self, history: pl.DataFrame) -> "WeeklyCalibration":
        """history: position, pred, actual (walk-forward predictions from earlier seasons)."""
        self.curves, self.top = {}, {}
        for pos in POSITIONS:
            h = history.filter(pl.col("position") == pos)
            if h.height < 100:  # too little history to shape a curve: leave this position as is
                continue
            x, y = h["pred"].to_numpy(), h["actual"].to_numpy()
            self.curves[pos] = IsotonicRegression(out_of_bounds="clip").fit(x, y)
            hi = np.quantile(x, 0.99)
            self.top[pos] = (hi, float(self.curves[pos].predict([hi])[0] - hi))
        return self

    def apply(self, preds: np.ndarray, positions: np.ndarray) -> np.ndarray:
        out = preds.astype(float).copy()
        for pos in POSITIONS:
            m = positions == pos
            if not m.any() or pos not in self.curves:
                continue
            hi, shift = self.top[pos]
            p = preds[m].astype(float)
            out[m] = np.where(p <= hi, self.curves[pos].predict(p), p + shift)
        return np.clip(out, 0, None)
