"""Gradient boosting per position and stat (SCRUM-49).

scikit-learn HistGradientBoostingRegressor, one per position x modeled stat (same list as Ridge):
    Poisson loss for counts (TDs, receptions, INTs: never predicts a negative), squared error for yards.
    Blanks are handled natively; injury report text becomes a category column.

Settings are reasonable defaults; proper tuning with time-based folds is SCRUM-50.

Retrains every REFIT_WEEKS weeks (and at the start of each season) instead of weekly, because
boosting is ~20x slower than Ridge. Features still update every week; only the learned patterns
refresh monthly. Rare stats come from the baseline, which does update weekly.
"""

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from src.features.assemble import feature_columns
from src.models.baseline import Baseline
from src.models.ridge import COUNT_STATS, MODELED

REFIT_WEEKS = 4
PARAMS = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=50,
              l2_regularization=0.1, early_stopping=False, random_state=0)


class BoostingModel:
    name = "boosting"

    def __init__(self, refit_weeks: int = REFIT_WEEKS, params: dict | None = None):
        self.refit_weeks = refit_weeks
        self.params = {**PARAMS, **(params or {})}
        self.baseline = Baseline()
        self.models: dict[tuple[str, str], HistGradientBoostingRegressor] = {}
        self.fitted_through: tuple[int, int] | None = None
        self.categories: dict[str, list[str]] = {}
        self.columns: dict[str, list[str]] = {}

    def _x(self, df: pl.DataFrame, cols: list[str]) -> pd.DataFrame:
        text = [c for c in cols if df.schema[c] == pl.String]
        x = df.select([pl.col(c).cast(pl.Float64) for c in cols if c not in text] + text).to_pandas()
        for c in text:  # same category list at fit and predict time; unseen values become blanks
            cats = self.categories.setdefault(c, sorted(df[c].drop_nulls().unique()))
            x[c] = pd.Categorical(x[c].where(x[c].isin(cats)), categories=cats)
        return x

    def _needs_refit(self, latest: tuple[int, int]) -> bool:
        if self.fitted_through is None:
            return True
        season, week = self.fitted_through
        return latest[0] != season or latest[1] - week >= self.refit_weeks

    def fit(self, history: pl.DataFrame) -> "BoostingModel":
        self.baseline.fit(history)
        latest = history.select(pl.col("season").max()).item()
        latest = (latest, history.filter(pl.col("season") == latest)["week"].max())
        if not self._needs_refit(latest):
            return self
        train = history.filter(~pl.col("left_injured"))
        self.categories = {}
        self.models = {}
        for pos, stats in MODELED.items():
            rows = train.filter(pl.col("position") == pos)
            if rows.height < 50:
                continue
            # Columns that are blank for every row of this position (e.g. rb_checkdown for QBs) are dropped.
            self.columns[pos] = [c for c in feature_columns(rows) if rows[c].null_count() < rows.height]
            x = self._x(rows, self.columns[pos])
            for stat in stats:
                y = rows[f"y_{stat}"].to_numpy()
                if y.sum() <= 0:  # Poisson can't fit an all-zero stat; the baseline (0) covers it
                    continue
                loss = "poisson" if stat in COUNT_STATS else "squared_error"
                model = HistGradientBoostingRegressor(loss=loss, categorical_features="from_dtype", **self.params)
                self.models[(pos, stat)] = model.fit(x, y)
        self.fitted_through = latest
        return self

    def predict(self, rows: pl.DataFrame) -> pl.DataFrame:
        preds = self.baseline.predict(rows)
        for pos, stats in MODELED.items():
            mask = (rows["position"] == pos).to_numpy()
            if not mask.any():
                continue
            if pos not in self.columns:
                continue
            x = self._x(rows.filter(pl.col("position") == pos), self.columns[pos])
            for stat in stats:
                model = self.models.get((pos, stat))
                if model is None:
                    continue
                values = preds[stat].to_numpy().copy()
                values[mask] = np.clip(model.predict(x), 0, None)
                preds = preds.with_columns(pl.Series(stat, values))
        return preds
