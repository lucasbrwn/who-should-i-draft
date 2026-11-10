"""Ridge regression per position and stat (SCRUM-41).

Pipeline (scikit-learn), one per position x modeled stat:
    numbers/yes-no  -> fill blanks with the median + a "was missing" flag -> scale
    text (injury report status, practice code) -> one yes/no column per value
    -> Ridge, predictions clipped at 0

Modeled stats follow the design doc (section 1); rare stats (2-pt, fumbles, return TDs, WR/TE
rushing...) come from the baseline, since there are too few events to learn them.

alpha (how hard Ridge shrinks the weights) is chosen from ALPHAS using only earlier seasons:
validate on the latest complete season in the history, after training on the seasons before it.
With no earlier season to train on (testing 2021: data starts in 2020) it uses DEFAULT_ALPHA (100).
"""

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.assemble import feature_columns
from src.models.baseline import COMPONENTS, Baseline

MODELED = {
    "QB": ["passing_yards", "passing_tds", "passing_interceptions", "rushing_yards", "rushing_tds"],
    "RB": ["rushing_yards", "rushing_tds", "receptions", "receiving_yards", "receiving_tds"],
    "WR": ["receptions", "receiving_yards", "receiving_tds"],
    "TE": ["receptions", "receiving_yards", "receiving_tds"],
}
COUNT_STATS = {"passing_tds", "passing_interceptions", "rushing_tds", "receptions", "receiving_tds"}
# Design doc said 0.1-100, but every model picked 100 (the edge): 134 overlapping features
# (last 3 / last 5 / season / last season of the same stat) need strong shrinkage. Widened 2026-11-10.
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
DEFAULT_ALPHA = 100.0
LAST_WEEK_MIN = 17  # a season counts as complete once the history reaches week 17


def make_model(numeric: list[str], text: list[str], alpha: float) -> Pipeline:
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                              StandardScaler()), numeric),
        ("text", make_pipeline(SimpleImputer(strategy="constant", fill_value="none"),
                               OneHotEncoder(handle_unknown="ignore")), text),
    ])
    return Pipeline([("prep", prep), ("ridge", Ridge(alpha=alpha))])


def score(stat: str, actual: np.ndarray, pred: np.ndarray) -> float:
    """Lower is better. Poisson deviance for counts, MAE for yards (design doc section 6)."""
    if stat in COUNT_STATS:
        return mean_poisson_deviance(actual, np.clip(pred, 1e-6, None))
    return mean_absolute_error(actual, pred)


class RidgeModel:
    name = "ridge"

    def __init__(self):
        self.baseline = Baseline()
        self.alpha_cache: dict[tuple[int, str, str], float] = {}

    def _columns(self, df: pl.DataFrame) -> tuple[list[str], list[str]]:
        feats = feature_columns(df)
        text = [c for c in feats if df.schema[c] == pl.String]
        return [c for c in feats if c not in text], text

    def _xy(self, df: pl.DataFrame, stat: str):
        numeric, text = self._columns(df)
        x = df.select([pl.col(c).cast(pl.Float64) for c in numeric] + text).to_pandas()
        return x, df[f"y_{stat}"].to_numpy(), numeric, text

    def _validation_season(self, history: pl.DataFrame) -> int | None:
        complete = history.group_by("season").agg(pl.max("week")).filter(pl.col("week") >= LAST_WEEK_MIN)["season"]
        return int(complete.max()) if complete.len() else None

    def _alpha(self, train: pl.DataFrame, pos: str, stat: str) -> float:
        val_season = self._validation_season(train)
        if val_season is None or train.filter(pl.col("season") < val_season).height == 0:
            return DEFAULT_ALPHA
        key = (val_season, pos, stat)
        if key not in self.alpha_cache:
            fit_rows, val_rows = train.filter(pl.col("season") < val_season), train.filter(pl.col("season") == val_season)
            x_fit, y_fit, numeric, text = self._xy(fit_rows, stat)
            x_val, y_val, _, _ = self._xy(val_rows, stat)
            scores = {a: score(stat, y_val, np.clip(make_model(numeric, text, a).fit(x_fit, y_fit).predict(x_val), 0, None))
                      for a in ALPHAS}
            self.alpha_cache[key] = min(scores, key=scores.get)
        return self.alpha_cache[key]

    def fit(self, history: pl.DataFrame) -> "RidgeModel":
        self.baseline.fit(history)
        train = history.filter(~pl.col("left_injured"))
        self.models, self.alphas = {}, {}
        for pos, stats in MODELED.items():
            rows = train.filter(pl.col("position") == pos)
            if rows.height < 50:
                continue
            for stat in stats:
                alpha = self._alpha(rows, pos, stat)
                x, y, numeric, text = self._xy(rows, stat)
                self.models[(pos, stat)] = make_model(numeric, text, alpha).fit(x, y)
                self.alphas[(pos, stat)] = alpha
        return self

    def predict(self, rows: pl.DataFrame) -> pl.DataFrame:
        preds = self.baseline.predict(rows)  # rare stats, and anything Ridge can't predict yet
        for pos, stats in MODELED.items():
            mask = (rows["position"] == pos).to_numpy()
            if not mask.any():
                continue
            pos_rows = rows.filter(pl.col("position") == pos)
            for stat in stats:
                model = self.models.get((pos, stat))
                if model is None:
                    continue
                x, _, _, _ = self._xy(pos_rows, stat)
                values = preds[stat].to_numpy().copy()
                values[mask] = np.clip(model.predict(x), 0, None)
                preds = preds.with_columns(pl.Series(stat, values))
        return preds

    def coefficients(self) -> pl.DataFrame:
        """Scaled coefficients of the latest fit: one row per position x stat x feature."""
        rows = []
        for (pos, stat), model in self.models.items():
            names = model.named_steps["prep"].get_feature_names_out()
            for name, coef in zip(names, model.named_steps["ridge"].coef_):
                rows.append({"position": pos, "stat": stat, "feature": name.split("__", 1)[1], "coef": float(coef),
                             "alpha": self.alphas[(pos, stat)]})
        return pl.DataFrame(rows).sort("position", "stat", pl.col("coef").abs(), descending=[False, False, True])


assert all(s in COMPONENTS for stats in MODELED.values() for s in stats)
