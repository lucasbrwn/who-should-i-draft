"""Tune gradient boosting settings with time-based folds (SCRUM-50).

For each position x modeled stat, try N_SETTINGS random settings (plus the defaults) on two folds:
    train through 2021 -> validate on 2022
    train through 2022 -> validate on 2023
Never a random split, never 2024 (kept for the tuned-vs-default comparison) or 2025 (final exam).
Score: Poisson deviance for counts, MAE for yards (design doc section 6), averaged over both folds.
The best settings are saved to models/boosting_params.json, which BoostingModel(tuned=True) loads.

Two objectives:
    stat_error (first try): best settings per position x stat by that stat's own error. On 2024 this
        tied the defaults (better on 2022-23, didn't transfer), saved as boosting_params_stat_error.json.
    rank (default): one setting per position, chosen by how well the predicted fantasy points RANK
        players each week (mean weekly Spearman), which is what drafting cares about.

Usage:
    python -m src.models.tune [--objective rank|stat_error]   # ~12 minutes
    python -m src.backtest --models boosting boosting_tuned --seasons 2024
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from src.models.boosting import PARAMS, TUNED_KEYS, TUNED_PATH, BoostingModel, usable_columns
from scipy.stats import spearmanr

from src.backtest.harness import STARTERS, points
from src.models.ridge import COUNT_STATS, MODELED, score

FEATURES_PATH = Path("data/processed/features.parquet")
FOLDS = [(2021, 2022), (2022, 2023)]  # (train through, validate on)
N_SETTINGS = 25
SEED = 0


def random_settings(n: int, seed: int = SEED) -> list[dict]:
    """Design doc ranges: learning rate 0.02-0.2, 15-63 leaves, 20-200 games per leaf, L2 0-1."""
    rng = np.random.default_rng(seed)
    return [{
        "learning_rate": float(np.exp(rng.uniform(np.log(0.02), np.log(0.2)))),
        "max_leaf_nodes": int(rng.integers(15, 64)),
        "min_samples_leaf": int(rng.integers(20, 201)),
        "l2_regularization": float(rng.uniform(0, 1)),
        "max_iter": int(rng.choice([100, 200, 300, 450, 600])),
    } for _ in range(n)]


def tune(feats: pl.DataFrame, n_settings: int = N_SETTINGS) -> dict[str, dict]:
    candidates = [{k: PARAMS[k] for k in TUNED_KEYS}] + random_settings(n_settings)
    train = feats.filter(~pl.col("left_injured"))
    best = {}
    for pos, stats in MODELED.items():
        rows = train.filter(pl.col("position") == pos)
        folds = []
        for through, val in FOLDS:
            helper = BoostingModel()  # reuses its feature prep (category handling) so tuning matches the model
            fit_rows, val_rows = rows.filter(pl.col("season") <= through), rows.filter(pl.col("season") == val)
            cols = usable_columns(fit_rows)
            folds.append((helper._x(fit_rows, cols), fit_rows, helper._x(val_rows, cols), val_rows))
        for stat in stats:
            loss = "poisson" if stat in COUNT_STATS else "squared_error"
            scores = []
            for params in candidates:
                fold_scores = []
                for x_fit, fit_rows, x_val, val_rows in folds:
                    model = HistGradientBoostingRegressor(loss=loss, categorical_features="from_dtype",
                                                          early_stopping=False, random_state=0, **params)
                    model.fit(x_fit, fit_rows[f"y_{stat}"].to_numpy())
                    fold_scores.append(score(stat, val_rows[f"y_{stat}"].to_numpy(), np.clip(model.predict(x_val), 0, None)))
                scores.append(float(np.mean(fold_scores)))
            i = int(np.argmin(scores))
            best[f"{pos}|{stat}"] = {**candidates[i], "score": scores[i], "default_score": scores[0]}
            print(f"{pos} {stat}: best {scores[i]:.4f} vs default {scores[0]:.4f}  {candidates[i]}")
    return best


def weekly_spearman(rows: pl.DataFrame, pred_points: np.ndarray, pos: str) -> float:
    """Mean over weeks of the rank correlation between predicted and actual fantasy points."""
    actual = points(rows.select("position", *[pl.col(c).alias(c[2:]) for c in rows.columns if c.startswith("y_")]))
    df = rows.select("season", "week").with_columns(pred=pl.Series(pred_points), actual=actual)
    values = [spearmanr(g["pred"], g["actual"]).statistic for _, g in df.group_by("season", "week")
              if g.height >= 2 * STARTERS[pos]]
    # A candidate that predicts the same points for everyone has no ranking (NaN): score it as the worst.
    values = [v for v in values if not np.isnan(v)]
    return float(np.mean(values)) if values else -1.0


def tune_rank(feats: pl.DataFrame, n_settings: int = N_SETTINGS) -> dict[str, dict]:
    """One setting per position, chosen by weekly rank correlation of predicted fantasy points.
    Rare stats are left at 0 for every candidate (same for all, so it doesn't change the choice)."""
    candidates = [{k: PARAMS[k] for k in TUNED_KEYS}] + random_settings(n_settings)
    train = feats.filter(~pl.col("left_injured"))
    best = {}
    for pos, stats in MODELED.items():
        rows = train.filter(pl.col("position") == pos)
        folds = []
        for through, val in FOLDS:
            helper = BoostingModel()
            fit_rows, val_rows = rows.filter(pl.col("season") <= through), rows.filter(pl.col("season") == val)
            cols = usable_columns(fit_rows)
            folds.append((helper._x(fit_rows, cols), fit_rows, helper._x(val_rows, cols), val_rows))
        scores = []
        for params in candidates:
            fold_scores = []
            for x_fit, fit_rows, x_val, val_rows in folds:
                pred = val_rows.select("position")
                for stat in stats:
                    loss = "poisson" if stat in COUNT_STATS else "squared_error"
                    model = HistGradientBoostingRegressor(loss=loss, categorical_features="from_dtype",
                                                          early_stopping=False, random_state=0, **params)
                    model.fit(x_fit, fit_rows[f"y_{stat}"].to_numpy())
                    pred = pred.with_columns(pl.Series(stat, np.clip(model.predict(x_val), 0, None)))
                fold_scores.append(weekly_spearman(val_rows, points(pred).to_numpy(), pos))
            scores.append(float(np.mean(fold_scores)))
        i = int(np.argmax(scores))  # higher rank correlation = better
        for stat in stats:
            best[f"{pos}|{stat}"] = {**candidates[i], "spearman": scores[i], "default_spearman": scores[0]}
        print(f"{pos}: best spearman {scores[i]:.4f} vs default {scores[0]:.4f}  {candidates[i]}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", choices=["rank", "stat_error"], default="rank")
    args = parser.parse_args()
    start = time.time()
    feats = pl.read_parquet(FEATURES_PATH)
    best = tune_rank(feats) if args.objective == "rank" else tune(feats)
    TUNED_PATH.parent.mkdir(parents=True, exist_ok=True)
    TUNED_PATH.write_text(json.dumps(best, indent=2))
    print(f"Wrote {TUNED_PATH} in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
