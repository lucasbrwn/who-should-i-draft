"""Season floor, ceiling and breakout chance (SCRUM-52 season half + SCRUM-94 #2).

Lucas, 2026-11-14: late picks should be valued by upside, not mean projection.

Three models on the same season-level rows and inputs as the season model (src/models/season.py),
all walk-forward (each season predicted from earlier seasons only, 2025 untouched):
    floor     20th percentile of season points   (gradient boosting, quantile loss)
    ceiling   80th percentile of season points
    breakout  chance of a starter-level season: top-12 QB/TE or top-24 RB/WR in actual points
              (gradient boosting classifier)

Change from the design doc: it planned floor/ceiling by simulating the WEEKLY model's errors. For
season totals, predicting the percentiles directly is simpler and is checked the same way: about
80% of seasons should land under the ceiling and 80% over the floor. A weekly simulation can still
come later for best-ball weekly ceilings.

Usage:
    python -m src.models.upside     # walk-forward 2023-2024 calibration report (+2022 trained on 2021)
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from src.backtest.harness import bootstrap
from src.models.season import B_PARAMS, EVAL_SEASONS, EXAM_SEASON, LIVE_SEASON, OUT_DIR, POSITIONS, b_inputs, season_table

REPORT_DIR = Path("docs/reports")
STARTER_RANK = {"QB": 12, "RB": 24, "WR": 24, "TE": 12}
FLOOR_Q, CEILING_Q = 0.2, 0.8


def add_breakout_label(df: pl.DataFrame) -> pl.DataFrame:
    rank = pl.col("actual").rank("ordinal", descending=True).over("season", "position")
    return df.with_columns(breakout=(rank <= pl.col("position").replace_strict(STARTER_RANK, default=0)).cast(pl.Int8))


LABELS = {"breakout"}


def usable(df: pl.DataFrame) -> list[str]:
    return [c for c in b_inputs(df) if c not in LABELS and df[c].null_count() < df.height]
  # derived from `actual`: never an input (an early version leaked it and scored a "perfect" 100%)


class UpsideModel:
    """Quantile models for floor/ceiling, widened by conformal calibration (CQR): on each training
    season held out from the others, measure how far actual seasons fall outside the predicted range,
    and widen both ends by the amount that makes the target coverage hold. Uses training seasons only."""

    def fit(self, train: pl.DataFrame) -> "UpsideModel":
        self.cols = usable(train)
        self.q_lo, self.q_hi = self._quantiles(train, self.cols)
        self.breakout = HistGradientBoostingClassifier(categorical_features="from_dtype", **B_PARAMS).fit(
            self._x(train), train["breakout"].to_numpy())
        self.widen = self._conformal(train)
        return self

    def _quantiles(self, train: pl.DataFrame, cols: list[str]):
        x, y = self._x(train, cols), train["actual"].to_numpy()
        lo = HistGradientBoostingRegressor(loss="quantile", quantile=FLOOR_Q, categorical_features="from_dtype", **B_PARAMS).fit(x, y)
        hi = HistGradientBoostingRegressor(loss="quantile", quantile=CEILING_Q, categorical_features="from_dtype", **B_PARAMS).fit(x, y)
        return lo, hi

    def _conformal(self, train: pl.DataFrame) -> dict[str, tuple[float, float]]:
        """Per position: (lower, upper) widening in points, from leave-one-training-season-out errors.
        Per position because RB/TE seasons swing more than QB seasons (one shared width left RB/TE
        ceilings covering only 53-69%)."""
        seasons = sorted(train["season"].unique().to_list())
        if len(seasons) < 2:
            return {}  # one training season: nothing to hold out (2022 boards), ranges stay as fit
        errs = []
        for s in seasons:
            fit_on, held = train.filter(pl.col("season") != s), train.filter(pl.col("season") == s)
            cols = usable(fit_on)  # e.g. "two seasons ago" is all blank when only 2021 is left
            lo, hi = self._quantiles(fit_on, cols)
            x, y = self._x(held, cols), held["actual"].to_numpy()
            errs.append(pl.DataFrame({"position": held["position"], "lo": lo.predict(x) - y, "hi": y - hi.predict(x)}))
        e = pl.concat(errs)
        return {pos: (float(g["lo"].quantile(1 - FLOOR_Q)), float(g["hi"].quantile(CEILING_Q)))
                for (pos,), g in e.group_by("position")}

    def _x(self, df: pl.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        cols = self.cols if cols is None else cols
        x = df.select([pl.col(c).cast(pl.Float64) if c != "position" else pl.col(c) for c in cols]).to_pandas()
        x["position"] = pd.Categorical(x["position"], categories=POSITIONS)
        return x

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        x = self._x(df)
        pos = df["position"].to_list()
        w_lo = np.array([max(0.0, self.widen.get(p, (0.0, 0.0))[0]) for p in pos])
        w_hi = np.array([max(0.0, self.widen.get(p, (0.0, 0.0))[1]) for p in pos])
        lo = np.clip(self.q_lo.predict(x) - w_lo, 0, None)
        hi = np.clip(self.q_hi.predict(x) + w_hi, 0, None)
        return df.select("player_id", "season").with_columns(
            floor=pl.Series(np.minimum(lo, hi)), ceiling=pl.Series(np.maximum(lo, hi)),
            breakout_p=pl.Series(self.breakout.predict_proba(x)[:, 1]))


def draft_pool(df: pl.DataFrame) -> pl.DataFrame:
    """Players who matter for a draft: those who scored or were projected like a rosterable player."""
    return df.filter((pl.col("a_ppg") >= 5) | (pl.col("actual") >= 50))


def evaluate(results: pl.DataFrame) -> str:
    pool = draft_pool(results)
    lines = [f"# Season floor / ceiling / breakout ({date.today()})", "",
             f"Walk-forward over {EVAL_SEASONS} (2025 untouched). Draft-relevant players (projected 5+ PPG or scored 50+).", "",
             "## Range calibration (target: 80% / 80% / 60%)", "",
             "| position | players | under ceiling | over floor | inside range |", "| --- | --- | --- | --- | --- |"]
    for pos in [*POSITIONS, "ALL"]:
        s = pool if pos == "ALL" else pool.filter(pl.col("position") == pos)
        under = (s["actual"] <= s["ceiling"]).mean()
        over = (s["actual"] >= s["floor"]).mean()
        inside = ((s["actual"] >= s["floor"]) & (s["actual"] <= s["ceiling"])).mean()
        lines.append(f"| {pos} | {s.height} | {under:.0%} | {over:.0%} | {inside:.0%} |")
    lines += ["", "## Breakout chance (starter-level season)", "",
              "Calibration: players grouped by predicted chance; actual breakout rate should match.", "",
              "| predicted chance | players | actual breakout rate |", "| --- | --- | --- |"]
    bins = [0, 0.05, 0.15, 0.3, 0.5, 0.7, 1.01]
    for lo, hi in zip(bins, bins[1:]):
        s = pool.filter(pl.col("breakout_p") >= lo, pl.col("breakout_p") < hi)
        if s.height:
            lines.append(f"| {lo:.0%}-{min(hi, 1):.0%} | {s.height} | {s['breakout'].mean():.0%} |")
    p, y = pool["breakout_p"].to_numpy(), pool["breakout"].to_numpy().astype(float)
    brier, base = np.mean((p - y) ** 2), np.mean((y.mean() - y) ** 2)
    lines += ["", f"Brier score {brier:.3f} vs {base:.3f} for always guessing the base rate "
                  f"(skill {1 - brier / base:.0%}; higher is better).",
              f"Brier difference per player vs base rate [95% CI]: {bootstrap((p - y) ** 2 - (y.mean() - y) ** 2)}"]
    return "\n".join(lines) + "\n"


def upside_board(season: int, data: pl.DataFrame | None = None) -> pl.DataFrame:
    """floor / ceiling / breakout_p for a season, trained on earlier seasons only."""
    data = add_breakout_label(season_table() if data is None else data)
    train = data.filter(pl.col("season") < season)  # 2025 included since the exam (v2)
    return UpsideModel().fit(train).predict(data.filter(pl.col("season") == season))


def main() -> None:
    data = add_breakout_label(season_table())
    results = []
    for season in EVAL_SEASONS:
        train, test = data.filter(pl.col("season") < season), data.filter(pl.col("season") == season)
        results.append(test.join(UpsideModel().fit(train).predict(test), on=["player_id", "season"]))
    text = evaluate(pl.concat(results))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"upside_{date.today()}.md").write_text(text, encoding="utf-8")
    print(text)
    for season in [*EVAL_SEASONS, LIVE_SEASON]:
        upside_board(season, data).write_parquet(OUT_DIR / f"upside_{season}.parquet")


if __name__ == "__main__":
    main()
