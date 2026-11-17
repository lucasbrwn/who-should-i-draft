"""Weekly bust / ceiling ranges (SCRUM-52).

bust    = 10th percentile outcome: about 1 week in 10 he scores less
ceiling = 90th percentile outcome: about 1 week in 10 he scores more

Two methods, compared on a pre-registered rule (python -m src.models.ranges):

    BucketRanges     v1: split each position's projections into 5 buckets; bust/ceiling = the 10th/90th
                     percentile of what actually happened in that bucket. Every player in a bucket gets
                     the same range. TOP bucket: projection + the bucket's 10th/90th percentile miss
                     instead, because a fixed number capped the ceiling (Gibbs 26.0 proj -> 26.3 ceiling;
                     the best projections beat it ~20% of the time, not 10%).
    QuantileRanges   v2: linear quantile regression per position,
                         quantile_q(actual) = a + b * projection + c * volatility
                     volatility = standard deviation of his PPR points over his previous 16 games (left-
                     injured games excluded, crosses seasons; known before kickoff). A smooth line, so a
                     bigger projection always gets a bigger range, with no cap at the top.

RESULT 2026-11-17: v2 FAILED the rule (pinball -0.008 [-0.014,-0.002] better, but coverage QB 75%,
TE 91%), so v1 stays, with the top-bucket fix (the agreed fallback). Caveat for any re-test: 43% of
2024 TE games score <= 0.5 (backup blockers), so all-player coverage is inflated by easy zeros;
a fairer coverage check would use fantasy-relevant players only. That would be a NEW rule, set first.

Rule (fixed before results, 2026-11-17): fit on the 2022-23 walk-forward backtest, judge on 2024.
v2 replaces v1 only if (1) 77-83% of 2024 games land inside [bust, ceiling] at EVERY position and
(2) average pinball loss is lower than v1 with a 95% week-bootstrap CI below zero.

Pinball loss scores a quantile guess: miss low by x -> q*x, miss high by x -> (1-q)*x. It's the standard
score for ranges because it punishes both ranges that miss and ranges that are uselessly wide.
"""

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import QuantileRegressor

POSITIONS = ["QB", "RB", "WR", "TE"]
QUANTILES = {"bust": 0.10, "ceiling": 0.90}
# What the app shows (Lucas, 2026-11-17): the middle half of outcomes, ~11 pts wide for a starter,
# instead of a 10-90 range ~20 pts wide. Inside about 1 week in 2 (2024, proj >= 10: 49-54%).
LIKELY = {"bust": 0.25, "ceiling": 0.75}
VOL_GAMES, VOL_MIN_GAMES = 16, 4
BACKTEST_PATH = Path("data/processed/weekly_predictions_boosting.parquet")
PW_PATH = Path("data/processed/player_weeks.parquet")
N_BUCKETS = 5


def add_volatility(pw: pl.DataFrame) -> pl.DataFrame:
    """sd of PPR points over each player's previous VOL_GAMES games (this game excluded, so no leakage).
    Rows with blank points (upcoming games) still get a value from the games before them."""
    played = pw.filter(~pl.col("left_injured")).sort("player_id", "season", "week")
    return played.with_columns(
        volatility=pl.col("fantasy_points_ppr").rolling_std(VOL_GAMES, min_samples=VOL_MIN_GAMES)
        .shift(1).over("player_id")
    ).select("player_id", "season", "week", "volatility")


def pinball(actual: np.ndarray, guess: np.ndarray, q: float) -> np.ndarray:
    diff = actual - guess
    return np.maximum(q * diff, (q - 1) * diff)


def finish(rows: pl.DataFrame, bust: np.ndarray, ceiling: np.ndarray) -> pl.DataFrame:
    """Bust is never negative or above the projection; ceiling never below it."""
    proj = rows["proj"].to_numpy()
    return rows.with_columns(bust=pl.Series(np.clip(np.minimum(bust, proj), 0, None)),
                             ceiling=pl.Series(np.maximum(ceiling, proj)))


class BucketRanges:
    name = "buckets_v1"

    def __init__(self, quantiles: dict[str, float] = QUANTILES):
        """quantiles: {"bust": low, "ceiling": high}; LIKELY gives the 25-75 likely range."""
        self.quantiles = quantiles

    def fit(self, train: pl.DataFrame) -> "BucketRanges":
        self.table = {}
        for pos in POSITIONS:
            rows = train.filter(pl.col("position") == pos)
            proj, actual = rows["proj"].to_numpy(), rows["actual"].to_numpy()
            edges = np.quantile(proj, np.linspace(0, 1, N_BUCKETS + 1)[1:-1])
            bucket = np.searchsorted(edges, proj, side="right")
            top = bucket == N_BUCKETS - 1
            self.table[pos] = (edges, {k: np.array([np.quantile(actual[bucket == b], q) for b in range(N_BUCKETS)])
                                       for k, q in self.quantiles.items()},
                               {k: float(np.quantile(actual[top] - proj[top], q)) for k, q in self.quantiles.items()})
        return self

    def predict(self, rows: pl.DataFrame) -> pl.DataFrame:
        out = []
        for pos in POSITIONS:
            part = rows.filter(pl.col("position") == pos)
            edges, values, top_miss = self.table[pos]
            proj = part["proj"].to_numpy()
            bucket = np.searchsorted(edges, proj, side="right")
            top = bucket == N_BUCKETS - 1
            bust = np.where(top, proj + top_miss["bust"], values["bust"][bucket])
            ceiling = np.where(top, proj + top_miss["ceiling"], values["ceiling"][bucket])
            out.append(finish(part, bust, ceiling))
        return pl.concat(out)


class QuantileRanges:
    name = "quantile_v2"

    def fit(self, train: pl.DataFrame) -> "QuantileRanges":
        self.models, self.fill = {}, {}
        for pos in POSITIONS:
            rows = train.filter(pl.col("position") == pos)
            self.fill[pos] = rows["volatility"].median()  # players with < 4 past games get a typical value
            x = self._x(rows, pos)
            for k, q in QUANTILES.items():
                model = QuantileRegressor(quantile=q, alpha=0.0, solver="highs")
                self.models[(pos, k)] = model.fit(x, rows["actual"].to_numpy())
        return self

    def _x(self, rows: pl.DataFrame, pos: str) -> np.ndarray:
        return rows.select("proj", pl.col("volatility").fill_null(self.fill[pos])).to_numpy()

    def predict(self, rows: pl.DataFrame) -> pl.DataFrame:
        out = []
        for pos in POSITIONS:
            part = rows.filter(pl.col("position") == pos)
            if part.height == 0:
                continue
            x = self._x(part, pos)
            out.append(finish(part, self.models[(pos, "bust")].predict(x), self.models[(pos, "ceiling")].predict(x)))
        return pl.concat(out)


def load_backtest() -> pl.DataFrame:
    bt = pl.read_parquet(BACKTEST_PATH).rename({"pred": "proj"})
    return bt.join(add_volatility(pl.read_parquet(PW_PATH)), on=["player_id", "season", "week"], how="left")


def score(scored: pl.DataFrame) -> pl.DataFrame:
    a = scored["actual"].to_numpy()
    return scored.with_columns(
        inside=(pl.col("actual") >= pl.col("bust")) & (pl.col("actual") <= pl.col("ceiling")),
        loss=pl.Series((pinball(a, scored["bust"].to_numpy(), 0.1) + pinball(a, scored["ceiling"].to_numpy(), 0.9)) / 2),
        width=pl.col("ceiling") - pl.col("bust"),
    )


def week_bootstrap_diff(a: pl.DataFrame, b: pl.DataFrame, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean loss difference (b - a) with a 95% CI from resampling whole weeks (same weeks for both)."""
    key = ["season", "week"]
    wk = a.group_by(key).agg(la=pl.col("loss").sum(), n=pl.len()).join(
        b.group_by(key).agg(lb=pl.col("loss").sum()), on=key).sort(key)
    la, lb, cnt = wk["la"].to_numpy(), wk["lb"].to_numpy(), wk["n"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(wk), (n, len(wk)))
    diffs = (lb[idx].sum(1) - la[idx].sum(1)) / cnt[idx].sum(1)
    return float((lb.sum() - la.sum()) / cnt.sum()), *np.percentile(diffs, [2.5, 97.5])


def main() -> None:
    bt = load_backtest()
    train, test = bt.filter(pl.col("season").is_in([2022, 2023])), bt.filter(pl.col("season") == 2024)
    results = {m.name: score(m.fit(train).predict(test)) for m in (BucketRanges(), QuantileRanges())}
    print("2024 (fit on 2022-23): share inside [bust, ceiling], pinball loss, average width")
    for name, r in results.items():
        print(f"\n{name}")
        print(r.group_by("position").agg(inside=pl.col("inside").mean().round(3), loss=pl.col("loss").mean().round(3),
                                         width=pl.col("width").mean().round(1), n=pl.len()).sort("position"))
    v1, v2 = results["buckets_v1"], results["quantile_v2"]
    diff, lo, hi = week_bootstrap_diff(v1, v2)
    cov = v2.group_by("position").agg(pl.col("inside").mean())["inside"]
    print(f"\nv2 - v1 pinball loss: {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    for pos in POSITIONS:
        d, l, h = week_bootstrap_diff(v1.filter(pl.col("position") == pos), v2.filter(pl.col("position") == pos))
        print(f"   {pos}: {d:+.3f} [{l:+.3f}, {h:+.3f}]")
    passed = bool(((cov >= 0.77) & (cov <= 0.83)).all()) and hi < 0
    print("RULE:", "PASS -> use quantile_v2" if passed else "FAIL -> keep buckets_v1")


if __name__ == "__main__":
    main()
