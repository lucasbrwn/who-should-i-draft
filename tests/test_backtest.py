"""Baseline model (SCRUM-40) and walk-forward harness (SCRUM-42). Rules agreed 2026-11-10."""

import numpy as np
import polars as pl
import pytest

from src.backtest.harness import bootstrap, run, weekly_metrics
from src.features.assemble import TARGETS
from src.models.baseline import Baseline


def game(player, season, week, position="WR", left_injured=False, **y):
    row = {"player_id": player, "player_display_name": player, "position": position, "season": season, "week": week,
           "left_injured": left_injured}
    row.update({f"y_{t}": 0.0 for t in TARGETS})
    row.update({f"y_{k}": float(v) for k, v in y.items()})
    return row


def df(rows):
    return pl.DataFrame(rows)


def predict_one(history_rows, player="a", position="WR"):
    model = Baseline().fit(df(history_rows))
    return model.predict(df([game(player, 2023, 10, position)])).row(0, named=True)


def test_recent_games_count_more():
    # 3 games ago = 0 catches, last game = 10. Weights 0.5 and 1 -> (0*0.5 + 10*1) / 1.5
    pred = predict_one([game("a", 2023, 7, receptions=0), game("a", 2023, 8, receptions=5),
                        game("a", 2023, 9, receptions=10)])
    w = np.array([0.5 ** (2 / 3), 0.5 ** (1 / 3), 1.0])
    assert pred["receptions"] == pytest.approx((w @ [0, 5, 10]) / w.sum())


def test_only_last_8_games_count():
    rows = [game("a", 2023, 1, receptions=100)] + [game("a", 2023, w, receptions=2) for w in range(2, 10)]
    assert predict_one(rows)["receptions"] == pytest.approx(2.0)


def test_last_season_is_used_in_week_1():
    model = Baseline().fit(df([game("a", 2022, 17, receptions=6)]))
    assert model.predict(df([game("a", 2023, 1)])).row(0, named=True)["receptions"] == 6.0


def test_left_injured_games_are_skipped():
    pred = predict_one([game("a", 2023, 8, receptions=6), game("a", 2023, 9, receptions=0, left_injured=True)])
    assert pred["receptions"] == 6.0


def test_new_player_gets_position_average():
    pred = predict_one([game("b", 2023, 9, "TE", receptions=4), game("c", 2023, 9, "TE", receptions=2)], player="new", position="TE")
    assert pred["receptions"] == 3.0


def test_harness_never_fits_on_the_predicted_week_or_later():
    rows = [game(p, 2021, w, receptions=w) for w in range(1, 6) for p in "abc"]
    feats = df(rows)

    class Checked(Baseline):
        name = "checked"

        def fit(self, history):
            self.seen = history
            return super().fit(history)

        def predict(self, rows):
            wk = rows["week"][0]
            assert self.seen.height == 0 or self.seen["week"].max() < wk
            return super().predict(rows)

    out = run(Checked(), feats, seasons=[2021])
    assert out.height == len(rows)


def test_final_exam_season_is_refused():
    with pytest.raises(ValueError, match="final exam"):
        run(Baseline(), df([game("a", 2025, 1)]), seasons=[2025])


def test_perfect_predictions_score_perfectly():
    results = pl.DataFrame({
        "model": ["m"] * 30, "position": ["QB"] * 30, "season": [2021] * 30, "week": [1] * 30,
        "player_id": [str(i) for i in range(30)], "pred": [float(i) for i in range(30)], "actual": [float(i) for i in range(30)],
    })
    m = weekly_metrics(results).row(0, named=True)
    assert (m["mae"], m["spearman"], m["top_n_hit"]) == (0.0, pytest.approx(1.0), 1.0)


def test_bootstrap_interval_contains_the_mean():
    ci = bootstrap(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert ci.low <= ci.mean == 3.0 <= ci.high
