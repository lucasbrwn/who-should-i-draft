"""Boosting tuning (SCRUM-50). Rules agreed 2026-11-10."""

import json

import polars as pl
import pytest

from src.models import boosting, tune
from test_ridge import make_feats


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(tune, "MODELED", {"WR": ["receptions"]})
    return make_feats(seasons=(2020, 2021, 2022, 2023, 2024), weeks=range(1, 6), players=12)


def test_random_settings_stay_in_the_design_doc_ranges():
    for s in tune.random_settings(200):
        assert 0.02 <= s["learning_rate"] <= 0.2
        assert 15 <= s["max_leaf_nodes"] <= 63
        assert 20 <= s["min_samples_leaf"] <= 200
        assert 0 <= s["l2_regularization"] <= 1


def test_tuning_never_looks_at_2024_or_later(small):
    before = tune.tune(small, n_settings=2)
    scrambled = small.with_columns(
        y_receptions=pl.when(pl.col("season") >= 2024).then(999.0).otherwise(pl.col("y_receptions"))
    )
    assert tune.tune(scrambled, n_settings=2) == before


def test_best_is_never_worse_than_the_defaults(small):
    result = tune.tune(small, n_settings=3)["WR|receptions"]
    assert result["score"] <= result["default_score"]


def test_tuned_model_loads_saved_settings(small, tmp_path, monkeypatch):
    path = tmp_path / "params.json"
    path.write_text(json.dumps({"WR|receptions": {"max_iter": 7, "learning_rate": 0.1, "score": 1.0}}))
    monkeypatch.setattr(boosting, "TUNED_PATH", path)
    model = boosting.BoostingModel(tuned=True).fit(small)
    assert model.name == "boosting_tuned"
    assert model.models[("WR", "receptions")].max_iter == 7          # tuned value used
    assert "score" not in model.models[("WR", "receptions")].get_params()


def test_tuned_settings_can_apply_to_td_stats_only(small, tmp_path):
    path = tmp_path / "params.json"
    path.write_text(json.dumps({"WR|receptions": {"max_iter": 7}, "WR|receiving_tds": {"max_iter": 9}}))
    model = boosting.BoostingModel(tuned=path, tuned_stats=boosting.TD_STATS, name="td")
    assert model.tuned == {"WR|receiving_tds": {"max_iter": 9}}
    assert model.name == "td"


def test_rank_tuning_is_never_worse_than_defaults_and_shares_one_setting(small, monkeypatch):
    monkeypatch.setattr(tune, "MODELED", {"WR": ["receptions", "receiving_yards"]})
    monkeypatch.setattr(tune, "STARTERS", {"WR": 2})
    feats = small.with_columns(y_receiving_yards=pl.col("y_receptions") * 11)
    best = tune.tune_rank(feats, n_settings=2)
    assert best["WR|receptions"]["spearman"] >= best["WR|receptions"]["default_spearman"]
    strip = lambda d: {k: v for k, v in d.items() if k not in ("spearman", "default_spearman")}  # noqa: E731
    assert strip(best["WR|receptions"]) == strip(best["WR|receiving_yards"])
