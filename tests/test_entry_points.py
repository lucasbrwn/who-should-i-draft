"""Every command-line entry point must at least import (catches syntax errors in files no other test loads)."""

import importlib

import pytest

MODULES = [
    "src.predict.__main__", "src.predict.week", "src.predict.benchmarks", "src.predict.next_week",
    "src.ingest.__main__", "src.ingest.props", "src.ingest.depth_charts", "src.features.__main__",
    "src.backtest.__main__", "src.backtest.vs_espn", "src.backtest.play_chance_test", "src.backtest.scenario_test",
    "src.backtest.route_estimate_test", "src.backtest.calibration_test", "src.backtest.weak_points",
]


@pytest.mark.parametrize("module", MODULES)
def test_imports(module):
    importlib.import_module(module)
