import importlib

import pytest

MODULES = [
    "src.ingest",
    "src.scoring",
    "src.features",
    "src.models",
    "src.backtest",
    "src.draft",
]


@pytest.mark.parametrize("name", MODULES)
def test_packages_import(name):
    importlib.import_module(name)
