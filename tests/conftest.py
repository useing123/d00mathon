"""Shared fixtures: every test runs offline (mock provider) with isolated data/results/cache dirs."""
import pytest

from cqa import config


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROVIDER", "mock")
    monkeypatch.setattr(config, "ENSEMBLE", False)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    return tmp_path
