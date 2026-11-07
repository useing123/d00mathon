"""Shared fixtures: every test runs offline (mock provider) with isolated data/results/cache dirs."""
import pytest

from cqa import config


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    for v in ("CQA_PII_MASK", "CQA_EVIDENCE_STEP", "CQA_MODEL", "CQA_ENSEMBLE", "CQA_JURORS", "CQA_JUDGE"):
        monkeypatch.delenv(v, raising=False)   # shell env must not leak into tests
    monkeypatch.setattr(config, "PROVIDER", "mock")
    monkeypatch.setattr(config, "ENSEMBLE", False)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    return tmp_path
