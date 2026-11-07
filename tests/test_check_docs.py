import importlib.util
from pathlib import Path

_p = Path(__file__).resolve().parent.parent / "scripts" / "check_docs.py"
_spec = importlib.util.spec_from_file_location("check_docs", _p)
check_docs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_docs)


def test_detects_bad_patterns():
    assert check_docs.check_text("Qwen3-8B + LoRA, 1 эпоха")
    assert check_docs.check_text("trained for one epoch")
    assert check_docs.check_text("170+ тестов, CI")
    assert check_docs.check_text("192 автотеста")
    assert check_docs.check_text("we have 158 tests")


def test_allows_correct_patterns():
    assert not check_docs.check_text("16 шагов (≈ 0.32 эпохи)")
    assert not check_docs.check_text("16 steps ≈ 0.32 epoch; 2 epochs planned")
    assert not check_docs.check_text("Полный набор тестов, CI")


def test_repo_docs_clean():
    bad = []
    for p in check_docs.doc_files():
        bad += [f"{p.name}:{m}" for m in check_docs.check_text(p.read_text(encoding="utf-8"))]
    assert not bad, "\n".join(bad)
