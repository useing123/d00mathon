# Contributing

1. `make install` — venv + deps + ruff.
2. `make test` — runs fully offline on the mock provider (`CQA_PROVIDER=mock`); no API key needed.
3. `make lint` must pass (ruff config in `pyproject.toml`).
4. Never commit `.env`, `data/`, `results/`, `.cache/` or real client transcripts. Use `examples/` for synthetic data only.
5. Changing a prompt or rubric? Bump `PROMPT_VERSION` in `cqa/prompts.py` if the change should invalidate the per-criterion cache globally (rubric edits invalidate only their own criterion automatically).
6. Scoring changes must come with a test in `tests/test_scoring*.py` and an update to `docs/SCORING.md`.
