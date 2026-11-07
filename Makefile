PY ?= .venv/bin/python
PORT ?= 8090
GROUP ?= DEV

.PHONY: install test lint fmt serve demo eval-dev eval-group docker

install:
	python3 -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt ruff

test:
	CQA_PROVIDER=mock $(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

fmt:
	$(PY) -m ruff check . --fix

serve:
	$(PY) -m cqa serve --port $(PORT)

demo:  ## offline end-to-end run on synthetic dialogs, no API key
	$(PY) -m cqa evaluate --mock --fresh --input examples/sample_dialogs.json --group DEMO

eval-dev:
	$(PY) -m cqa evaluate --group DEV

eval-group:
	$(PY) -m cqa evaluate --group $(GROUP)

docker:
	docker build -t cqa .
