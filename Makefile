PY ?= .venv/bin/python
PORT ?= 8090
GROUP ?= DEV

.PHONY: submit install test lint fmt serve demo eval-dev eval-group docker

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

submit:  ## control window: evaluate GROUP in --final mode, export v1.0, validate, commit results/prod/results.json
	@test -n "$(GROUP)" && test "$(GROUP)" != "DEV" || (echo "usage: make submit GROUP=<prod group id>"; exit 2)
	$(PY) -m cqa evaluate --final --refresh --group $(GROUP) --out results/$(GROUP)_final.json
	$(PY) -m cqa export --format v1 --group $(GROUP) --run $(GROUP)_final --out results/prod/results.json
	$(PY) -m cqa validate results/prod/results.json
	git add -f results/prod/results.json
	git -c user.name=cqa -c user.email=cqa@local commit -m "Submission: results/prod/results.json (group $(GROUP), v1.0)" results/prod/results.json

submit-dev:  ## regenerate results/dev/results.json from a DEV run (RUN=dev_jury_v6)
	$(PY) -m cqa export --format v1 --group DEV $(if $(RUN),--run $(RUN)) --out results/dev/results.json
	$(PY) -m cqa validate results/dev/results.json
