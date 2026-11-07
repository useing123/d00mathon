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

submit:  ## control window: fetch GROUP -> evaluate (default preset) -> export JSON 1.0 -> validate -> commit (push is manual)
	@test -n "$(GROUP)" || (echo "usage: make submit GROUP=<group id>  (DEV -> results/dev, anything else -> results/prod)"; exit 2)
	$(eval SUB := $(if $(filter DEV dev,$(GROUP)),dev,prod))
	$(PY) -m cqa evaluate --refresh --group $(GROUP) --out results/$(GROUP)_submit.json
	$(PY) -m cqa export --format v1 --group $(GROUP) --run $(GROUP)_submit --out results/$(SUB)/results.json
	$(PY) -m cqa validate results/$(SUB)/results.json --group $(GROUP) --allow-custom
	git add -f results/$(SUB)/results.json
	git -c user.name=cqa -c user.email=cqa@local commit -m "Submission: results/$(SUB)/results.json (group $(GROUP), JSON 1.0)" results/$(SUB)/results.json
	@echo "committed results/$(SUB)/results.json - now: git push"

submit-dev:  ## re-export results/dev/results.json from an existing DEV run, no LLM calls (RUN=dev_jury_v7)
	$(PY) -m cqa export --format v1 --group DEV $(if $(RUN),--run $(RUN)) --out results/dev/results.json
	$(PY) -m cqa validate results/dev/results.json --group DEV
