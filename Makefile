.PHONY: verify test lint smoke

PYTHON ?= python

verify: test lint smoke

test:
	$(PYTHON) -m pytest tests -q

lint:
	$(PYTHON) -m ruff check streaminfer tests

smoke:
	./scripts/smoke-local.sh
