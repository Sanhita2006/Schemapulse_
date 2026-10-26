# --------------------------------------------------------------------------- #
# SchemaPulse — Execution-Verified PostgreSQL Optimization Agent
#
#   make install      editable install + dev extras
#   make demo-up      start the two demo PostgreSQL servers
#   make seed         create schema + deterministic data in both servers
#   make inspect      deterministic (no-AI) baseline inspection
#   make diagnose     full optimize workflow against the golden query
#   make demo         end-to-end golden-path demo
#   make test         unit tests (no database required)
#   make test-all     unit + integration (needs demo-up && seed)
#   make lint         ruff check + format --check
#   make typecheck    mypy
#   make build        build wheel + sdist
#   make demo-down    stop the demo servers
#   make check        lint + typecheck + unit tests
#
# All Python entry points go through the project venv at .venv
# --------------------------------------------------------------------------- #

VENV    := .venv
PY      := $(VENV)/Scripts/python.exe
SP      := $(VENV)/Scripts/schemapulse.exe
PIP     := $(VENV)/Scripts/python.exe -m pip
UV      ?= uv

.DEFAULT_GOAL := help
.PHONY: help venv install demo-up demo-down seed inspect diagnose demo report doctor \
        provider test test-unit test-all lint fmt typecheck check build clean

help:
	@$(PY) -c "import sys;print('SchemaPulse Makefile targets:')" 2>NUL || echo SchemaPulse Makefile targets:
	@echo   install / venv / demo-up / seed / inspect / diagnose / demo
	@echo   report / doctor / provider / test / test-all / lint / fmt
	@echo   typecheck / check / build / demo-down / clean

venv: ## Create the project virtualenv with uv (Python 3.12)
	$(UV) venv $(VENV) --python 3.12

install: ## Editable install with dev extras
	$(UV) pip install --python $(PY) -e ".[dev]"

demo-up: ## Start the demo PostgreSQL servers (baseline :5432, shadow :5433)
	docker compose up -d --wait

demo-down: ## Stop the demo PostgreSQL servers
	docker compose down -v

seed: ## Create schema + deterministic data in both servers (idempotent)
	$(SP) demo --seed-only

inspect: ## Deterministic inspection of the golden slow query (no AI)
	$(SP) inspect demo/slow_orders.sql

diagnose: ## Full diagnose workflow against the golden query
	$(SP) diagnose demo/slow_orders.sql

demo: ## End-to-end golden-path demo (seed + inspect + diagnose + report)
	$(SP) demo

report: ## Render the latest HTML report from the last diagnosis
	$(SP) report

doctor: ## Environment health check
	$(SP) doctor

provider: ## Show the active Gemma provider and whether it is live
	$(SP) provider

test-unit: ## Unit tests only (no database, no network)
	$(PY) -m pytest tests/unit -q

test-all: ## Unit + integration tests (requires: make demo-up && make seed)
	$(PY) -m pytest -q

test: test-unit ## Default test target

lint: ## Lint + format check with ruff
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

fmt: ## Auto-format and autofix with ruff
	$(PY) -m ruff format src tests scripts
	$(PY) -m ruff check --fix src tests scripts

typecheck: ## Static type check with mypy
	$(PY) -m mypy

build: ## Build wheel + sdist
	$(UV) build

check: lint typecheck test-unit ## lint + typecheck + unit tests

clean: ## Remove build artifacts and caches
	-$(PY) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['build','dist','.pytest_cache','.ruff_cache','.mypy_cache','.schemapulse']]"
