# TAMIU TwinMaker: developer commands. POSIX shell + make. Windows users: run under WSL.
SHELL := /bin/sh

ENV_FILE := $(wildcard .env)
RUN := uv run $(if $(ENV_FILE),--env-file .env,)

API_PORT ?= 8000
DASH_PORT ?= 8501
API_URL ?= http://localhost:$(API_PORT)

# `make demo` runs an example project in its own database and data folder, with its own key,
# so it never touches your .env, your project or your data.
EXAMPLE ?= water_quality
DEMO_DAYS ?= 45
DEMO_DIR := data/demo-$(EXAMPLE)
DEMO_ENV := PROJECT_DIR=examples/$(EXAMPLE) DB_URL=sqlite:///$(DEMO_DIR)/twin.db \
            DATA_DIR=$(DEMO_DIR) API_KEY=demo-key API_URL=$(API_URL)
DEMO_RUN := $(DEMO_ENV) uv run

.PHONY: help setup init test test-e2e lint fmt api dash sim import backtest train forecast \
        promote scorecard tasks demo stop reset-db

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup: ## install dependencies into .venv (uv); create .env from .env.example if missing
	uv sync
	@test -f .env || { cp .env.example .env; echo "created .env from .env.example (edit API_KEY before exposing the API)"; }

init: ## start your own project from an example: make init EXAMPLE=water_quality
	@test ! -e project || { echo "project/ already exists; nothing copied"; exit 1; }
	@test -d examples/$(EXAMPLE) || { echo "no example '$(EXAMPLE)'; choose one of: $$(ls examples)"; exit 1; }
	cp -R examples/$(EXAMPLE) project
	@echo "created project/ from examples/$(EXAMPLE). Now set PROJECT_DIR=project in .env and edit project/*.yaml"

test: ## unit tests (excludes the e2e marker)
	uv run pytest -q -m "not e2e"

test-e2e: ## end-to-end test (starts a real API, runs simulator, import and ML jobs)
	uv run pytest -q -m e2e tests/e2e

lint: ## ruff check + ruff format --check
	uv run ruff check .
	uv run ruff format --check .

fmt: ## ruff format + autofix
	uv run ruff format .
	uv run ruff check --fix .

api: ## run the API (foreground) on API_PORT
	$(RUN) uvicorn api.app:create_app --factory --host 127.0.0.1 --port $(API_PORT)

dash: ## run the dashboard (foreground) on DASH_PORT
	$(RUN) streamlit run dashboard/app.py --server.port $(DASH_PORT) --server.headless true

sim: ## run the simulator: make sim ARGS="--days 30"  (add --live to keep posting)
	$(RUN) python -m sim.run $(ARGS)

import: ## import a CSV: make import CSV=file.csv MAPPING=project/imports/x.yaml
	$(RUN) python -m connectors.csv_import $(CSV) --mapping $(MAPPING) $(ARGS)

tasks: ## list the forecast tasks of the project
	$(RUN) python -m ml.run tasks

backtest: ## score every model of a task: make backtest TASK=name
	$(RUN) python -m ml.run backtest --task $(TASK)

train: ## train and register a model: make train TASK=name
	$(RUN) python -m ml.run train --task $(TASK) $(ARGS)

forecast: ## store forecasts from the model in use: make forecast TASK=name
	$(RUN) python -m ml.run forecast --task $(TASK) $(ARGS)

promote: ## put a registered model into use: make promote VERSION=...
	$(RUN) python -m ml.run promote --version $(VERSION)

scorecard: ## realized error of stored forecasts: make scorecard TASK=name
	$(RUN) python -m ml.run scorecard --task $(TASK)

reset-db: ## drop and recreate the tables behind DB_URL (deletes all stored rows)
	$(RUN) python -m db.reset

demo: ## one-command demo on simulated data: make demo [EXAMPLE=greenhouse]
	@$(MAKE) -s stop
	@test -d examples/$(EXAMPLE) || { echo "no example '$(EXAMPLE)'; choose one of: $$(ls examples)"; exit 1; }
	@if curl -sf $(API_URL)/health > /dev/null 2>&1; then \
	  echo "something already answers on $(API_URL); stop it or run with API_PORT=<other>"; exit 1; fi
	@mkdir -p .run $(DEMO_DIR)
	@$(DEMO_RUN) python -m db.reset > .run/reset.log 2>&1 || { cat .run/reset.log; exit 1; }
	@echo "starting API on port $(API_PORT) (log: .run/api.log) ..."
	@$(DEMO_ENV) nohup uv run uvicorn api.app:create_app --factory --host 127.0.0.1 --port $(API_PORT) > .run/api.log 2>&1 & echo $$! > .run/api.pid
	@i=0; until curl -sf $(API_URL)/health > /dev/null 2>&1; do \
	  i=$$((i+1)); if [ $$i -gt 60 ]; then echo "API did not start; see .run/api.log"; $(MAKE) -s stop; exit 1; fi; sleep 0.5; done
	@echo "API healthy. Simulating $(DEMO_DAYS) days of node data ..."
	@$(DEMO_RUN) python -m sim.run --days $(DEMO_DAYS) --seed 1 || { $(MAKE) -s stop; exit 1; }
	@task=$$($(DEMO_RUN) python -m ml.run tasks | head -1); \
	  echo "Backtesting task $$task ..."; \
	  $(DEMO_RUN) python -m ml.run backtest --task $$task > .run/backtest.log 2>&1 || { cat .run/backtest.log; $(MAKE) -s stop; exit 1; }; \
	  echo "Training (newest 3 days held back so stored forecasts can be scored) ..."; \
	  $(DEMO_RUN) python -m ml.run train --task $$task --holdback-days 3 > .run/train.log 2>&1 || { cat .run/train.log; $(MAKE) -s stop; exit 1; }; \
	  echo "Storing forecasts ..."; \
	  $(DEMO_RUN) python -m ml.run forecast --task $$task --backfill-steps 200 > .run/forecast.log 2>&1 || { cat .run/forecast.log; $(MAKE) -s stop; exit 1; }
	@echo "starting dashboard on port $(DASH_PORT) (log: .run/dash.log) ..."
	@$(DEMO_ENV) nohup uv run streamlit run dashboard/app.py --server.port $(DASH_PORT) --server.headless true > .run/dash.log 2>&1 & echo $$! > .run/dash.pid
	@echo ""
	@echo "Dashboard: http://localhost:$(DASH_PORT)"
	@echo "API docs:  $(API_URL)/docs   (demo API key: demo-key)"
	@echo "Everything shown is SIMULATED. Stop with: make stop"

stop: ## stop the demo API and dashboard (safe when nothing is running)
	@for name in api dash; do \
	  if [ -f .run/$$name.pid ]; then \
	    pid=$$(cat .run/$$name.pid); \
	    if kill -0 $$pid 2>/dev/null; then \
	      kids=$$(pgrep -P $$pid); grandkids=""; for k in $$kids; do grandkids="$$grandkids $$(pgrep -P $$k)"; done; \
	      kill $$pid $$kids $$grandkids 2>/dev/null; \
	      for n in 1 2 3 4 5 6 7 8 9 10; do kill -0 $$pid 2>/dev/null || break; sleep 0.5; done; \
	      kill -9 $$pid $$kids $$grandkids 2>/dev/null; \
	      echo "stopped $$name (pid $$pid)"; \
	    fi; \
	    rm -f .run/$$name.pid; \
	  fi; \
	done
