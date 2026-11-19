# TAMIU TwinMaker

A toolkit for building a digital twin of a sensor network. Describe your nodes and channels in
YAML, and you get:

1. **Ingest**: an HTTP API that accepts readings from sensor nodes (or CSV files), checks them,
   flags suspicious values and stores them without duplicates.
2. **Dashboard**: live signals per node with quality flags, forecasts, model scores and a CSV
   import page.
3. **Forecasting**: models that predict a channel a fixed time ahead, plus the chance of crossing
   a level you set, each backtested against simple baselines before it is trusted.

```
nodes / CSV files --> POST /ingest --> validation + QC --> SQLite
                                                             |
                    dashboard <-- read API <-----------------+
                                                             |
              backtest -> train -> forecast -> scorecard <---+
```

It is domain-neutral: the same code runs the two shipped examples, a stream water-quality
network and a greenhouse. TwinMaker grew out of a TAMIU senior-design project on water-quality
monitoring; this repository is the reusable framework without that project's site-specific
parts.

## Try it (one command)

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), `make`, `curl`. Windows: use WSL.

```sh
make setup      # install dependencies, create .env
make demo       # simulate 45 days of data, backtest, train, forecast, open the dashboard
make stop
```

`make demo` starts the API on port 8000 and the dashboard on http://localhost:8501, using the
water-quality example in its own database (`data/demo-water_quality/`). Try the other example
with `make demo EXAMPLE=greenhouse`, or other ports with `make demo API_PORT=8010 DASH_PORT=8511`.

**Everything the demo shows is simulated.** The simulator's numbers are arbitrary values from a
config file, so the model scores you see there prove that the pipeline works, not that anything
real is predictable. The dashboard and the reports carry a banner whenever simulated rows are
involved.

## Build your own twin

| step | what you do | guide |
|---|---|---|
| 1. Describe it | `make init EXAMPLE=water_quality`, set `PROJECT_DIR=project` in `.env`, edit `project/channels.yaml` and `project/nodes.yaml` | [docs/CONFIGURE.md](docs/CONFIGURE.md) |
| 2. Run it | `make api` in one terminal, `make dash` in another | below |
| 3. Send data | point your nodes at `POST /ingest`, import CSV history, or run the simulator | [docs/INGEST.md](docs/INGEST.md), [contract/CONTRACT.md](contract/CONTRACT.md) |
| 4. Watch it | Overview and Live signals pages | below |
| 5. Forecast it | write a task file, then backtest, train, forecast | [docs/TASKS.md](docs/TASKS.md) |

A minimal message from a node:

```sh
curl -s -X POST localhost:8000/ingest -H "X-API-Key: dev-key-change-me" -H "Content-Type: application/json" -d '{
  "schema_version": 1, "node_id": "node-upstream", "source": "sensor",
  "sent_at": "2026-10-10T15:04:05Z",
  "readings": [{"ts": "2026-10-10T15:04:00Z", "channel": "water_temp", "value": 21.4, "unit": "degC"}]
}'
```

## Dashboard pages

| page | shows |
|---|---|
| Overview | a card per node: last seen, quality counts of the last hour, latest values; the models in use |
| Live signals | time series per node and channel. Good values are a line, suspect ones hollow orange markers, bad ones red crosses; reference labels are diamonds. Raw or resampled, with CSV download |
| Forecast | the latest forecast with its likely range and event chance, past forecasts against what was measured, and the scorecard |
| Models | backtest scores of every model next to the baselines, verdicts in plain words, what the best model relies on, the model registry, and buttons to backtest, train, forecast and switch models |
| Import data | upload a CSV, map its columns to channels, import |

The sidebar sets the time range and the display time zone. Data is stored in UTC; only the
display changes.

## Commands

`make help` lists everything. The main ones:

| command | does |
|---|---|
| `make api`, `make dash` | run the API (`:8000`, docs at `/docs`) and the dashboard (`:8501`) |
| `make sim ARGS="--days 30"` | post simulated history; add `--live` to keep posting |
| `make import CSV=... MAPPING=...` | import a CSV file |
| `make backtest TASK=...` | score every model of a task against the baselines |
| `make train TASK=...` | fit and register a model |
| `make forecast TASK=...` | store forecasts from the model in use |
| `make scorecard TASK=...` | compare stored forecasts with what was measured since |
| `make promote VERSION=...` | put another registered model into use |
| `make reset-db` | drop and recreate all tables (deletes stored rows) |
| `make test`, `make test-e2e`, `make lint` | 180 unit tests, 1 end-to-end test, ruff |

## Settings (.env)

| variable | default | meaning |
|---|---|---|
| `PROJECT_DIR` | `examples/water_quality` | the project folder in use |
| `API_KEY` | `dev-key-change-me` | required for every write (`X-API-Key` header). Change it before exposing the API |
| `DB_URL` | `sqlite:///data/twin.db` | the database |
| `DATA_DIR` | `data` | backtest reports and model artifacts |
| `API_URL` | `http://localhost:8000` | where the dashboard, simulator and importer find the API |

## Layout

| path | contents |
|---|---|
| `examples/` | two complete project folders to copy from |
| `contract/` | the message format ([CONTRACT.md](contract/CONTRACT.md)), its validation, project loading |
| `api/` | FastAPI app: ingest, QC, read endpoints, ML endpoints, background jobs |
| `db/` | SQLAlchemy models (nodes, channels, observations, models, predictions, alerts) |
| `sim/` | config-driven simulator and the HTTP client with retries |
| `connectors/` | CSV importer |
| `ml/` | task specs, features, chronological splits, models, backtest, registry, forecast, scorecard |
| `dashboard/` | Streamlit pages; they read everything through the API |
| `tests/` | unit tests per package, dashboard smoke tests against a real API, one end-to-end test |

## Principles the code enforces

- **UTC everywhere.** Timestamps without an offset are rejected at the door.
- **Idempotent writes.** Re-sending a message, re-importing a file or re-running a forecast
  changes nothing.
- **Quality control only downgrades.** Suspicious values are flagged and kept, never altered or
  silently dropped; models train on `good` rows only.
- **No random splits for time series.** Backtests are chronological, and a model is called
  better than a baseline only when the data can show it.
- **No self-updating models.** Training, promotion and forecasting run only when asked, and
  every stored forecast records which model version made it and when.
- **Simulated data is always labeled.**
- **Configuration over code.** Sites, sensors, limits and forecast tasks live in YAML.

## Limits to know about

- SQLite with a single API process: fine for tens of nodes at minute-scale sampling, not for
  high-rate or multi-server use. `DB_URL` accepts other SQLAlchemy URLs, but the idempotent
  insert is written for SQLite.
- One shared API key and no TLS: run it on a trusted network or behind a reverse proxy.
- No schema migrations: `make reset-db` recreates the tables.
- No alert engine yet. The `alerts` table exists, but nothing writes to it.
- Forecast jobs started from the dashboard run inside the API process and their status is lost
  on restart (their results are not).
- Models are per-task regressors on lagged features (ridge, random forest, gradient boosting).
  No deep learning, no hyperparameter search.
- No ESP32 firmware is included; [contract/CONTRACT.md](contract/CONTRACT.md) is what firmware
  must speak.
