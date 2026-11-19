# Forecast your data

A **task** says what to predict: one channel, a fixed time ahead, from recent history. Each task
is one YAML file in the project's `tasks/` folder.

## The loop

```sh
make backtest TASK=do_6h     # 1. score every model against simple baselines; changes nothing
make train TASK=do_6h        # 2. fit the best one on all data and register it
make forecast TASK=do_6h     # 3. store forecasts for every node (run this on a schedule)
make scorecard TASK=do_6h    # 4. later: how did the stored forecasts turn out?
```

The same four steps are buttons on the dashboard's **Models** page; results show on **Forecast**.
Nothing retrains or switches models by itself. To forecast continuously, schedule step 3 (for
example with cron) at the task's resolution.

## A task file

Only `target.channel` is required.

```yaml
description: Dissolved oxygen six hours ahead.
target:
  channel: dissolved_oxygen
  nodes: all                 # or [node-a, node-b]; several nodes share one model
resolution: 1h               # readings are averaged onto this grid (15min, 1h, 1D, ...)
horizon_steps: 6             # how many grid steps ahead
seasonal_period_steps: 24    # optional: adds a "one cycle ago" baseline (24 x 1h = yesterday)
event:                       # optional: also estimate the chance of crossing a level
  threshold: 6.0
  direction: below           # above | below
features:
  target_lags: [0, 1, 2, 3, 6, 12, 24]   # target value N steps before the forecast is issued
  rolling_windows: [6, 24]               # mean and spread of the target over the last N steps
  channels:                              # other channels at the same node, with their lags
    water_temp: [0, 3, 6]
  calendar: [hour_of_day]                # also: day_of_week, day_of_year
models:                      # each listed model is backtested with exactly these settings
  ridge: {alpha: 1.0}
  random_forest: {n_estimators: 150, min_samples_leaf: 5}
  gradient_boosting: {max_iter: 200, learning_rate: 0.05, max_leaf_nodes: 15}
```

Other keys and their defaults:

| key | default | meaning |
|---|---|---|
| `data.min_quality` | `good` | which rows may be used: `good`, `suspect` (good and suspect) or `bad` (all) |
| `data.exclude_sources` | `[]` | sources to leave out, for example `[simulator]` once real data exists |
| `data.max_gap_steps` | `2` | gaps up to this many grid steps are filled with the last known value |
| `split.n_folds` | `4` | walk-forward test folds |
| `split.holdout_frac` | `0.2` | newest share of the data kept for one final test |
| `split.min_train_rows` | `100` | a fold with fewer training rows is skipped |
| `intervals.level` | `0.9` | nominal coverage of the likely range |
| `eval.n_boot`, `eval.ci` | `500`, `0.9` | bootstrap settings for the verdicts |
| `deploy.algorithm` | best walk-forward MAE | force which algorithm `train` fits |
| `seed` | `7` | same data and seed give the same numbers |

## How the backtest stays honest

- **Features never see the future.** Every input for a forecast issued at time *t* is computed
  from readings at or before *t*. A test changes all future values and checks that no feature
  moves.
- **Splits are chronological.** The data is cut by time into walk-forward folds plus a final
  hold-out (the newest 20%). A training row is used only if its outcome was already known when
  the first test forecast was issued. There is no random split anywhere.
- **Every model sits next to baselines.** *Persistence* (the current value carried forward) is
  always scored; *seasonal naive* (the value one cycle earlier) when the task sets a period. A
  forecast model is only worth running if it beats them.
- **"Beats" needs evidence.** The report says a model beats a baseline only when the 90%
  bootstrap interval of the error difference excludes zero. The bootstrap resamples whole days,
  because errors within a day are not independent. Otherwise the wording is "does NOT
  demonstrably beat".
- **No tuning on the test data.** Model settings are fixed in the task file; there is no
  hyperparameter search.
- **Simulated data is labeled.** A backtest that used `source='simulator'` rows says so at the
  top of the report and on the dashboard.

Reports are written to `data/reports/<task>/<timestamp>/report.md` and `metrics.json`.

## Likely range and event chance

Both come from the model's own out-of-sample walk-forward errors:

- **Likely range**: forecast plus or minus the 90% quantile of past absolute errors
  (split-conformal). The backtest reports how often the hold-out values really fell inside.
- **Event chance**: the share of past errors that, added to the forecast, would cross the
  threshold.

Both assume future errors resemble past ones. They will be too narrow after a change the model
has never seen (a new sensor, a new season, a fault).

## Registry and scorecard

`train` saves the model under `data/artifacts/` and adds a row to the registry. The first model
of a task goes into use at once; later ones are **candidates** until you run
`make promote VERSION=...` (or press "Use this model"). Every stored forecast carries its model
version and issue time, and the persistence forecast is stored next to it.

The scorecard compares stored forecasts with what was measured afterwards, on forecasts for
times after the model's training data ended. It is the real test: it uses no data the model was
fitted to and it keeps growing as time passes.

## How much data is enough?

More than the defaults will accept. The backtest warns when the first fold trains on fewer than
200 rows, when a test window covers fewer than 14 days, and when there are fewer than 10 event
rows. A model that looks good on three weeks of one season has not been tested on the others.
Treat early verdicts as provisional and re-run the backtest as data accumulates.
