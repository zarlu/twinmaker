"""Forecast task specs: ``<project>/tasks/<name>.yaml`` loaded, defaulted and validated.

A task says *what to forecast*: one target channel, how far ahead, from which inputs, and
(optionally) which threshold crossing counts as an event. The runner has no task-specific code.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import pandas as pd

from contract.project import REPO_ROOT, ProjectError, project_dir, read_yaml
from contract.registry import load_registry

ALGORITHMS = ("ridge", "random_forest", "gradient_boosting")
CALENDAR = ("hour_of_day", "day_of_week", "day_of_year")
PERSISTENCE_VERSION = "persistence-v1"

DEFAULTS: dict[str, Any] = {
    "description": "",
    "resolution": "1h",
    "horizon_steps": 1,
    "seasonal_period_steps": None,  # e.g. 24 on an hourly grid = "same time yesterday"
    "event": None,
    "features": {
        "target_lags": [0, 1, 2, 3],
        "rolling_windows": [],
        "channels": {},
        "calendar": [],
    },
    "data": {"min_quality": "good", "exclude_sources": [], "max_gap_steps": 2},
    "split": {"n_folds": 4, "holdout_frac": 0.2, "min_train_rows": 100},
    "models": {
        "ridge": {"alpha": 1.0},
        "random_forest": {"n_estimators": 200, "min_samples_leaf": 5},
        "gradient_boosting": {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15},
    },
    "intervals": {"level": 0.9},
    "eval": {"n_boot": 500, "ci": 0.9},
    "deploy": {"algorithm": None},  # None = best walk-forward MAE of the latest backtest
    "seed": 7,
}


class TaskError(Exception):
    """A task spec is missing or invalid."""


def tasks_dir() -> Path:
    return project_dir() / "tasks"


def data_dir() -> Path:
    """Where reports and model artifacts are written (``DATA_DIR``, default ``data``)."""
    path = Path(os.environ.get("DATA_DIR") or "data")
    return path if path.is_absolute() else REPO_ROOT / path


def list_tasks() -> list[str]:
    """Names of the task files of the active project."""
    folder = tasks_dir()
    if not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.glob("*.yaml") if not p.name.startswith("_"))


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in over.items():
        # `models` is replaced as a whole: listing two models means running two models.
        if isinstance(value, dict) and isinstance(out.get(key), dict) and key != "models":
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_task(name: str) -> dict[str, Any]:
    """Load, default and validate one task."""
    path = tasks_dir() / f"{name}.yaml"
    try:
        raw = read_yaml(path)
    except ProjectError as exc:
        raise TaskError(f"unknown task '{name}': {exc}") from exc
    return validate_task({**raw, "name": name})


def validate_task(raw: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults and check every field; raises ``TaskError`` with the offending key."""
    spec = _merge(DEFAULTS, raw)
    name = spec.get("name", "?")

    def fail(msg: str) -> None:
        raise TaskError(f"task '{name}': {msg}")

    registry = load_registry()
    target = spec.get("target")
    if not isinstance(target, dict) or "channel" not in target:
        fail("`target.channel` is required")
    if target["channel"] not in registry:
        fail(f"target channel '{target['channel']}' is not in channels.yaml")
    target.setdefault("nodes", "all")
    target.setdefault("kind", str(registry[target["channel"]].default_kind))
    target["unit"] = registry[target["channel"]].unit
    try:
        step = pd.Timedelta(spec["resolution"])
    except ValueError:
        fail(f"`resolution` '{spec['resolution']}' is not a duration such as 15min, 1h or 1D")
    if step <= pd.Timedelta(0):
        fail("`resolution` must be positive")
    if int(spec["horizon_steps"]) < 1:
        fail("`horizon_steps` must be at least 1")
    spec["horizon_steps"] = int(spec["horizon_steps"])
    spec["horizon_h"] = round(spec["horizon_steps"] * step.total_seconds() / 3600.0, 6)

    feats = spec["features"]
    if not feats["target_lags"] or any(int(v) < 0 for v in feats["target_lags"]):
        fail("`features.target_lags` needs at least one lag, all >= 0 (0 = value at issue time)")
    if any(int(w) < 2 for w in feats["rolling_windows"]):
        fail("`features.rolling_windows` entries must be >= 2 steps")
    for channel, lags in feats["channels"].items():
        if channel not in registry:
            fail(f"feature channel '{channel}' is not in channels.yaml")
        if not lags or any(int(v) < 0 for v in lags):
            fail(f"`features.channels.{channel}` needs a list of lags >= 0")
    unknown = set(feats["calendar"]) - set(CALENDAR)
    if unknown:
        fail(f"unknown calendar features {sorted(unknown)}; allowed: {list(CALENDAR)}")

    event = spec["event"]
    if event is not None:
        if "threshold" not in event:
            fail("`event.threshold` is required when `event` is set")
        event.setdefault("direction", "above")
        if event["direction"] not in ("above", "below"):
            fail("`event.direction` must be 'above' or 'below'")
        event["threshold"] = float(event["threshold"])

    if not spec["models"]:
        fail("`models` needs at least one algorithm")
    for algo in spec["models"]:
        if algo not in ALGORITHMS:
            fail(f"unknown algorithm '{algo}'; allowed: {list(ALGORITHMS)}")
        spec["models"][algo] = dict(spec["models"][algo] or {})
    deploy_algo = spec["deploy"].get("algorithm")
    if deploy_algo is not None and deploy_algo not in spec["models"]:
        fail(f"`deploy.algorithm` '{deploy_algo}' is not listed under `models`")
    if not 0 < float(spec["split"]["holdout_frac"]) < 0.5:
        fail("`split.holdout_frac` must be between 0 and 0.5")
    if int(spec["split"]["n_folds"]) < 1:
        fail("`split.n_folds` must be at least 1")
    if not 0.5 <= float(spec["intervals"]["level"]) < 1:
        fail("`intervals.level` must be in [0.5, 1)")
    if spec["data"]["min_quality"] not in ("good", "suspect", "bad"):
        fail("`data.min_quality` must be good, suspect or bad")
    return spec


def step_of(spec: dict[str, Any]) -> pd.Timedelta:
    return pd.Timedelta(spec["resolution"])


def event_hit(values: Any, spec: dict[str, Any]) -> Any:
    """Boolean array: does each value count as the task's event?"""
    event = spec["event"]
    if event["direction"] == "above":
        return values >= event["threshold"]
    return values <= event["threshold"]
