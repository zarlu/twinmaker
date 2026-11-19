"""Task specs: defaults, validation messages, and the shipped examples."""

from __future__ import annotations

from pathlib import Path

import pytest

from contract.project import REPO_ROOT, ProjectError, load_project, project_dir
from contract.registry import load_registry
from ml.config import TaskError, list_tasks, load_task, validate_task
from sim.generate import load_sim_config

BASE = {"name": "t", "target": {"channel": "water_temp"}}


def test_defaults_are_filled_in() -> None:
    spec = validate_task(dict(BASE))
    assert spec["resolution"] == "1h" and spec["horizon_steps"] == 1 and spec["horizon_h"] == 1.0
    assert spec["target"] == {"channel": "water_temp", "nodes": "all", "kind": "measurement",
                              "unit": "degC"}  # fmt: skip
    assert spec["event"] is None and set(spec["models"]) == {
        "ridge", "random_forest", "gradient_boosting"}  # fmt: skip


def test_test_task_loads_with_its_overrides() -> None:
    spec = load_task("temp_3h")
    assert spec["horizon_h"] == 3.0 and list(spec["models"]) == ["ridge", "random_forest"]
    assert spec["event"] == {"threshold": 22.0, "direction": "above"}
    assert spec["split"]["min_train_rows"] == 50 and spec["eval"]["ci"] == 0.9
    assert list_tasks() == ["temp_3h"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"target": {}}, "`target.channel` is required"),
        ({"target": {"channel": "nope"}}, "not in channels.yaml"),
        ({"resolution": "soon"}, "not a duration"),
        ({"horizon_steps": 0}, "at least 1"),
        ({"features": {"target_lags": []}}, "at least one lag"),
        ({"features": {"channels": {"nope": [0]}}}, "feature channel 'nope'"),
        ({"features": {"calendar": ["moon_phase"]}}, "unknown calendar features"),
        ({"event": {"direction": "above"}}, "`event.threshold` is required"),
        ({"event": {"threshold": 1, "direction": "sideways"}}, "'above' or 'below'"),
        ({"models": {"lstm": {}}}, "unknown algorithm 'lstm'"),
        ({"models": {}}, "at least one algorithm"),
        ({"deploy": {"algorithm": "ridge"}, "models": {"random_forest": {}}}, "not listed"),
        ({"split": {"holdout_frac": 0.9}}, "holdout_frac"),
        ({"intervals": {"level": 1.5}}, "intervals.level"),
    ],
)
def test_invalid_specs_name_the_offending_key(change: dict, message: str) -> None:
    with pytest.raises(TaskError, match=message):
        validate_task({**BASE, **change})


def test_unknown_task_and_missing_project(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(TaskError, match="unknown task 'nope'"):
        load_task("nope")
    monkeypatch.setenv("PROJECT_DIR", "no/such/folder")
    with pytest.raises(ProjectError, match="does not exist"):
        project_dir()


EXAMPLES = sorted(p.name for p in (REPO_ROOT / "examples").iterdir() if p.is_dir())


def test_examples_are_present() -> None:
    assert EXAMPLES == ["greenhouse", "water_quality"]


@pytest.mark.parametrize("example", EXAMPLES)
def test_every_example_project_is_valid(example: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each shipped example must load: registry, nodes, simulator and every task."""
    monkeypatch.setenv("PROJECT_DIR", str(Path("examples") / example))
    registry = load_registry()
    project = load_project()
    assert project["name"] and set(project["key_channels"]) <= set(registry)
    cfg = load_sim_config()
    assert set(cfg["channels"]) <= set(registry) and len(cfg["nodes"]) == 2
    tasks = list_tasks()
    assert tasks
    for name in tasks:
        spec = load_task(name)
        assert spec["target"]["channel"] in cfg["channels"]  # the demo can forecast it
        assert set(spec["features"]["channels"]) <= set(cfg["channels"])
