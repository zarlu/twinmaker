"""End to end on the shipped water-quality example, using only the documented commands:

real API process -> simulator over HTTP -> CSV import -> backtest -> train -> forecast ->
scorecard -> read everything back through the API.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.e2e
REPO = Path(__file__).resolve().parents[2]
KEY = "e2e-key"
TASK = "do_6h"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict]:
    tmp = tmp_path_factory.mktemp("e2e")
    url = f"http://127.0.0.1:{_free_port()}"
    env = {
        **os.environ,
        "PROJECT_DIR": "examples/water_quality",
        "DB_URL": f"sqlite:///{tmp / 'twin.db'}",
        "DATA_DIR": str(tmp / "data"),
        "API_KEY": KEY,
        "API_URL": url,
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.app:create_app", "--factory", "--port",
         url.rsplit(":", 1)[1]],  # fmt: skip
        cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    try:
        deadline = time.time() + 30
        while True:
            try:
                if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            assert time.time() < deadline, "API did not start"
            time.sleep(0.2)
        yield {"url": url, "env": env}
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def _cli(stack: dict, *args: str) -> str:
    done = subprocess.run(
        [sys.executable, "-m", *args], cwd=REPO, env=stack["env"], capture_output=True, text=True,
        timeout=300,
    )  # fmt: skip
    assert done.returncode == 0, f"{args}: {done.stdout}\n{done.stderr}"
    return done.stdout


def test_the_whole_loop(stack: dict) -> None:
    url = stack["url"]
    project = httpx.get(f"{url}/project").json()
    assert project["name"] == "Water quality example" and project["tasks"] == [TASK]

    # 1. simulated nodes post over HTTP; a second run is all duplicates
    out = _cli(stack, "sim.run", "--days", "14", "--seed", "1")
    assert " 0 rejected, 0 failed requests" in out and "source='simulator'" in out
    nodes = {n["node_id"]: n for n in httpx.get(f"{url}/nodes").json()}
    assert set(nodes) == {"node-upstream", "node-downstream"}
    assert all(n["n_rows"] > 5000 and n["minutes_since_seen"] < 20 for n in nodes.values())
    flagged = httpx.get(
        f"{url}/observations", params={"node_id": "node-upstream", "channel": "turbidity"}
    ).json()
    assert "suspect" in flagged["quality"]  # server QC caught injected spikes

    # 2. CSV import of the shipped sample, with unit conversion and local time
    sample = "examples/water_quality/imports/sample_logger.csv"
    mapping = "examples/water_quality/imports/logger_wide.yaml"
    out = _cli(stack, "connectors.csv_import", sample, "--mapping", mapping)
    assert "8 rows -> 31 readings for 2 node(s)" in out and "accepted 31" in out
    assert "accepted 0, duplicates 31" in _cli(
        stack, "connectors.csv_import", sample, "--mapping", mapping
    )
    imported = httpx.get(
        f"{url}/observations",
        params={"source": "logger-export", "channel": "water_temp", "node_id": "node-upstream"},
    ).json()
    assert imported["ts"][0].startswith("2026-06-01T13:00:00")  # 08:00 CDT
    assert imported["value"][0] == pytest.approx((70.5 - 32) * 5 / 9, abs=1e-4)

    # 3. backtest -> train -> forecast -> scorecard
    assert _cli(stack, "ml.run", "tasks").split() == [TASK]
    backtest = json.loads(_cli(stack, "ml.run", "backtest", "--task", TASK))
    assert backtest["best_model"] in ("ridge", "random_forest", "gradient_boosting")
    assert (Path(backtest["report_path"]) / "report.md").read_text().startswith("# Backtest")
    train = json.loads(_cli(stack, "ml.run", "train", "--task", TASK, "--holdback-days", "2"))
    assert train["status"] == "champion" and train["algorithm"] == backtest["best_model"]
    forecast = json.loads(
        _cli(stack, "ml.run", "forecast", "--task", TASK, "--backfill-steps", "60")
    )
    assert forecast["nodes"] == 2 and forecast["written"] == 2 * 61 * 2
    card = json.loads(_cli(stack, "ml.run", "scorecard", "--task", TASK))
    assert card["n_scored"] > 20 and len(card["rows"]) == 2

    # 4. the API serves what the jobs produced
    bench = httpx.get(f"{url}/benchmarks/{TASK}").json()
    assert bench["simulated"] is True and "holdout" in bench["blocks"]
    assert "logger-export" in bench["sources"]  # imported June rows are far from the rest
    (model,) = httpx.get(f"{url}/models").json()
    assert model["version"] == train["version"] and model["status"] == "champion"
    preds = httpx.get(f"{url}/predictions", params={"model_version": model["version"]}).json()
    assert preds["count"] == 2 * 61 and all(p is not None for p in preds["p_event"])
    assert (
        httpx.get(f"{url}/scorecard", params={"task": TASK}).json()["n_scored"] == card["n_scored"]
    )
