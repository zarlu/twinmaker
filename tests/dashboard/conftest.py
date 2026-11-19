"""Start real API servers (subprocess uvicorn, temporary SQLite) for AppTest smoke tests."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from sim.client import SimClient
from sim.generate import generate, load_sim_config, to_messages

REPO = Path(__file__).resolve().parents[2]
KEY = "dash-test-key"
TASK = "temp_3h"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ApiServer:
    def __init__(self, tmp: Path) -> None:
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.key = KEY
        env = {
            **os.environ,
            "DB_URL": f"sqlite:///{tmp / 'api.db'}",
            "DATA_DIR": str(tmp / "data"),
            "API_KEY": KEY,
        }
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "api.app:create_app",
                "--factory",
                "--port",
                str(self.port),
            ],  # fmt: skip
            cwd=REPO,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                if httpx.get(f"{self.url}/health", timeout=1).status_code == 200:
                    return
            except httpx.HTTPError:
                time.sleep(0.2)
        self.stop()
        raise RuntimeError("API server did not start")

    def fill(self, days: float = 12.0) -> None:
        """Post simulated data ending now."""
        now = datetime.now(UTC)
        frames, _ = generate(load_sim_config(), now - timedelta(days=days), now, seed=1)
        client = SimClient(self.url, KEY)
        for message in to_messages(frames, 96, sent_at=now):
            client.post("/ingest", message)
        assert client.stats.accepted > 0 and client.stats.failed_requests == 0

    def job(self, kind: str, **args: object) -> dict:
        """Run one ML job through the API and wait for it."""
        headers = {"X-API-Key": KEY}
        job = httpx.post(f"{self.url}/jobs/{kind}", json=args, headers=headers, timeout=30).json()
        deadline = time.time() + 120
        while job["status"] in ("queued", "running") and time.time() < deadline:
            time.sleep(0.3)
            job = httpx.get(f"{self.url}/jobs/{job['id']}", timeout=30).json()
        assert job["status"] == "done", job
        return job

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


@pytest.fixture(scope="module")
def empty_api(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ApiServer]:
    server = ApiServer(tmp_path_factory.mktemp("empty"))
    yield server
    server.stop()


@pytest.fixture(scope="module")
def populated_api(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ApiServer]:
    """API with recent simulated data (ending now) so time ranges hit rows; no model yet."""
    server = ApiServer(tmp_path_factory.mktemp("populated"))
    server.fill()
    yield server
    server.stop()


@pytest.fixture(scope="module")
def trained_api(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ApiServer]:
    """API with data, a backtest report, a champion trained 2 days ago, and stored forecasts."""
    server = ApiServer(tmp_path_factory.mktemp("trained"))
    server.fill()
    server.job("backtest", task=TASK)
    server.job("train", task=TASK, holdback_days=2)
    server.job("forecast", task=TASK, backfill_steps=60)
    yield server
    server.stop()
