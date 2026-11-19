"""The dashboard's only way to get data: HTTP calls to the API (no SQL, no business logic)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

DEFAULT_API_URL = "http://localhost:8000"


class ApiError(Exception):
    """The API could not be reached or answered with an error."""


@dataclass(frozen=True)
class ApiClient:
    """Thin httpx wrapper. Every method raises ``ApiError`` instead of leaking exceptions."""

    base_url: str
    api_key: str = ""
    timeout: float = 15.0

    @classmethod
    def from_env(cls) -> ApiClient:
        return cls(
            base_url=os.environ.get("API_URL", DEFAULT_API_URL).rstrip("/"),
            api_key=os.environ.get("API_KEY", ""),
        )

    @staticmethod
    def _check(path: str, r: httpx.Response) -> Any:
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise ApiError(f"{path} -> HTTP {r.status_code}: {detail}")
        return r.json()

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        """Write call; needs ``API_KEY`` in the dashboard's environment."""
        try:
            r = httpx.post(
                f"{self.base_url}{path}",
                json=body,
                headers={"X-API-Key": self.api_key},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise ApiError(f"API not reachable at {self.base_url} ({type(exc).__name__})") from exc
        return self._check(path, r)

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        clean = {k: v for k, v in (params or {}).items() if v is not None}
        try:
            r = httpx.get(f"{self.base_url}{path}", params=clean, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise ApiError(f"API not reachable at {self.base_url} ({type(exc).__name__})") from exc
        return self._check(path, r)

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def nodes(self) -> list[dict[str, Any]]:
        return self._get("/nodes")

    def channels(self) -> list[dict[str, Any]]:
        return self._get("/channels")

    def project(self) -> dict[str, Any]:
        return self._get("/project")

    def observations(
        self,
        *,
        node_id: str | None = None,
        channel: str | None = None,
        kind: str | None = None,
        source: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        min_quality: str | None = None,
        resample: str | None = None,
        agg: str | None = None,
    ) -> dict[str, Any]:
        """Columnar observations exactly as ``GET /observations`` returns them."""
        return self._get(
            "/observations",
            {
                "node_id": node_id,
                "channel": channel,
                "kind": kind,
                "source": source,
                "start": _iso(start),
                "end": _iso(end),
                "min_quality": min_quality,
                "resample": resample,
                "agg": agg,
            },
        )

    def tasks(self) -> list[dict[str, Any]]:
        return self._get("/tasks")

    def models(self, task: str | None = None) -> list[dict[str, Any]]:
        return self._get("/models", {"task": task})

    def benchmarks(self, task: str) -> dict[str, Any]:
        return self._get(f"/benchmarks/{task}")

    def predictions(
        self,
        *,
        node_id: str | None = None,
        target: str | None = None,
        model_version: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, Any]:
        return self._get(
            "/predictions",
            {
                "node_id": node_id,
                "target": target,
                "model_version": model_version,
                "start": _iso(start),
                "end": _iso(end),
            },
        )

    def scorecard(self, task: str) -> dict[str, Any]:
        return self._get("/scorecard", {"task": task})

    def start_job(self, kind: str, **args: Any) -> dict[str, Any]:
        return self._post(f"/jobs/{kind}", {k: v for k, v in args.items() if v is not None})

    def job(self, job_id: str) -> dict[str, Any]:
        return self._get(f"/jobs/{job_id}")

    def jobs(self) -> list[dict[str, Any]]:
        return self._get("/jobs")


def _iso(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value else None
