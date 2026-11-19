"""``POST /jobs/{kind}``: start an ML job in the background; ``GET /jobs`` to follow it.

Jobs are explicit requests (API key required). State is kept in memory: it is lost when the API
restarts, but everything a job produces (reports, registry rows, predictions) is on disk.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from pydantic import BaseModel

from api.deps import require_api_key
from ml import jobs
from ml.config import TaskError

log = logging.getLogger(__name__)
router = APIRouter(tags=["jobs"])
MAX_KEPT = 50


class JobKind(StrEnum):
    backtest = "backtest"
    train = "train"
    forecast = "forecast"
    promote = "promote"


class JobRequest(BaseModel):
    """Arguments of a job. ``task`` for backtest/train/forecast, ``version`` for promote."""

    task: str | None = None
    version: str | None = None
    algorithm: str | None = None
    holdback_days: float = 0.0
    backfill_steps: int = 0


class JobStore:
    """In-memory job records, newest last."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def add(self, kind: str, args: dict[str, Any]) -> dict[str, Any]:
        job = {
            "id": uuid.uuid4().hex[:12],
            "kind": kind,
            "args": args,
            "status": "queued",
            "created_at": datetime.now(UTC),
            "finished_at": None,
            "result": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job["id"]] = job
            for old in list(self._jobs)[:-MAX_KEPT]:
                del self._jobs[old]
        return job

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(fields)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(j) for j in reversed(self._jobs.values())]


def _run(app: Any, job_id: str, kind: JobKind, req: JobRequest) -> None:
    store: JobStore = app.state.jobs
    store.update(job_id, status="running")
    try:
        with app.state.session_factory() as session:
            if kind == JobKind.backtest:
                result = jobs.backtest(session, req.task)
            elif kind == JobKind.train:
                result = jobs.train(session, req.task, req.algorithm, req.holdback_days)
            elif kind == JobKind.forecast:
                result = jobs.forecast(session, req.task, req.backfill_steps)
            else:
                result = jobs.promote(session, req.version)
        store.update(job_id, status="done", result=result, finished_at=datetime.now(UTC))
    except (jobs.JobError, TaskError) as exc:
        store.update(job_id, status="failed", error=str(exc), finished_at=datetime.now(UTC))
    except Exception as exc:  # noqa: BLE001 - a job must never take the API down
        log.exception("job %s (%s) crashed", job_id, kind)
        store.update(
            job_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=datetime.now(UTC),
        )


@router.post("/jobs/{kind}", dependencies=[Depends(require_api_key)], status_code=202)
def start_job(
    kind: JobKind, req: JobRequest, request: Request, background: BackgroundTasks
) -> dict[str, Any]:
    """Queue a job and return its record at once; poll ``GET /jobs/{id}`` for the outcome."""
    if kind == JobKind.promote and not req.version:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="promote needs `version`")
    if kind != JobKind.promote and not req.task:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"{kind} needs `task`")
    job = request.app.state.jobs.add(str(kind), req.model_dump(exclude_defaults=True))
    background.add_task(_run, request.app, job["id"], kind, req)
    return job


@router.get("/jobs")
def list_jobs(request: Request) -> list[dict[str, Any]]:
    """Recent jobs, newest first."""
    return request.app.state.jobs.all()


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> dict[str, Any]:
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"unknown job '{job_id}'")
    return job
