"""ML read endpoints: tasks, model registry, backtest benchmarks, predictions, scorecard.

Nothing here fits a model. Benchmarks are the stored result of the latest backtest; predictions
are rows the forecast job stored; the scorecard joins them to outcomes measured since.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from api.deps import SessionDep
from db.models import Model, Prediction
from ml import jobs, report
from ml.config import TaskError, list_tasks, load_task

router = APIRouter(tags=["ml"])


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _json(text: str | None) -> Any:
    return json.loads(text) if text else None


def _task(name: str) -> dict[str, Any]:
    try:
        return load_task(name)
    except TaskError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/tasks")
def get_tasks() -> list[dict[str, Any]]:
    """Every forecast task of the project (invalid task files are listed with their error)."""
    out = []
    for name in list_tasks():
        try:
            spec = load_task(name)
        except TaskError as exc:
            out.append({"name": name, "error": str(exc)})
            continue
        out.append(
            {
                "name": name,
                "description": spec["description"],
                "target": spec["target"]["channel"],
                "unit": spec["target"]["unit"],
                "nodes": spec["target"]["nodes"],
                "resolution": spec["resolution"],
                "horizon_steps": spec["horizon_steps"],
                "horizon_h": spec["horizon_h"],
                "event": spec["event"],
                "models": list(spec["models"]),
                "has_report": report.latest_folder(name) is not None,
            }
        )
    return out


@router.get("/models")
def list_models(session: SessionDep, task: str | None = None) -> list[dict[str, Any]]:
    """Model registry rows, champions first, newest first."""
    stmt = select(Model).order_by(Model.status != "champion", Model.created_at.desc())
    if task:
        stmt = stmt.where(Model.task == task)
    return [
        {
            "version": m.version,
            "task": m.task,
            "algorithm": m.algorithm,
            "target": m.target,
            "horizon_h": m.horizon_h,
            "features": _json(m.features_json),
            "train_start": _aware(m.train_start),
            "train_end": _aware(m.train_end),
            "train_sources": m.train_sources,
            "n_train": m.n_train,
            "metrics": _json(m.metrics_json),
            "baseline_metrics": _json(m.baseline_metrics_json),
            "status": m.status,
            "notes": m.notes,
            "created_at": _aware(m.created_at),
        }
        for m in session.scalars(stmt).all()
    ]


@router.get("/benchmarks/{task}")
def get_benchmarks(task: str) -> dict[str, Any]:
    """The latest backtest result of a task: every baseline and model, with verdicts."""
    _task(task)
    result = report.latest(task)
    if result is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail=f"no backtest report for task '{task}' yet"
        )
    return result


@router.get("/predictions")
def get_predictions(
    session: SessionDep,
    node_id: str | None = None,
    target: str | None = None,
    model_version: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> dict[str, Any]:
    """Stored predictions as columnar JSON, ordered by ``valid_at``."""
    stmt = select(Prediction).order_by(Prediction.valid_at, Prediction.node_id)
    if node_id:
        stmt = stmt.where(Prediction.node_id == node_id)
    if target:
        stmt = stmt.where(Prediction.target == target)
    if model_version:
        stmt = stmt.where(Prediction.model_version == model_version)
    if start:
        stmt = stmt.where(Prediction.valid_at >= _aware(start))
    if end:
        stmt = stmt.where(Prediction.valid_at <= _aware(end))
    rows = session.scalars(stmt).all()
    cols = ("node_id", "target", "horizon_h", "yhat", "lo", "hi", "p_event", "model_version")
    out: dict[str, Any] = {"count": len(rows)}
    out["issued_at"] = [_aware(r.issued_at) for r in rows]
    out["valid_at"] = [_aware(r.valid_at) for r in rows]
    for c in cols:
        out[c] = [getattr(r, c) for r in rows]
    return out


@router.get("/scorecard")
def get_scorecard(session: SessionDep, task: str) -> dict[str, Any]:
    """Realized error of the champion and of persistence on outcomes measured since."""
    _task(task)
    return jobs.scorecard(session, task)
