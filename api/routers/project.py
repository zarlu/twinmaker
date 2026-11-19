"""``GET /project``: what this twin is (name, time zone, key channels, forecast tasks)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from contract.project import load_nodes, load_project
from ml.config import list_tasks

router = APIRouter(tags=["project"])


@router.get("/project")
def get_project() -> dict[str, Any]:
    """The active project's ``project.yaml``, its configured nodes and its task names."""
    return {**load_project(), "nodes": load_nodes(), "tasks": list_tasks()}
