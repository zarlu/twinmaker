"""Where the project configuration lives and how to load it.

A *project* is a folder of YAML files that describes one digital twin: ``project.yaml`` (name,
display time zone), ``channels.yaml`` (what is measured), ``nodes.yaml`` (where), ``tasks/``
(what to forecast), ``simulator.yaml`` and ``imports/`` (CSV mappings). The folder is chosen by
the ``PROJECT_DIR`` environment variable (default ``project``, relative to the repository root).
Nothing about a specific site or sensor is written in Python.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROJECT_DIR = "project"


class ProjectError(Exception):
    """The project folder or one of its files is missing or invalid."""


def project_dir() -> Path:
    """The active project folder (``PROJECT_DIR``; relative paths are under the repo root)."""
    path = Path(os.environ.get("PROJECT_DIR") or DEFAULT_PROJECT_DIR)
    path = path if path.is_absolute() else REPO_ROOT / path
    if not path.is_dir():
        raise ProjectError(
            f"project folder {path} does not exist. Create one with "
            "`make init EXAMPLE=water_quality` or point PROJECT_DIR at an examples/ folder."
        )
    return path


def read_yaml(path: Path) -> Any:
    """Parse one YAML file with a clear error when it is missing."""
    if not path.is_file():
        raise ProjectError(f"{path} not found")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_project() -> dict[str, Any]:
    """``project.yaml`` with defaults filled in."""
    folder = project_dir()
    raw = read_yaml(folder / "project.yaml") if (folder / "project.yaml").is_file() else {}
    return {
        "name": raw.get("name") or folder.name,
        "description": raw.get("description") or "",
        "timezone": raw.get("timezone") or "UTC",
        "key_channels": list(raw.get("key_channels") or []),
        "project_dir": str(folder),
    }


def load_nodes() -> list[dict[str, Any]]:
    """The node list of ``nodes.yaml`` (empty when the file is absent)."""
    path = project_dir() / "nodes.yaml"
    if not path.is_file():
        return []
    return list(read_yaml(path).get("nodes") or [])
