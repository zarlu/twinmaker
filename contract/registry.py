"""Channel registry loaded from the project's ``channels.yaml`` (the source of truth)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from contract.enums import Kind
from contract.project import ProjectError, project_dir, read_yaml


class ChannelSpec(BaseModel):
    """One row of the channel registry."""

    model_config = ConfigDict(frozen=True)

    channel: str
    unit: str
    default_kind: Kind
    valid_min: float
    valid_max: float
    label: str = ""  # human name for charts; defaults to the channel key
    max_step_15min: float | None = None
    stuck_after: int | None = None
    saturation_min: float | None = None  # values at or above are flagged `saturated`
    description: str = ""


def channels_path() -> Path:
    """``channels.yaml`` of the active project."""
    return project_dir() / "channels.yaml"


@lru_cache(maxsize=8)
def _load(path: Path) -> dict[str, ChannelSpec]:
    channels = read_yaml(path).get("channels") or {}
    if not channels:
        raise ProjectError(f"{path} defines no channels")
    out = {}
    for name, fields in channels.items():
        spec = ChannelSpec(channel=name, **fields)
        if spec.valid_min >= spec.valid_max:
            raise ProjectError(f"channel {name!r}: valid_min must be below valid_max")
        out[name] = spec
    return out


def load_registry(path: Path | None = None) -> dict[str, ChannelSpec]:
    """Load and validate the channel registry; cached per file."""
    return _load(path or channels_path())


def get_channel(name: str) -> ChannelSpec | None:
    """Return the spec for ``name`` or None when the channel is unknown."""
    return load_registry().get(name)
