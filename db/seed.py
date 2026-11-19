"""Seed the ``channels`` and ``nodes`` tables from the project's YAML (idempotent upserts)."""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from contract.project import load_nodes
from contract.registry import load_registry
from db.models import Channel, Node

log = logging.getLogger(__name__)


def seed_channels(session: Session) -> int:
    """Upsert every channel of ``channels.yaml``. Returns the number of channels."""
    registry = load_registry()
    for spec in registry.values():
        row = session.get(Channel, spec.channel)
        if row is None:
            row = Channel(channel=spec.channel)
            session.add(row)
        row.unit = spec.unit
        row.description = spec.description
        row.default_kind = str(spec.default_kind)
        row.valid_min = spec.valid_min
        row.valid_max = spec.valid_max
        row.max_step_15min = spec.max_step_15min
    session.commit()
    log.info("seeded %d channels", len(registry))
    return len(registry)


def seed_nodes(session: Session) -> int:
    """Upsert the nodes of ``nodes.yaml``. Returns the number of nodes."""
    nodes = load_nodes()
    for i, entry in enumerate(nodes):
        row = session.get(Node, entry["node_id"])
        if row is None:
            row = Node(node_id=entry["node_id"])
            session.add(row)
        row.name = entry.get("name")
        row.role = entry.get("role") or "node"
        row.lat = entry.get("lat")
        row.lon = entry.get("lon")
        row.position = entry.get("position", float(i))
        row.notes = entry.get("notes")
    session.commit()
    log.info("seeded %d nodes", len(nodes))
    return len(nodes)


def seed_all(session: Session) -> None:
    """Seed channels and nodes."""
    seed_channels(session)
    seed_nodes(session)
