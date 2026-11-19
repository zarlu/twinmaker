"""A database filled with simulated data for the ML tests (in-process, no HTTP)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from api.ingest import ingest_readings
from db.seed import seed_all
from sim.generate import generate, load_sim_config, to_messages

NOW = datetime(2026, 10, 20, 0, 0, tzinfo=UTC)
TASK = "temp_3h"


def fill(session: Session, days: float = 20.0, end: datetime = NOW) -> None:
    seed_all(session)
    frames, _ = generate(load_sim_config(), end - timedelta(days=days), end, seed=1)
    for message in to_messages(frames, 96 * 5, sent_at=end):
        ingest_readings(session, message["node_id"], message["source"], message["readings"])


@pytest.fixture
def filled(session: Session) -> Session:
    """Session on a database holding 20 days of two simulated nodes ending at NOW."""
    fill(session)
    return session
