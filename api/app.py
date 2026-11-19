"""App factory. ``uvicorn api.app:create_app --factory`` or ``create_app(settings)`` in tests."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from api import __version__
from api.routers import (
    channels,
    health,
    ingest,
    jobs,
    labels,
    ml,
    nodes,
    observations,
    project,
)
from api.settings import Settings
from contract.project import load_project
from db.engine import init_db, make_engine, make_session_factory
from db.seed import seed_all

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app, create tables and seed channels and nodes.

    ``settings`` defaults to the environment, so tests can point the app at a temporary database.
    """
    settings = settings or Settings.from_env()
    info = load_project()
    engine = make_engine(settings.db_url)
    init_db(engine)
    session_factory = make_session_factory(engine)
    with session_factory() as session:
        seed_all(session)

    app = FastAPI(
        title=f"{info['name']} - TwinMaker API",
        version=__version__,
        description=(
            "Ingest, read and forecast API of a TwinMaker digital twin. "
            "UTC everywhere; simulated data is labeled source='simulator'."
        ),
    )
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.jobs = jobs.JobStore()
    for r in (health, project, ingest, nodes, channels, observations, labels, ml, jobs):
        app.include_router(r.router)
    log.info("app created (db=%s)", settings.db_url)
    return app
