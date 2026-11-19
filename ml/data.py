"""Read stored observations and put them on the task's regular time grid.

One frame per node, indexed by grid time (UTC, bucket start), one column per channel. A bucket's
value is the mean of the rows inside it that pass the task's quality filter. Short gaps are
forward-filled (the past carried forward, never the future); longer gaps stay missing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Observation
from ml.config import step_of

ALLOWED_QUALITY = {
    "good": ("good",),
    "suspect": ("good", "suspect"),
    "bad": ("good", "suspect", "bad"),
}


def task_channels(spec: dict[str, Any]) -> list[str]:
    """Target first, then every other feature channel."""
    target = spec["target"]["channel"]
    return [target, *[c for c in spec["features"]["channels"] if c != target]]


def read_rows(
    session: Session, spec: dict[str, Any], until: datetime | None = None
) -> pd.DataFrame:
    """Long frame (ts, node_id, channel, value, source) of the rows the task may use."""
    channels = task_channels(spec)
    target = spec["target"]
    stmt = select(
        Observation.ts,
        Observation.node_id,
        Observation.channel,
        Observation.value,
        Observation.kind,
        Observation.source,
    ).where(
        Observation.channel.in_(channels),
        Observation.quality.in_(ALLOWED_QUALITY[spec["data"]["min_quality"]]),
    )
    if target["nodes"] != "all":
        stmt = stmt.where(Observation.node_id.in_([str(n) for n in target["nodes"]]))
    if spec["data"]["exclude_sources"]:
        stmt = stmt.where(Observation.source.notin_(spec["data"]["exclude_sources"]))
    if until is not None:
        stmt = stmt.where(Observation.ts <= until)
    rows = session.execute(stmt).all()
    df = pd.DataFrame(rows, columns=["ts", "node_id", "channel", "value", "kind", "source"])
    if df.empty:
        return df
    # The target uses its configured kind; feature channels never use labels.
    is_target = df["channel"] == target["channel"]
    keep = (is_target & (df["kind"] == target["kind"])) | (~is_target & (df["kind"] != "label"))
    df = df[keep].drop(columns="kind")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def to_grids(
    rows: pd.DataFrame, spec: dict[str, Any], now: datetime | None = None
) -> dict[str, pd.DataFrame]:
    """Regular grid per node. Buckets that are still filling at ``now`` are left out."""
    step = step_of(spec)
    channels = task_channels(spec)
    now_ts = pd.Timestamp(now or datetime.now(UTC))
    grids: dict[str, pd.DataFrame] = {}
    if rows.empty:
        return grids
    rows = rows.assign(bucket=rows["ts"].dt.floor(step))
    rows = rows[rows["bucket"] + step <= now_ts]
    for node, part in rows.groupby("node_id", sort=True):
        wide = part.pivot_table(index="bucket", columns="channel", values="value", aggfunc="mean")
        if spec["target"]["channel"] not in wide:
            continue
        full = pd.date_range(wide.index.min(), wide.index.max(), freq=step, tz="UTC")
        wide = wide.reindex(full).reindex(columns=channels)
        limit = int(spec["data"]["max_gap_steps"])
        if limit > 0:
            wide = wide.ffill(limit=limit)
        wide.index.name = "t"
        grids[str(node)] = wide
    return grids


def load_grids(
    session: Session,
    spec: dict[str, Any],
    until: datetime | None = None,
    now: datetime | None = None,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """(grids per node, distinct sources of the rows used)."""
    rows = read_rows(session, spec, until)
    sources = sorted(rows["source"].unique()) if not rows.empty else []
    return to_grids(rows, spec, now), sources
