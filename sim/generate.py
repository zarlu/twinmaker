"""Generate invented sensor series from ``<project>/simulator.yaml``.

Each channel is a sum of simple, documented parts:

    value(t) = base + node offset
             + daily.amplitude  * cos(2*pi * (local hour - daily.peak_hour) / 24)
             + yearly.amplitude * cos(2*pi * (day of year - yearly.peak_day) / 365.25)
             + trend_per_day * days since the start of the run
             + slow noise (AR(1) with standard deviation noise.sd and memory noise.memory)
             + sum over depends_on: gain * (other channel at t - lag_hours, minus its base)
             + events (a ramped bump of `delta` for `duration_hours`)
             + white measurement noise (jitter)

then clipped to ``clip`` and rounded. ``depends_on`` is what gives a forecasting model something
real to learn: one channel leading another. Faults (spike, stuck, dropout) are applied last.

These are arbitrary shapes chosen to exercise the pipeline. They are not a model of any real
system, and every row is sent with ``source='simulator'``.
"""

from __future__ import annotations

import zlib
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from contract.project import ProjectError, load_nodes, project_dir, read_yaml
from contract.registry import load_registry


class SimConfigError(ProjectError):
    """simulator.yaml is invalid."""


def load_sim_config() -> dict[str, Any]:
    """``simulator.yaml`` of the active project, validated against the channel registry."""
    cfg = read_yaml(project_dir() / "simulator.yaml")
    registry = load_registry()
    channels = cfg.get("channels") or {}
    if not channels:
        raise SimConfigError("simulator.yaml: `channels` is empty")
    for name, c in channels.items():
        if name not in registry:
            raise SimConfigError(f"simulator.yaml: channel '{name}' is not in channels.yaml")
        if "base" not in c:
            raise SimConfigError(f"simulator.yaml: channel '{name}' needs `base`")
        for dep in c.get("depends_on") or []:
            if dep["channel"] not in channels:
                raise SimConfigError(
                    f"simulator.yaml: '{name}' depends on '{dep['channel']}', "
                    "which is not simulated"
                )
    cfg.setdefault("step_minutes", 15)
    cfg.setdefault("utc_offset_hours", 0)
    cfg.setdefault("events", [])
    cfg.setdefault("faults", {})
    node_cfg = cfg.get("nodes") or {n["node_id"]: {} for n in load_nodes()}
    if not node_cfg:
        raise SimConfigError("simulator.yaml: no `nodes`, and nodes.yaml lists none either")
    cfg["nodes"] = {str(k): (v or {}) for k, v in node_cfg.items()}
    cfg["order"] = _dependency_order(channels)
    return cfg


def _dependency_order(channels: dict[str, Any]) -> list[str]:
    order: list[str] = []
    state: dict[str, int] = {}

    def visit(name: str) -> None:
        if state.get(name) == 2:
            return
        if state.get(name) == 1:
            raise SimConfigError(f"simulator.yaml: circular depends_on involving '{name}'")
        state[name] = 1
        for dep in channels[name].get("depends_on") or []:
            visit(dep["channel"])
        state[name] = 2
        order.append(name)

    for name in channels:
        visit(name)
    return order


def _rng(seed: int, *parts: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32("/".join(parts).encode())])


def _ar1(rng: np.random.Generator, n: int, sd: float, memory: float) -> np.ndarray:
    if sd <= 0:
        return np.zeros(n)
    shocks = rng.normal(0.0, sd * np.sqrt(max(1 - memory**2, 1e-9)), n)
    out = np.empty(n)
    out[0] = rng.normal(0.0, sd)
    for i in range(1, n):
        out[i] = memory * out[i - 1] + shocks[i]
    return out


def _event_bump(index: pd.DatetimeIndex, event: dict[str, Any], step_h: float) -> np.ndarray:
    n = len(index)
    day = float(event["at_day"])
    start_h = day * 24 if day >= 0 else n * step_h + day * 24
    hours = np.arange(n) * step_h - start_h
    duration = float(event.get("duration_hours", 24))
    ramp = max(float(event.get("ramp_hours", 0)), 1e-9)
    up = np.clip(hours / ramp, 0, 1)
    down = np.clip((duration - hours) / ramp, 0, 1)
    return float(event["delta"]) * np.minimum(up, down)


def generate_node(
    cfg: dict[str, Any], node: str, index: pd.DatetimeIndex, seed: int
) -> pd.DataFrame:
    """Clean (fault-free) series of one node: one column per simulated channel."""
    n = len(index)
    step_h = cfg["step_minutes"] / 60.0
    node_cfg = cfg["nodes"].get(node, {})
    local = index + pd.Timedelta(hours=float(cfg["utc_offset_hours"]))
    hour = (local.hour + local.minute / 60.0).to_numpy()
    doy = local.day_of_year.to_numpy()
    days = np.arange(n) * step_h / 24.0
    out: dict[str, np.ndarray] = {}
    for name in cfg["order"]:
        c = cfg["channels"][name]
        rng = _rng(seed, node, name)
        value = np.full(n, float(c["base"]) + float((node_cfg.get("offsets") or {}).get(name, 0)))
        daily, yearly = c.get("daily") or {}, c.get("yearly") or {}
        if daily:
            peak = float(daily.get("peak_hour", 15))
            value += float(daily["amplitude"]) * np.cos(2 * np.pi * (hour - peak) / 24.0)
        if yearly:
            peak = float(yearly.get("peak_day", 200))
            value += float(yearly["amplitude"]) * np.cos(2 * np.pi * (doy - peak) / 365.25)
        value += float(c.get("trend_per_day", 0)) * days
        noise = c.get("noise") or {}
        value += _ar1(rng, n, float(noise.get("sd", 0)), float(noise.get("memory", 0.9)))
        for dep in c.get("depends_on") or []:
            other = out[dep["channel"]] - float(cfg["channels"][dep["channel"]]["base"])
            lag = int(round(float(dep.get("lag_hours", 0)) / step_h))
            shifted = np.concatenate([np.full(lag, other[0]), other[: n - lag]]) if lag else other
            value += float(dep["gain"]) * shifted
        for event in cfg["events"]:
            if event["channel"] == name and node in (event.get("nodes") or [node]):
                value += _event_bump(index, event, step_h)
        out[name] = value
    # Measurement noise, clipping and rounding come last so dependencies use the clean signal.
    final = {}
    for name in cfg["order"]:
        c = cfg["channels"][name]
        value = out[name] + _rng(seed, node, name, "jitter").normal(0, float(c.get("jitter", 0)), n)
        if c.get("clip"):
            value = np.clip(value, float(c["clip"][0]), float(c["clip"][1]))
        final[name] = np.round(value, int(c.get("decimals", 3)))
    return pd.DataFrame(final, index=index)


def apply_faults(
    cfg: dict[str, Any], node: str, frame: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Inject faults; returns the faulty frame (NaN = not sent) and a log of what was injected."""
    faults = cfg["faults"]
    frame = frame.copy()
    log: list[dict[str, Any]] = []
    n = len(frame)
    step_h = cfg["step_minutes"] / 60.0
    days = n * step_h / 24.0
    rng = _rng(seed, node, "faults")

    def channels_of(spec: dict[str, Any]) -> list[str]:
        return [c for c in (spec.get("channels") or list(frame.columns)) if c in frame]

    spike = faults.get("spike")
    if spike:
        for ch in channels_of(spike):
            for i in rng.integers(0, n, rng.poisson(float(spike.get("per_day", 0.1)) * days)):
                frame.iloc[i, frame.columns.get_loc(ch)] += float(spike["size"]) * rng.choice(
                    [-1, 1]
                )
                log.append({"fault": "spike", "node": node, "channel": ch, "ts": frame.index[i]})
    stuck = faults.get("stuck")
    if stuck:
        length = max(2, int(round(float(stuck.get("duration_hours", 3)) / step_h)))
        for ch in channels_of(stuck):
            for i in rng.integers(0, n, rng.poisson(float(stuck.get("per_week", 0.5)) * days / 7)):
                col = frame.columns.get_loc(ch)
                frame.iloc[i : i + length, col] = frame.iloc[i, col]
                log.append({"fault": "stuck", "node": node, "channel": ch, "ts": frame.index[i]})
    dropout = faults.get("dropout")
    if dropout:
        length = max(1, int(round(float(dropout.get("duration_hours", 2)) / step_h)))
        for i in rng.integers(0, n, rng.poisson(float(dropout.get("per_week", 0.5)) * days / 7)):
            frame.iloc[i : i + length, :] = np.nan
            log.append({"fault": "dropout", "node": node, "channel": "*", "ts": frame.index[i]})
    return frame, log


def time_index(start: datetime, end: datetime, step_minutes: int) -> pd.DatetimeIndex:
    step = pd.Timedelta(minutes=step_minutes)
    first = pd.Timestamp(start).ceil(step)
    return pd.date_range(first, pd.Timestamp(end).floor(step), freq=step, tz="UTC")


def generate(
    cfg: dict[str, Any], start: datetime, end: datetime, seed: int = 1, faults: bool = True
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """Series of every node between ``start`` and ``end`` (UTC), plus the fault log."""
    index = time_index(start, end, int(cfg["step_minutes"]))
    frames: dict[str, pd.DataFrame] = {}
    log: list[dict[str, Any]] = []
    for node in cfg["nodes"]:
        frame = generate_node(cfg, node, index, seed)
        if faults and cfg["faults"]:
            frame, node_log = apply_faults(cfg, node, frame, seed)
            log += node_log
        frames[node] = frame
    return frames, log


def to_messages(
    frames: dict[str, pd.DataFrame], batch_rows: int = 24, sent_at: datetime | None = None
) -> list[dict[str, Any]]:
    """Contract v1 messages: per node, ``batch_rows`` time steps per message."""
    registry = load_registry()
    messages = []
    for node, frame in frames.items():
        for start in range(0, len(frame), batch_rows):
            chunk = frame.iloc[start : start + batch_rows]
            readings = []
            for ts, row in chunk.iterrows():
                stamp = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
                for channel, value in row.items():
                    if pd.isna(value):
                        continue
                    readings.append(
                        {
                            "ts": stamp,
                            "channel": channel,
                            "value": float(value),
                            "unit": registry[channel].unit,
                        }
                    )
            if readings:
                when = sent_at or chunk.index[-1].to_pydatetime()
                messages.append(
                    {
                        "schema_version": 1,
                        "node_id": node,
                        "source": "simulator",
                        "sent_at": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "firmware": "simulator",
                        "readings": readings,
                    }
                )
    return messages
