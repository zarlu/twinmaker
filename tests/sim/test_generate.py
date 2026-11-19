"""The config-driven simulator: determinism, contract validity, dependencies, faults."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from contract import IngestMessage
from sim.generate import SimConfigError, _dependency_order, generate, load_sim_config, to_messages

END = datetime(2026, 10, 20, 0, 0, tzinfo=UTC)
START = END - timedelta(days=10)


def test_same_seed_same_data_other_seed_other_data() -> None:
    cfg = load_sim_config()
    a, _ = generate(cfg, START, END, seed=1)
    b, _ = generate(cfg, START, END, seed=1)
    c, _ = generate(cfg, START, END, seed=2)
    assert a["node-intake"].equals(b["node-intake"])
    assert not a["node-intake"].equals(c["node-intake"])


def test_grid_values_stay_inside_clip_and_nodes_differ() -> None:
    cfg = load_sim_config()
    frames, _ = generate(cfg, START, END, seed=1, faults=False)
    frame = frames["node-intake"]
    assert len(frame) == 10 * 96 + 1 and frame.index.tz is not None
    assert frame["water_temp"].between(0, 45).all() and not frame.isna().any().any()
    gap = frames["node-downstream"]["water_temp"].mean() - frame["water_temp"].mean()
    assert 0 < gap < 2.5  # the configured +1.0 offset, plus each node's own noise


def test_dependent_channel_follows_its_driver_with_the_configured_lag() -> None:
    cfg = load_sim_config()
    frame = generate(cfg, START - timedelta(days=30), END, seed=1, faults=False)[0]["node-intake"]
    air, water = frame["air_temp"], frame["water_temp"]
    corr = {lag: np.corrcoef(air.shift(lag).dropna(), water.iloc[lag:])[0, 1] for lag in (0, 12)}
    assert corr[12] > corr[0] > 0  # 12 steps of 15 min = the 3 h lag in simulator.yaml


def test_messages_are_valid_contract_messages_marked_simulator() -> None:
    cfg = load_sim_config()
    frames, _ = generate(cfg, END - timedelta(hours=6), END, seed=1, faults=False)
    messages = to_messages(frames, batch_rows=8, sent_at=END)
    assert {m["node_id"] for m in messages} == {"node-intake", "node-downstream"}
    n_readings = 0
    for raw in messages:
        msg = IngestMessage.model_validate(raw)
        assert msg.source == "simulator" and len(msg.readings) <= 8 * 2
        n_readings += len(msg.readings)
    assert n_readings == 2 * 25 * 2  # 2 nodes x 25 time steps x 2 channels


def test_faults_are_logged_and_dropouts_are_not_sent() -> None:
    cfg = load_sim_config()
    clean, _ = generate(cfg, START - timedelta(days=30), END, seed=1, faults=False)
    faulty, log = generate(cfg, START - timedelta(days=30), END, seed=1, faults=True)
    kinds = {entry["fault"] for entry in log}
    assert kinds == {"spike", "dropout"}
    missing = int(faulty["node-intake"]["water_temp"].isna().sum())
    assert missing > 0
    sent = sum(len(m["readings"]) for m in to_messages(faulty))
    assert sent < sum(len(m["readings"]) for m in to_messages(clean))


def test_circular_dependency_is_reported() -> None:
    channels = {
        "a": {"base": 1, "depends_on": [{"channel": "b", "gain": 1}]},
        "b": {"base": 1, "depends_on": [{"channel": "a", "gain": 1}]},
    }
    with pytest.raises(SimConfigError, match="circular"):
        _dependency_order(channels)
