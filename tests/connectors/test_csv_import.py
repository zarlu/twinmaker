"""CSV import: mapping validation, unit conversion, time zones, idempotent posting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from connectors.csv_import import (
    MappingError,
    main,
    post_readings,
    readings_from_frame,
    slug,
)
from sim.client import SimClient

WIDE = pd.DataFrame(
    {
        "when": ["2026-06-01 08:00", "2026-06-01 08:15", "not a time", "2026-06-01 08:45"],
        "site": ["Node Intake", "Node Intake", "Node Intake", "node-downstream"],
        "temp_f": ["68", "69.8", "70", ""],
        "pH": ["7.5", "abc", "7.6", "7.7"],
    }
)
MAPPING = {
    "format": "wide",
    "timestamp": {"column": "when", "timezone": "America/Chicago"},
    "node": {"column": "site"},
    "source": "logger",
    "columns": {
        "temp_f": {"channel": "water_temp", "scale": 5 / 9, "offset": -160 / 9},
        "pH": {"channel": "ph"},
    },
}


def test_slug_makes_valid_node_ids() -> None:
    assert slug(" Node Intake #2 ") == "node-intake-2"
    assert slug("---") == ""


def test_wide_file_converts_units_and_time_zone_and_reports_problems() -> None:
    grouped, problems = readings_from_frame(WIDE, MAPPING)
    assert set(grouped) == {"node-intake", "node-downstream"}
    temps = [r for r in grouped["node-intake"] if r["channel"] == "water_temp"]
    assert [r["ts"] for r in temps] == ["2026-06-01T13:00:00Z", "2026-06-01T13:15:00Z"]  # CDT+5
    assert temps[0]["value"] == pytest.approx(20.0) and temps[0]["unit"] == "degC"
    assert [r["channel"] for r in grouped["node-downstream"]] == ["ph"]  # blank temp skipped
    text = " ".join(problems)
    assert "1 rows skipped: timestamp" in text and "1 values in 'pH' skipped" in text


def test_long_file_and_fixed_node() -> None:
    df = pd.DataFrame(
        {
            "t": ["2026-06-01T08:00:00Z", "2026-06-01T08:00:00Z", "2026-06-01T08:00:00Z"],
            "param": ["Temp", "pH", "Salinity"],
            "val": ["21.5", "7.4", "3"],
        }
    )
    mapping = {
        "format": "long",
        "timestamp": {"column": "t"},
        "node": {"value": "node-bench"},
        "kind": "label",
        "long": {
            "channel_column": "param",
            "value_column": "val",
            "channels": {"Temp": {"channel": "water_temp"}, "pH": {"channel": "ph"}},
        },
    }
    grouped, problems = readings_from_frame(df, mapping)
    readings = grouped["node-bench"]
    assert {(r["channel"], r["value"], r["kind"]) for r in readings} == {
        ("water_temp", 21.5, "label"),
        ("ph", 7.4, "label"),
    }
    assert "Salinity" in " ".join(problems)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"timestamp": {"column": "nope"}}, "timestamp column"),
        ({"node": {}}, "`node` needs"),
        ({"node": {"value": "Bad Id"}}, "must match"),
        ({"columns": {"temp_f": {"channel": "unknown"}}}, "not in channels.yaml"),
        ({"columns": {"missing": {"channel": "ph"}}}, "not in the file"),
        ({"format": "tall"}, "`format`"),
        ({"source": "has space"}, "`source`"),
    ],
)
def test_bad_mappings_are_explained(change: dict, message: str) -> None:
    with pytest.raises(MappingError, match=message):
        readings_from_frame(WIDE, {**MAPPING, **change})


def test_posting_twice_stores_once(client: TestClient, auth: dict) -> None:
    grouped, _ = readings_from_frame(WIDE, MAPPING)
    first = post_readings(SimClient("", auth["X-API-Key"], http=client), grouped, "logger")
    assert first["accepted"] == 4 and first["rejected"] == 0
    again = post_readings(SimClient("", auth["X-API-Key"], http=client), grouped, "logger")
    assert again["accepted"] == 0 and again["duplicates"] == 4
    obs = client.get("/observations", params={"source": "logger"}).json()
    assert obs["count"] == 4 and set(obs["kind"]) == {"measurement"}
    assert "node-intake" in {n["node_id"] for n in client.get("/nodes").json()}


def test_command_line_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    csv = tmp_path / "x.csv"
    WIDE.to_csv(csv, index=False)
    mapping = tmp_path / "m.yaml"
    mapping.write_text(
        "format: wide\ntimestamp: {column: when}\nnode: {value: node-bench}\n"
        "columns: {pH: {channel: ph}}\n"
    )
    assert main([str(csv), "--mapping", str(mapping), "--dry-run"]) == 0
    assert "4 rows -> 2 readings for 1 node(s)" in capsys.readouterr().out
    mapping.write_text("format: wide\ntimestamp: {column: zzz}\nnode: {value: a}\ncolumns: {}\n")
    assert main([str(csv), "--mapping", str(mapping), "--dry-run"]) == 1
