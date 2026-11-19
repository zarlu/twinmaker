"""The JSON examples in contract/examples/ are checked against the water-quality example."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from contract import IngestMessage
from contract.project import REPO_ROOT

EXAMPLES = sorted((REPO_ROOT / "contract" / "examples").glob("*.json"))
REASONS = {
    "invalid_naive_timestamp": "naive timestamp",
    "invalid_wrong_unit": "must be 'degC'",
    "invalid_unknown_channel": "unknown channel 'salinity'",
    "invalid_duplicate_key_in_batch": "repeats the key",
    "invalid_bad_node_id": "node_id",
}


@pytest.fixture(autouse=True)
def _water_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROJECT_DIR", "examples/water_quality")


def test_examples_exist() -> None:
    names = [p.stem for p in EXAMPLES]
    assert sum(n.startswith("valid_") for n in names) == 2
    assert {n for n in names if n.startswith("invalid_")} == set(REASONS)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_example(path: Path) -> None:
    raw = json.loads(path.read_text())
    if path.stem.startswith("valid_"):
        msg = IngestMessage.model_validate(raw)
        assert all(r.ts.utcoffset().total_seconds() == 0 for r in msg.readings)
    else:
        with pytest.raises(ValidationError, match=REASONS[path.stem]):
            IngestMessage.model_validate(raw)


def test_offsets_are_normalized_and_per_reading_fields_override() -> None:
    raw = json.loads((EXAMPLES[0].parent / "valid_buffered_upload_with_offset.json").read_text())
    msg = IngestMessage.model_validate(raw)
    assert msg.readings[0].ts.isoformat() == "2026-10-10T12:00:00+00:00"
    assert [r.source for r in msg.readings] == ["sensor", "sensor", "weather-api", "lab"]
    assert [str(r.kind) for r in msg.readings] == [
        "measurement",
        "measurement",
        "external",
        "label",
    ]
