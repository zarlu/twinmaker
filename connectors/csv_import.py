"""Import historical CSV files through ``/ingest``.

    python -m connectors.csv_import data.csv --mapping project/imports/logger.yaml

A mapping file says how the CSV's columns become contract readings, so no code changes are
needed for a new file layout. Two layouts are supported:

  wide  one row per time step, one column per channel
  long  one row per value, with a column naming the channel

Mapping keys:

    format: wide                      # wide | long
    timestamp:
      column: time                    # required
      timezone: UTC                   # zone of timestamps that carry no offset (default UTC)
      format: null                    # optional strptime pattern, e.g. "%d/%m/%Y %H:%M"
    node:
      column: station                 # a column holding the node id ...
      value: node-a                   # ... or one fixed node id for the whole file
    source: csv-import                # stored as the readings' source
    kind: null                        # measurement | label | external; default per channel
    columns:                          # wide: CSV column -> channel (value * scale + offset)
      temp_c: {channel: water_temp}
      cond_ms: {channel: conductivity, scale: 1000}
    long:                             # long layout only
      channel_column: parameter
      value_column: value
      channels: {Temp: {channel: water_temp}}

Values go through the same validation and QC as live data, and importing a file twice changes
nothing (duplicates are reported, not stored).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from contract.enums import SOURCE_PATTERN
from contract.models import NODE_ID_PATTERN
from contract.project import read_yaml
from contract.registry import load_registry
from sim.client import SimClient

BATCH = 2000
DEFAULT_SOURCE = "csv-import"


class MappingError(Exception):
    """The mapping does not fit the file or the channel registry."""


def slug(value: Any) -> str:
    """A node id from free text: lowercase, runs of other characters become '-'."""
    text = re.sub(r"[^a-z0-9_-]+", "-", str(value).strip().lower()).strip("-")
    return text[:64]


def validate_mapping(mapping: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    """Check a mapping against the CSV header and the registry; returns it with defaults."""
    registry = load_registry()
    m = dict(mapping or {})
    m.setdefault("format", "wide")
    m.setdefault("source", DEFAULT_SOURCE)
    if m["format"] not in ("wide", "long"):
        raise MappingError("`format` must be 'wide' or 'long'")
    if not re.fullmatch(SOURCE_PATTERN, str(m["source"])):
        raise MappingError(f"`source` '{m['source']}' must match {SOURCE_PATTERN}")
    if m.get("kind") not in (None, "measurement", "label", "external"):
        raise MappingError("`kind` must be measurement, label or external")
    ts = m.get("timestamp") or {}
    if ts.get("column") not in columns:
        raise MappingError(f"timestamp column '{ts.get('column')}' is not in the file: {columns}")
    node = m.get("node") or {}
    if not node.get("column") and not node.get("value"):
        raise MappingError("`node` needs `column` or `value`")
    if node.get("column") and node["column"] not in columns:
        raise MappingError(f"node column '{node['column']}' is not in the file")
    if node.get("value") and not re.fullmatch(NODE_ID_PATTERN, str(node["value"])):
        raise MappingError(f"node value '{node['value']}' must match {NODE_ID_PATTERN}")

    def check_channels(entries: dict[str, Any], where: str) -> None:
        if not entries:
            raise MappingError(f"`{where}` maps no channel")
        for key, entry in entries.items():
            if entry.get("channel") not in registry:
                raise MappingError(
                    f"`{where}.{key}`: channel '{entry.get('channel')}' is not in channels.yaml"
                )

    if m["format"] == "wide":
        check_channels(m.get("columns") or {}, "columns")
        missing = [c for c in m["columns"] if c not in columns]
        if missing:
            raise MappingError(f"mapped columns not in the file: {missing}")
    else:
        long = m.get("long") or {}
        for key in ("channel_column", "value_column"):
            if long.get(key) not in columns:
                raise MappingError(f"`long.{key}` '{long.get(key)}' is not in the file")
        check_channels(long.get("channels") or {}, "long.channels")
    return m


def parse_timestamps(series: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """UTC timestamps; values without an offset are read in ``timezone``. Bad values -> NaT."""
    parsed = pd.to_datetime(series, format=cfg.get("format") or None, errors="coerce")
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(
            cfg.get("timezone") or "UTC", ambiguous="NaT", nonexistent="NaT"
        )
    return parsed.dt.tz_convert("UTC")


def readings_from_frame(
    df: pd.DataFrame, mapping: dict[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """(readings per node id, problems). Rows with a bad timestamp or node are skipped."""
    m = validate_mapping(mapping, list(df.columns))
    registry = load_registry()
    problems: list[str] = []
    ts = parse_timestamps(df[m["timestamp"]["column"]], m["timestamp"])
    bad_ts = int(ts.isna().sum())
    if bad_ts:
        problems.append(f"{bad_ts} rows skipped: timestamp could not be read")
    node_cfg = m["node"]
    if node_cfg.get("column"):
        nodes = df[node_cfg["column"]].map(slug)
        bad_node = int((nodes == "").sum())
        if bad_node:
            problems.append(f"{bad_node} rows skipped: empty node id")
    else:
        nodes = pd.Series(str(node_cfg["value"]), index=df.index)
    keep = ts.notna() & (nodes != "")
    stamps = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def emit(mask: pd.Series, values: pd.Series, entry: dict[str, Any], label: str) -> None:
        spec = registry[entry["channel"]]
        numeric = pd.to_numeric(values, errors="coerce")
        blank = values.isna() | (values.astype(str).str.strip() == "")
        not_number = int((numeric.isna() & ~blank & mask).sum())
        if not_number:
            problems.append(f"{not_number} values in '{label}' skipped: not a number")
        numeric = numeric * float(entry.get("scale", 1)) + float(entry.get("offset", 0))
        ok = mask & numeric.notna()
        for node, stamp, value in zip(nodes[ok], stamps[ok], numeric[ok], strict=True):
            reading = {"ts": stamp, "channel": spec.channel, "value": float(value),
                       "unit": spec.unit}  # fmt: skip
            if m.get("kind"):
                reading["kind"] = m["kind"]
            out[node].append(reading)

    if m["format"] == "wide":
        for column, entry in m["columns"].items():
            emit(keep, df[column], entry, column)
    else:
        long = m["long"]
        names = df[long["channel_column"]].astype(str).str.strip()
        unmapped = sorted(set(names[keep]) - set(long["channels"]))
        if unmapped:
            problems.append(f"rows skipped for unmapped channel names: {unmapped[:10]}")
        for name, entry in long["channels"].items():
            emit(keep & (names == str(name)), df[long["value_column"]], entry, str(name))
    return dict(out), problems


def post_readings(
    client: Any, grouped: dict[str, list[dict[str, Any]]], source: str
) -> dict[str, Any]:
    """POST readings in batches. ``client`` needs ``post(endpoint, body)`` and ``stats``."""
    now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    for node, readings in grouped.items():
        for start in range(0, len(readings), BATCH):
            client.post(
                "/ingest",
                {
                    "schema_version": 1,
                    "node_id": node,
                    "source": source,
                    "sent_at": now,
                    "firmware": "csv-import",
                    "readings": readings[start : start + BATCH],
                },
            )
    s = client.stats
    return {
        "accepted": s.accepted,
        "duplicates": s.duplicates,
        "rejected": s.rejected,
        "failed_requests": s.failed_requests,
        "rejection_examples": dict(s.rejection_reasons.most_common(5)),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m connectors.csv_import")
    p.add_argument("csv", type=Path)
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true", help="parse and count, post nothing")
    p.add_argument("--api-url", default=os.environ.get("API_URL", "http://localhost:8000"))
    p.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    args = p.parse_args(argv)
    try:
        df = pd.read_csv(args.csv, dtype=str, encoding="utf-8-sig")
        mapping = read_yaml(args.mapping)
        grouped, problems = readings_from_frame(df, mapping)
    except (MappingError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    total = sum(len(r) for r in grouped.values())
    print(f"{args.csv}: {len(df)} rows -> {total} readings for {len(grouped)} node(s)")
    for problem in problems:
        print(f"  note: {problem}")
    if args.dry_run or total == 0:
        return 0
    if not args.api_key:
        print("error: API_KEY is not set (see .env) and --api-key was not given", file=sys.stderr)
        return 1
    result = post_readings(
        SimClient(args.api_url, args.api_key), grouped, mapping.get("source", DEFAULT_SOURCE)
    )
    print(
        f"accepted {result['accepted']}, duplicates {result['duplicates']}, "
        f"rejected {result['rejected']}, failed requests {result['failed_requests']}"
    )
    for reason, count in result["rejection_examples"].items():
        print(f"  rejected ({count}x): {reason}")
    return 1 if result["failed_requests"] else 0


if __name__ == "__main__":
    sys.exit(main())
