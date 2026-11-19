"""Pure helpers for the dashboard (no Streamlit): time zones, quality encoding, parameters.

Everything here is unit-tested; the views only wire these functions to widgets.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

DEFAULT_TZ = "UTC"
BANNER_TEXT = "SIMULATED DATA - pipeline validation only, not the real system"

TIME_RANGES: dict[str, timedelta | None] = {
    "last 6 h": timedelta(hours=6),
    "last 24 h": timedelta(hours=24),
    "last 7 d": timedelta(days=7),
    "last 30 d": timedelta(days=30),
    "custom": None,
}

RESAMPLE_OPTIONS: dict[str, str | None] = {
    "raw": None,
    "15 min": "15min",
    "1 h": "1h",
    "1 day": "1D",
}

QUALITY_STYLE: dict[str, dict[str, Any]] = {
    "good": {"mode": "lines", "line": {"width": 1.6}},
    "suspect": {
        "mode": "markers",
        "marker": {"symbol": "circle-open", "color": "orange", "size": 8},
    },
    "bad": {"mode": "markers", "marker": {"symbol": "x", "color": "red", "size": 8}},
}


def tz_options(project_tz: str | None) -> tuple[str, ...]:
    """Display zones offered in the sidebar: the project's zone first, then UTC."""
    zones = [project_tz or DEFAULT_TZ, "UTC"]
    return tuple(dict.fromkeys(zones))


def time_range_bounds(
    choice: str, now: datetime | None = None, custom: tuple[datetime, datetime] | None = None
) -> tuple[datetime, datetime]:
    """UTC (start, end) for a picker choice. ``custom`` is used when ``choice == 'custom'``."""
    now = now or datetime.now(UTC)
    if choice == "custom":
        if custom is None:
            raise ValueError("custom range needs explicit bounds")
        start, end = custom
        return _utc(start), _utc(end)
    delta = TIME_RANGES[choice]
    if delta is None:
        raise ValueError(f"unknown time range {choice!r}")
    return now - delta, now


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def to_display_tz(ts: pd.Series | list[Any], tz: str = DEFAULT_TZ) -> pd.Series:
    """Convert UTC timestamps to the display zone (data stays UTC; this is for display only)."""
    s = pd.to_datetime(pd.Series(ts), utc=True)
    return s.dt.tz_convert(ZoneInfo(tz))


def resample_param(label: str) -> str | None:
    """Map a dropdown label to the API ``resample`` value (None = raw)."""
    if label not in RESAMPLE_OPTIONS:
        raise KeyError(f"unknown resample label {label!r}")
    return RESAMPLE_OPTIONS[label]


def has_simulator_rows(sources: list[str] | pd.Series) -> bool:
    """True when any displayed row carries ``source='simulator'`` (drives the banner)."""
    return any(s == "simulator" for s in sources)


def observations_to_frame(body: dict[str, Any]) -> pd.DataFrame:
    """Columnar API response -> tidy DataFrame with a UTC ``ts`` column."""
    cols = {k: v for k, v in body.items() if isinstance(v, list)}
    df = pd.DataFrame(cols)
    if df.empty:
        return pd.DataFrame(
            columns=["ts", "node_id", "channel", "value", "unit", "quality", "kind", "source"]
        )
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def split_by_quality(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Rows per quality flag. ``good`` is what the line is drawn through; the others are markers."""
    return {q: df[df["quality"] == q] for q in ("good", "suspect", "bad")}


def minutes_silent_text(minutes: float | None) -> str:
    """Human text for ``/nodes.minutes_since_seen``."""
    if minutes is None:
        return "never seen"
    if minutes < 60:
        return f"{minutes:.0f} min ago"
    if minutes < 48 * 60:
        return f"{minutes / 60:.1f} h ago"
    return f"{minutes / 1440:.1f} d ago"


def latest_per_channel(df: pd.DataFrame) -> dict[str, tuple[float, str, str, datetime]]:
    """Newest (value, unit, quality, ts) per channel from a raw observations frame."""
    out: dict[str, tuple[float, str, str, datetime]] = {}
    if df.empty:
        return out
    for channel, g in df.sort_values("ts").groupby("channel"):
        row = g.iloc[-1]
        out[str(channel)] = (float(row["value"]), str(row["unit"]), str(row["quality"]), row["ts"])
    return out


# ------------------------------------------------------------------ forecasts and models
ALGO_COLOR = {
    "ridge": "#2a78d6",
    "random_forest": "#eb6834",
    "gradient_boosting": "#1baf7a",
}
BASELINE_COLOR = "#a9a8a2"
BLOCK_TITLE = {"walk_forward": "walk-forward", "holdout": "final hold-out"}
PERSISTENCE_VERSION = "persistence-v1"
VERDICT_TEXT = {
    "beats": "beats",
    "worse": "is worse than",
    "no_difference": "no demonstrable difference from",
    "too_few_days": "too few days to compare with",
    "not_scored": "not compared with",
}


def benchmark_frame(body: dict[str, Any], block: str) -> pd.DataFrame:
    """One row per predictor of a backtest block, baselines first, with verdict columns."""
    cols = ["predictor", "is_baseline", "mae", "rmse", "bias", "r2", "skill", "coverage", "color"]
    predictors = body.get("blocks", {}).get(block, {}).get("predictors", {})
    rows = []
    for name, e in predictors.items():
        if e.get("mae") is None:
            continue
        row = {
            "predictor": name.replace("_", " "),
            "is_baseline": bool(e["is_baseline"]),
            "mae": e["mae"],
            "rmse": e["rmse"],
            "bias": e["bias"],
            "r2": e["r2"],
            "skill": e.get("skill_vs_persistence"),
            "coverage": e.get("interval_coverage"),
            "color": BASELINE_COLOR if e["is_baseline"] else ALGO_COLOR.get(name, "#0b0b0b"),
        }
        for ref, v in e.get("vs", {}).items():
            row[f"vs {ref.replace('_', ' ')}"] = VERDICT_TEXT.get(v["verdict"], v["verdict"])
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    return df.sort_values("is_baseline", ascending=False, kind="stable").reset_index(drop=True)


def verdict_sentences(body: dict[str, Any], block: str) -> list[str]:
    """The report's own verdict sentences for the trained models of a block."""
    predictors = body.get("blocks", {}).get(block, {}).get("predictors", {})
    return [
        v["sentence"]
        for e in predictors.values()
        if not e["is_baseline"]
        for v in e.get("vs", {}).values()
    ]


def event_frame(body: dict[str, Any], block: str) -> pd.DataFrame:
    """Event scores per predictor of a block (empty when the task has no event)."""
    predictors = body.get("blocks", {}).get(block, {}).get("predictors", {})
    rows = [
        {
            "predictor": name.replace("_", " "),
            "precision": e["event"]["precision"],
            "recall": e["event"]["recall"],
            "F1": e["event"]["f1"],
            "Brier": e["event"]["brier"],
            "caught": e["event"]["tp"],
            "false alarms": e["event"]["fp"],
            "missed": e["event"]["fn"],
        }
        for name, e in predictors.items()
        if e.get("event")
    ]
    return pd.DataFrame(rows)


def predictions_to_frame(body: dict[str, Any]) -> pd.DataFrame:
    """Columnar /predictions response -> DataFrame with UTC timestamps."""
    cols = ["issued_at", "valid_at", "node_id", "target", "horizon_h", "yhat", "lo", "hi",
            "p_event", "model_version"]  # fmt: skip
    df = pd.DataFrame({c: body.get(c, []) for c in cols})
    for c in ("issued_at", "valid_at"):
        df[c] = pd.to_datetime(df[c], utc=True)
    return df


def latest_forecast(df: pd.DataFrame, version: str) -> dict[str, Any] | None:
    """The newest stored forecast of one model version (None when there is none)."""
    part = df[df["model_version"] == version]
    if part.empty:
        return None
    return part.sort_values("issued_at").iloc[-1].to_dict()


def event_phrase(event: dict[str, Any] | None, unit: str) -> str:
    """'at or below 6 mg/L' for a task's event."""
    if not event:
        return ""
    return f"at or {event['direction']} {event['threshold']:g} {unit}"


def describe_horizon(hours: float) -> str:
    """Human text for a horizon in hours."""
    if hours < 1:
        return f"{hours * 60:.0f} min"
    if hours < 48:
        return f"{hours:g} h"
    return f"{hours / 24:g} d"
