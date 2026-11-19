"""Unit tests for the dashboard's pure helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from dashboard.helpers import (
    BANNER_TEXT,
    benchmark_frame,
    describe_horizon,
    event_phrase,
    has_simulator_rows,
    latest_forecast,
    latest_per_channel,
    minutes_silent_text,
    observations_to_frame,
    predictions_to_frame,
    resample_param,
    split_by_quality,
    time_range_bounds,
    to_display_tz,
    tz_options,
    verdict_sentences,
)


def test_to_display_tz_converts_for_display_only() -> None:
    s = to_display_tz(["2026-07-01T18:00:00Z", "2026-01-01T18:00:00Z"], "America/Chicago")
    assert s.iloc[0].hour == 13 and s.iloc[0].utcoffset() == timedelta(hours=-5)  # CDT
    assert s.iloc[1].hour == 12 and s.iloc[1].utcoffset() == timedelta(hours=-6)  # CST
    utc = to_display_tz(["2026-07-01T18:00:00Z"], "UTC")
    assert utc.iloc[0].hour == 18
    assert str(s.iloc[0].tz) == "America/Chicago"


def test_resample_param_mapping() -> None:
    assert resample_param("raw") is None
    assert resample_param("15 min") == "15min"
    assert resample_param("1 h") == "1h"
    assert resample_param("1 day") == "1D"
    with pytest.raises(KeyError):
        resample_param("2 h")


def test_banner_logic() -> None:
    assert has_simulator_rows(["sensor", "simulator"]) is True
    assert has_simulator_rows(["sensor", "HI713"]) is False
    assert has_simulator_rows([]) is False
    assert has_simulator_rows(pd.Series(["simulator"])) is True
    assert "SIMULATED DATA" in BANNER_TEXT and "pipeline validation only" in BANNER_TEXT


def test_time_range_bounds() -> None:
    now = datetime(2026, 10, 10, 12, tzinfo=UTC)
    start, end = time_range_bounds("last 6 h", now)
    assert end == now and start == now - timedelta(hours=6)
    start, end = time_range_bounds("last 30 d", now)
    assert start == now - timedelta(days=30)
    start, end = time_range_bounds(
        "custom", now, custom=(datetime(2026, 10, 1), datetime(2026, 10, 2))
    )
    assert start.tzinfo is UTC and start.day == 1 and end.day == 2
    with pytest.raises(ValueError):
        time_range_bounds("custom", now)


def test_quality_split_and_frame() -> None:
    body = {
        "count": 3,
        "resample": None,
        "ts": ["2026-10-01T00:00:00Z", "2026-10-01T00:15:00Z", "2026-10-01T00:30:00Z"],
        "node_id": ["n"] * 3,
        "channel": ["ph"] * 3,
        "value": [7.0, 7.1, 9.9],
        "unit": ["pH"] * 3,
        "quality": ["good", "suspect", "bad"],
        "kind": ["measurement"] * 3,
        "source": ["simulator"] * 3,
        "meta": [{}, {"reason": "ph_settling"}, {"reason": "spike"}],
    }
    df = observations_to_frame(body)
    assert str(df["ts"].dt.tz) == "UTC" and len(df) == 3
    parts = split_by_quality(df)
    assert len(parts["good"]) == len(parts["suspect"]) == len(parts["bad"]) == 1
    assert parts["bad"]["value"].iloc[0] == 9.9
    latest = latest_per_channel(df)
    assert latest["ph"][0] == 9.9 and latest["ph"][2] == "bad"
    empty = observations_to_frame({"count": 0, "ts": [], "value": [], "quality": [], "source": []})
    assert empty.empty and latest_per_channel(empty) == {}


def test_minutes_silent_text() -> None:
    assert minutes_silent_text(None) == "never seen"
    assert minutes_silent_text(5.2) == "5 min ago"
    assert minutes_silent_text(150) == "2.5 h ago"
    assert minutes_silent_text(3 * 1440) == "3.0 d ago"


def test_tz_options_put_the_project_zone_first_without_duplicates() -> None:
    assert tz_options("America/Chicago") == ("America/Chicago", "UTC")
    assert tz_options("UTC") == ("UTC",)
    assert tz_options(None) == ("UTC",)


BODY = {
    "blocks": {
        "holdout": {
            "predictors": {
                "persistence": {"is_baseline": True, "mae": 1.0, "rmse": 1.2, "bias": 0.0,
                                "r2": 0.1, "skill_vs_persistence": 0.0, "vs": {}},
                "ridge": {"is_baseline": False, "mae": 0.5, "rmse": 0.7, "bias": 0.1, "r2": 0.8,
                          "skill_vs_persistence": 0.5, "interval_coverage": 0.9,
                          "vs": {"persistence": {"verdict": "beats", "sentence": "ridge beats."}}},
                "random_forest": {"is_baseline": False, "mae": None, "rmse": None, "bias": None,
                                  "r2": None, "skill_vs_persistence": None, "vs": {}},
            }
        }
    }
}  # fmt: skip


def test_benchmark_frame_puts_baselines_first_and_words_the_verdicts() -> None:
    df = benchmark_frame(BODY, "holdout")
    assert list(df["predictor"]) == ["persistence", "ridge"]  # unscored model left out
    assert df.loc[1, "vs persistence"] == "beats" and df.loc[1, "coverage"] == 0.9
    assert benchmark_frame(BODY, "walk_forward").empty
    assert verdict_sentences(BODY, "holdout") == ["ridge beats."]


def test_predictions_frame_and_latest_forecast() -> None:
    body = {
        "issued_at": ["2026-10-01T00:00:00Z", "2026-10-01T01:00:00Z", "2026-10-01T01:00:00Z"],
        "valid_at": ["2026-10-01T06:00:00Z", "2026-10-01T07:00:00Z", "2026-10-01T07:00:00Z"],
        "node_id": ["a", "a", "a"], "target": ["x"] * 3, "horizon_h": [6.0] * 3,
        "yhat": [1.0, 2.0, 9.0], "lo": [0.5, 1.5, None], "hi": [1.5, 2.5, None],
        "p_event": [0.1, 0.2, 0.0], "model_version": ["m1", "m1", "persistence-v1"],
    }  # fmt: skip
    df = predictions_to_frame(body)
    assert str(df["valid_at"].dt.tz) == "UTC"
    assert latest_forecast(df, "m1")["yhat"] == 2.0
    assert latest_forecast(df, "other") is None
    assert predictions_to_frame({"count": 0}).empty


def test_event_phrase_and_horizon_text() -> None:
    assert event_phrase({"threshold": 6.0, "direction": "below"}, "mg/L") == "at or below 6 mg/L"
    assert event_phrase(None, "mg/L") == ""
    assert [describe_horizon(h) for h in (0.5, 6, 72)] == ["30 min", "6 h", "3 d"]
