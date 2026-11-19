"""Turn grids into supervised rows without looking into the future.

For an issue time ``t`` and a horizon of ``h`` steps, every feature is computed from grid values
at or before ``t``; the label ``y`` is the target at ``t + h``. The only inputs dated after ``t``
are calendar features of the valid time, which are known in advance.

Columns of the result: ``node_id``, ``t`` (issue time), ``valid_at``, ``y`` (missing when the
future is not known yet), ``y_now`` (target at ``t``, the persistence forecast),
``y_seasonal`` (the seasonal-naive forecast, when the task sets a period) and the features
(prefix ``f_``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.config import step_of

META_COLUMNS = ["node_id", "t", "valid_at", "y", "y_now", "y_seasonal"]


def _calendar(valid_at: pd.DatetimeIndex, names: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if "hour_of_day" in names:
        hours = valid_at.hour + valid_at.minute / 60.0
        out["f_hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
        out["f_hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    if "day_of_week" in names:
        dow = valid_at.day_of_week
        out["f_dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        out["f_dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    if "day_of_year" in names:
        doy = valid_at.day_of_year
        out["f_doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        out["f_doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def node_rows(grid: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    """Supervised rows of one node (one row per grid time that has a current target value)."""
    h = spec["horizon_steps"]
    step = step_of(spec)
    feats = spec["features"]
    target = grid[spec["target"]["channel"]]
    cols: dict[str, Any] = {
        "t": grid.index,
        "valid_at": grid.index + h * step,
        "y": target.shift(-h).to_numpy(),
        "y_now": target.to_numpy(),
    }
    period = spec.get("seasonal_period_steps")
    if period:
        # Same phase of the last full cycle that is already observed at t.
        k = int(np.ceil(h / period))
        cols["y_seasonal"] = target.shift(k * int(period) - h).to_numpy()
    else:
        cols["y_seasonal"] = np.nan
    for lag in feats["target_lags"]:
        cols[f"f_target_lag{int(lag)}"] = target.shift(int(lag)).to_numpy()
    for window in feats["rolling_windows"]:
        roll = target.rolling(int(window), min_periods=max(2, int(window) // 2))
        cols[f"f_target_mean{int(window)}"] = roll.mean().to_numpy()
        cols[f"f_target_std{int(window)}"] = roll.std().to_numpy()
    for channel, lags in feats["channels"].items():
        if channel == spec["target"]["channel"]:
            continue
        series = grid[channel] if channel in grid else pd.Series(np.nan, index=grid.index)
        for lag in lags:
            cols[f"f_{channel}_lag{int(lag)}"] = series.shift(int(lag)).to_numpy()
    cols.update(_calendar(pd.DatetimeIndex(cols["valid_at"]), feats["calendar"]))
    frame = pd.DataFrame(cols)
    return frame[frame["y_now"].notna()].reset_index(drop=True)


def build_rows(grids: dict[str, pd.DataFrame], spec: dict[str, Any]) -> pd.DataFrame:
    """Rows of every node, sorted by issue time. Adds a node one-hot when there are several."""
    parts = []
    for node, grid in grids.items():
        rows = node_rows(grid, spec)
        rows.insert(0, "node_id", node)
        parts.append(rows)
    if not parts:
        return pd.DataFrame(columns=META_COLUMNS)
    out = pd.concat(parts, ignore_index=True)
    nodes = sorted(grids)
    if len(nodes) > 1:
        for node in nodes:
            out[f"f_node_{node}"] = (out["node_id"] == node).astype(float)
    return out.sort_values(["t", "node_id"], kind="stable").reset_index(drop=True)


def feature_columns(rows: pd.DataFrame) -> list[str]:
    return [c for c in rows.columns if c.startswith("f_")]


def align_columns(rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """The feature matrix in a stored model's column order (absent columns become zeros)."""
    out = rows.reindex(columns=columns)
    node_cols = [c for c in columns if c.startswith("f_node_")]
    out[node_cols] = out[node_cols].fillna(0.0)
    return out
