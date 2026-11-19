"""Forecast: what the champion model expects next, and how its past forecasts turned out."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.api_client import ApiClient, ApiError
from dashboard.helpers import (
    PERSISTENCE_VERSION,
    describe_horizon,
    event_phrase,
    has_simulator_rows,
    latest_forecast,
    observations_to_frame,
    predictions_to_frame,
    to_display_tz,
)
from dashboard.ui import api_error, banner, empty_state, page_setup, run_job, sidebar, style_fig

page_setup(
    "Forecast",
    "What the model in use expects next for each node, next to what was measured. Past forecasts "
    "stay on the chart so you can see how they turned out.",
)
client = ApiClient.from_env()
settings = sidebar(client)
banner_slot = st.empty()

try:
    tasks = [t for t in client.tasks() if "error" not in t]
    models = client.models()
except ApiError as exc:
    api_error(exc)
    st.stop()
if not tasks:
    empty_state("This project has no forecast task. Add one under the project's tasks/ folder.")
    st.stop()

c1, c2, c3 = st.columns([3, 2, 2])
task = c1.selectbox("Task", tasks, format_func=lambda t: t["name"], key="fc_task")
champion = next(
    (m for m in models if m["task"] == task["name"] and m["status"] == "champion"), None
)
unit, target = task["unit"], task["target"]
st.caption(task["description"])
if champion is None:
    empty_state(
        f"No model is in use for `{task['name']}` yet. Open the Models page, run a backtest and "
        "train a model; forecasts appear here afterwards."
    )
    st.stop()

if c3.button("Run forecast now", icon=":material/play_arrow:", width="stretch"):
    run_job(client, "forecast", task=task["name"])

horizon = timedelta(hours=float(task["horizon_h"]))
try:
    preds = predictions_to_frame(
        client.predictions(target=target, start=settings["start"], end=settings["end"] + horizon)
    )
except ApiError as exc:
    api_error(exc)
    st.stop()
preds = preds[preds["model_version"].isin([champion["version"], PERSISTENCE_VERSION])]
preds = preds[preds["horizon_h"] == task["horizon_h"]]
if preds.empty:
    empty_state(
        "No forecast is stored for this time range. Press **Run forecast now**, or widen the "
        "time range in the sidebar."
    )
    st.stop()

node = c2.selectbox("Node", sorted(preds["node_id"].unique()), key="fc_node")
preds = preds[preds["node_id"] == node].sort_values("valid_at")
mine = preds[preds["model_version"] == champion["version"]]
naive = preds[preds["model_version"] == PERSISTENCE_VERSION]

try:
    obs = observations_to_frame(
        client.observations(
            node_id=node,
            channel=target,
            start=settings["start"],
            end=settings["end"],
            min_quality="good",
            resample="1h" if horizon >= timedelta(hours=6) else "15min",
            agg="mean",
        )  # fmt: skip
    )
except ApiError as exc:
    api_error(exc)
    st.stop()
banner(has_simulator_rows(obs.get("source", [])), banner_slot)

tz = settings["tz"]
newest = latest_forecast(preds, champion["version"])
if newest is not None:
    valid = to_display_tz([newest["valid_at"]], tz).iloc[0]
    issued = to_display_tz([newest["issued_at"]], tz).iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Forecast for {valid:%b %d %H:%M}", f"{newest['yhat']:.3g} {unit}")
    if pd.notna(newest["lo"]):
        m2.metric("Likely range (90%)", f"{newest['lo']:.3g} to {newest['hi']:.3g}")
    if task["event"] and pd.notna(newest["p_event"]):
        m3.metric(
            f"Chance of {event_phrase(task['event'], unit)}", f"{100 * newest['p_event']:.0f}%"
        )
    m4.metric("Issued", f"{issued:%b %d %H:%M}", help=f"Time zone: {tz}")
    st.caption(
        f"Forecasts look {describe_horizon(task['horizon_h'])} ahead. Model in use: "
        f"`{champion['version']}`."
    )

fig = go.Figure()
HOVER = "%{y:.3g} " + unit + "<br>%{x}"
if not mine.empty and mine["lo"].notna().any():
    x = to_display_tz(mine["valid_at"], tz)
    fig.add_trace(go.Scatter(x=x, y=mine["hi"], mode="lines", line={"width": 0},
                             showlegend=False, hoverinfo="skip"))  # fmt: skip
    fig.add_trace(go.Scatter(x=x, y=mine["lo"], mode="lines", line={"width": 0}, fill="tonexty",
                             fillcolor="rgba(42,120,214,0.15)", name="likely range (90%)",
                             hoverinfo="skip"))  # fmt: skip
if not obs.empty:
    fig.add_trace(go.Scatter(x=to_display_tz(obs["ts"], tz), y=obs["value"], mode="lines",
                             line={"color": "#0F172A", "width": 1.8}, name="measured",
                             hovertemplate=HOVER + "<extra>measured</extra>"))  # fmt: skip
if not naive.empty:
    fig.add_trace(go.Scatter(x=to_display_tz(naive["valid_at"], tz), y=naive["yhat"], mode="lines",
                             line={"color": "#a9a8a2", "width": 1.2, "dash": "dot"},
                             name="baseline: no change",
                             hovertemplate=HOVER + "<extra>no change</extra>"))  # fmt: skip
if not mine.empty:
    fig.add_trace(go.Scatter(x=to_display_tz(mine["valid_at"], tz), y=mine["yhat"], mode="lines",
                             line={"color": "#2a78d6", "width": 2}, name="forecast",
                             hovertemplate=HOVER + "<extra>forecast</extra>"))  # fmt: skip
if task["event"]:
    fig.add_hline(y=task["event"]["threshold"], line_dash="dash", line_color="#d6453d",
                  annotation_text=f"event level {task['event']['threshold']:g} {unit}",
                  annotation_position="top left")  # fmt: skip
fig.update_layout(xaxis_title=f"time the value applies to ({tz})", yaxis_title=f"{target} ({unit})",
                  legend={"orientation": "h", "y": -0.2})  # fmt: skip
st.plotly_chart(style_fig(fig, 460), width="stretch")
st.caption(
    "Each forecast point is drawn at the time it predicts, and was issued "
    f'{describe_horizon(task["horizon_h"])} earlier. "No change" carries the value at issue '
    "time forward: the baseline a useful model must beat."
)

if task["event"] and mine["p_event"].notna().any():
    pfig = go.Figure(
        go.Scatter(
            x=to_display_tz(mine["valid_at"], tz),
            y=100 * mine["p_event"],
            mode="lines",
            line={"color": "#d6453d", "width": 1.8},
            fill="tozeroy",
            fillcolor="rgba(214,69,61,0.12)",
            hovertemplate="%{y:.0f}%<br>%{x}<extra></extra>",
        )  # fmt: skip
    )
    pfig.update_layout(xaxis_title=f"time the value applies to ({tz})",
                       yaxis_title=f"chance of {event_phrase(task['event'], unit)} (%)",
                       yaxis_range=[0, 100], showlegend=False)  # fmt: skip
    st.plotly_chart(style_fig(pfig, 240), width="stretch")
    st.caption(
        "The chance comes from how far off this model's past forecasts were. It assumes future "
        "errors look like past ones."
    )

st.subheader("How past forecasts turned out")
try:
    card = client.scorecard(task["name"])
except ApiError as exc:
    api_error(exc)
    st.stop()
if not card["rows"]:
    empty_state(card["note"] or "Nothing scored yet.")
else:
    table = pd.DataFrame(card["rows"])
    table["model"] = [
        "baseline: no change" if b else "forecast model" for b in table["is_baseline"]
    ]
    shown = table[["model", "n", "mae", "bias", "skill_vs_persistence", "interval_coverage"]]
    shown = shown.rename(columns={
        "n": "forecasts scored", "mae": f"average error ({unit})", "bias": f"bias ({unit})",
        "skill_vs_persistence": "improvement over no change", "interval_coverage": "range hit rate",
    })  # fmt: skip
    st.dataframe(
        shown, hide_index=True, width="stretch",
        column_config={
            "improvement over no change": st.column_config.NumberColumn(format="percent"),
            "range hit rate": st.column_config.NumberColumn(format="percent"),
        },
    )  # fmt: skip
    st.caption(
        f"{card['note']} Both rows are graded on the same {card['n_scored']} forecasts, all "
        "nodes together. Range hit rate should be near 90%."
    )
