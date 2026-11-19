"""Models: backtest scores of every model next to its baselines, the registry, and the jobs."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.api_client import ApiClient, ApiError
from dashboard.helpers import (
    BANNER_TEXT,
    BLOCK_TITLE,
    benchmark_frame,
    describe_horizon,
    event_frame,
    event_phrase,
    verdict_sentences,
)
from dashboard.ui import api_error, empty_state, page_setup, run_job, sidebar, style_fig

page_setup(
    "Models",
    "Test each model on data it has not seen, compare it with simple baselines, then choose "
    "which one makes the forecasts. A model only counts as better when the data can show it.",
)
client = ApiClient.from_env()
sidebar(client)

try:
    tasks = client.tasks()
except ApiError as exc:
    api_error(exc)
    st.stop()
for broken in [t for t in tasks if "error" in t]:
    st.error(f"Task file `{broken['name']}` is invalid: {broken['error']}")
tasks = [t for t in tasks if "error" not in t]
if not tasks:
    empty_state("This project has no forecast task. Add one under the project's tasks/ folder.")
    st.stop()

task = st.selectbox("Task", tasks, format_func=lambda t: t["name"], key="m_task")
unit = task["unit"]
summary = (
    f"Forecasts `{task['target']}` {describe_horizon(task['horizon_h'])} ahead on a "
    f"{task['resolution']} grid."
)
if task["event"]:
    summary += f" Event: {event_phrase(task['event'], unit)}."
st.caption(f"{task['description']} {summary}")

b1, b2, b3 = st.columns(3)
if b1.button("1. Run backtest", icon=":material/science:", width="stretch",
             help="Score every model against the baselines. Changes nothing in use."):  # fmt: skip
    run_job(client, "backtest", task=task["name"])
if b2.button("2. Train model", icon=":material/model_training:", width="stretch",
             help="Fit the best backtested algorithm on all data and register it."):  # fmt: skip
    run_job(client, "train", task=task["name"])
if b3.button("3. Run forecast", icon=":material/play_arrow:", width="stretch",
             help="Store a forecast from the model in use for every node."):  # fmt: skip
    run_job(client, "forecast", task=task["name"])

# ------------------------------------------------------------------------------ backtest
st.subheader("Backtest")
body = None
try:
    body = client.benchmarks(task["name"])
except ApiError as exc:
    if "404" in str(exc):
        empty_state("No backtest yet for this task. Press **1. Run backtest**.")
    else:
        api_error(exc)

if body is not None:
    if body.get("simulated"):
        st.warning(BANNER_TEXT, icon=":material/warning:")
    blocks = list(body["blocks"])
    if not blocks:
        st.warning("The backtest could not score anything. " + " ".join(body["warnings"]))
    else:
        default = blocks.index("holdout") if "holdout" in blocks else 0
        block = st.radio("Evaluation", blocks, index=default, horizontal=True,
                         format_func=lambda b: BLOCK_TITLE.get(b, b), key="m_block")  # fmt: skip
        data = body["blocks"][block]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("test rows", data["n"])
        m2.metric("days covered", data["n_days"])
        if task["event"]:
            m3.metric("event rows", data.get("event_positives", 0))
        m4.metric("backtest run", body["created_at"][:10])

        for sentence in verdict_sentences(body, block):
            st.markdown(f"- {sentence}")

        df = benchmark_frame(body, block)
        ci = f"{100 * body.get('ci', 0.9):.0f}%"
        fig = go.Figure(
            go.Bar(
                x=df["mae"],
                y=df["predictor"],
                orientation="h",
                marker_color=df["color"],
                text=[f"{v:.3g}" for v in df["mae"]],
                textposition="outside",
                hovertemplate="%{y}<br>%{x:.4g} " + unit + "<extra></extra>",
            )  # fmt: skip
        )
        fig.update_layout(xaxis_title=f"average error, MAE ({unit}); lower is better",
                          yaxis_title="grey = baseline", yaxis={"autorange": "reversed"},
                          showlegend=False)  # fmt: skip
        st.plotly_chart(style_fig(fig, 90 + 44 * len(df)), width="stretch")
        verdict_cols = [c for c in df.columns if c.startswith("vs ")]
        table = df[["predictor", "mae", "rmse", "bias", "r2", "skill", "coverage", *verdict_cols]]
        st.dataframe(
            table.rename(columns={"mae": f"MAE ({unit})", "rmse": "RMSE", "r2": "R2",
                                  "skill": "improvement over persistence",
                                  "coverage": "range hit rate"}),
            hide_index=True, width="stretch",
            column_config={
                "improvement over persistence": st.column_config.NumberColumn(format="percent"),
                "range hit rate": st.column_config.NumberColumn(format="percent"),
            },
        )  # fmt: skip
        st.caption(
            f'A verdict says "beats" or "is worse than" only when the {ci} interval of the error '
            "difference excludes zero. Range hit rate should be near "
            f"{100 * body.get('interval_level', 0.9):.0f}%."
        )

        events = event_frame(body, block)
        if not events.empty:
            st.markdown(f"**Catching the event** ({event_phrase(task['event'], unit)})")
            st.dataframe(events, hide_index=True, width="stretch")
            st.caption("A row is flagged when the estimated chance of the event is 50% or more.")

        with st.expander("What each baseline means"):
            for name, text in body.get("baseline_meaning", {}).items():
                st.markdown(f"- **{name.replace('_', ' ')}**: {text}")
        if body.get("importance"):
            imp = body["importance"]
            with st.expander(f"What the best model ({imp['model'].replace('_', ' ')}) relies on"):
                top = pd.Series(imp["features"]).head(12).iloc[::-1]
                ifig = go.Figure(go.Bar(x=top.values, y=[n[2:] for n in top.index],
                                        orientation="h", marker_color="#2a78d6"))  # fmt: skip
                ifig.update_layout(xaxis_title=f"increase in error when shuffled ({unit})",
                                   yaxis_title="input", showlegend=False)  # fmt: skip
                st.plotly_chart(style_fig(ifig, 80 + 26 * len(top)), width="stretch")
                st.caption("lagN = the value N grid steps before the forecast is issued.")
        notes = body["warnings"] + [f"Skipped: {s}" for s in body["skipped"]]
        if notes:
            with st.expander(f"Warnings ({len(notes)})", expanded=True):
                for w in notes:
                    st.markdown(f"- {w}")
        st.caption(f"Full report: `{body['report_path']}/report.md`")

# ------------------------------------------------------------------------------ registry
st.subheader("Trained models")
try:
    registry = client.models(task["name"])
except ApiError as exc:
    api_error(exc)
    st.stop()
if not registry:
    empty_state("No model trained yet. Run a backtest, then press **2. Train model**.")
else:
    rows = pd.DataFrame(
        [
            {
                "version": m["version"],
                "status": "in use" if m["status"] == "champion" else m["status"],
                "algorithm": m["algorithm"].replace("_", " "),
                "trained rows": m["n_train"],
                "trained up to (UTC)": str(m["train_end"])[:16],
                f"MAE ({unit})": (m["metrics"] or {}).get("mae"),
                "improvement over persistence": (m["metrics"] or {}).get("skill_vs_persistence"),
                "sources": m["train_sources"],
            }
            for m in registry
        ]
    )
    st.dataframe(
        rows, hide_index=True, width="stretch",
        column_config={
            "improvement over persistence": st.column_config.NumberColumn(format="percent")
        },
    )  # fmt: skip
    st.caption(
        "MAE here is each model's walk-forward error on its own training period. The first model "
        "of a task goes into use at once; later ones wait as candidates until you switch."
    )
    candidates = [m["version"] for m in registry if m["status"] != "champion"]
    if candidates:
        p1, p2 = st.columns([3, 1])
        pick = p1.selectbox("Put another model into use", candidates, key="m_promote")
        if p2.button("Use this model", width="stretch"):
            run_job(client, "promote", version=pick)
    with st.expander("History of each model"):
        for m in registry:
            st.markdown(f"`{m['version']}`: {m['notes']}")
