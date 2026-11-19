"""Overview: one card per node with its latest values, and the champion models."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import streamlit as st

from dashboard.api_client import ApiClient, ApiError
from dashboard.helpers import (
    has_simulator_rows,
    latest_per_channel,
    minutes_silent_text,
    observations_to_frame,
    to_display_tz,
)
from dashboard.ui import REFRESH_SECONDS, api_error, banner, empty_state, page_setup, sidebar

page_setup("Overview", "Every node, its latest readings, and the models now in use.")
client = ApiClient.from_env()
settings = sidebar(client)
banner_slot = st.empty()


def render() -> None:
    """Fetch nodes and their latest values, draw the cards."""
    try:
        nodes = client.nodes()
    except ApiError as exc:
        api_error(exc)
        return
    if not nodes:
        empty_state("No nodes yet. List them in the project's nodes.yaml or send data.")
        return
    key_channels = settings["project"].get("key_channels") or []
    simulated = False
    since = datetime.now(UTC) - timedelta(hours=24)
    cols = st.columns(min(len(nodes), 4))
    for i, node in enumerate(nodes):
        with cols[i % len(cols)], st.container(border=True):
            st.subheader(node["node_id"])
            st.caption(f"{node['role']} - {node.get('name') or ''}")
            st.markdown(f"**Last seen:** {minutes_silent_text(node['minutes_since_seen'])}")
            if node["last_seen"]:
                shown = to_display_tz([node["last_seen"]], settings["tz"]).iloc[0]
                st.caption(f"{shown:%Y-%m-%d %H:%M %Z}")
            c1, c2, c3 = st.columns(3)
            c1.metric("good (1 h)", node["good_1h"])
            c2.metric("suspect (1 h)", node["suspect_1h"])
            c3.metric("bad (1 h)", node["bad_1h"])
            try:
                body = client.observations(node_id=node["node_id"], start=since, kind="measurement")
            except ApiError as exc:
                st.warning(f"latest values unavailable: {exc}")
                continue
            df = observations_to_frame(body)
            simulated = simulated or has_simulator_rows(df.get("source", []))
            latest = latest_per_channel(df)
            if not latest:
                st.caption("no measurements in the last 24 h")
            for ch in key_channels or sorted(latest):
                if ch in latest:
                    value, unit, quality, _ = latest[ch]
                    flag = "" if quality == "good" else f" ({quality})"
                    st.markdown(f"`{ch}` {value:.3g} {unit}{flag}")
    banner(simulated, banner_slot)
    st.divider()
    with st.container(border=True):
        st.subheader("Models in use")
        try:
            champions = [m for m in client.models() if m["status"] == "champion"]
        except ApiError:
            champions = []
        if not champions:
            empty_state("No model trained yet. Open the Models page to backtest and train one.")
        for m in champions:
            st.markdown(f"`{m['version']}`")
            st.caption(
                f"Task {m['task']}: forecasts {m['target']} {m['horizon_h']:g} h ahead. Trained "
                f"on {m['n_train']} rows up to {str(m['train_end'])[:16]} UTC (sources: "
                f"{m['train_sources']}). Scores are on the Models page."
            )


if settings["auto_refresh"]:

    @st.fragment(run_every=REFRESH_SECONDS)
    def live() -> None:
        render()

    live()
else:
    render()
