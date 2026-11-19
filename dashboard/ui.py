"""Shared Streamlit pieces: sidebar, API status, banner, error states. No business logic."""

from __future__ import annotations

import contextlib
import time as time_module
from datetime import UTC, datetime, time, timedelta
from typing import Any

import streamlit as st

from dashboard.api_client import ApiClient, ApiError
from dashboard.helpers import BANNER_TEXT, DEFAULT_TZ, TIME_RANGES, time_range_bounds, tz_options

REFRESH_SECONDS = 15


INK = "#0F172A"
INK_MUTED = "#475569"
BORDER = "#E2E8F0"
SURFACE = "#FFFFFF"
FONT = "'Fira Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

STYLE = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@500;600&family=Fira+Sans:wght@400;500;600;700&display=swap');
html, body, .stApp, p, li, label, h1, h2, h3, h4, input, textarea, th, td,
.stMarkdown, .stButton > button {{ font-family: {FONT}; }}
[data-testid="stIconMaterial"] {{ font-family: "Material Symbols Rounded" !important; }}
.block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
h1 {{ font-weight: 700; letter-spacing: -0.02em; font-size: 2rem; padding-bottom: 0.1rem; }}
h2, h3 {{ font-weight: 600; letter-spacing: -0.01em; }}
h3 {{ font-size: 1.15rem; margin-top: 0.6rem; }}
[data-testid="stMetric"] {{
  background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px; padding: 14px 16px;
}}
[data-testid="stMetricLabel"] p {{ color: {INK_MUTED}; font-size: 0.82rem; font-weight: 500; }}
[data-testid="stMetricValue"] {{
  font-family: 'Fira Code', ui-monospace, monospace; font-weight: 600; font-size: 1.55rem;
  font-variant-numeric: tabular-nums; color: {INK};
}}
[data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 8px; background: {SURFACE}; }}
[data-testid="stAlert"] {{ border-radius: 8px; }}
[data-testid="stCaptionContainer"] {{ color: {INK_MUTED}; }}
.stButton > button {{ border-radius: 8px; font-weight: 600; transition: background 200ms; }}
.page-lead {{ color: {INK_MUTED}; font-size: 1rem; margin: -0.4rem 0 1.1rem 0; max-width: 70ch; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
</style>
"""


def page_setup(title: str, lead: str | None = None) -> None:
    """Page config and shared styling for every view (safe to call in each standalone view)."""
    with contextlib.suppress(st.errors.StreamlitAPIException):  # already set by app.py
        st.set_page_config(
            page_title=f"{title} - TwinMaker",
            layout="wide",
        )
    st.markdown(STYLE, unsafe_allow_html=True)
    st.title(title)
    if lead:
        st.markdown(f'<p class="page-lead">{lead}</p>', unsafe_allow_html=True)


def style_fig(fig: Any, height: int | None = None) -> Any:
    """One look for every Plotly chart: same font, white plot area, light grid."""
    fig.update_layout(
        font={"family": FONT, "color": INK, "size": 13},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=SURFACE,
        margin={"l": 10, "r": 16, "t": 36, "b": 10},
        hoverlabel={"font": {"family": FONT}},
    )
    fig.update_xaxes(gridcolor=BORDER, linecolor=BORDER, zeroline=False,
                     title_font={"color": INK_MUTED, "size": 12})  # fmt: skip
    fig.update_yaxes(gridcolor=BORDER, linecolor=BORDER, zeroline=False,
                     title_font={"color": INK_MUTED, "size": 12})  # fmt: skip
    if height:
        fig.update_layout(height=height)
    return fig


def sidebar(client: ApiClient) -> dict[str, Any]:
    """Render the sidebar and return the current settings (tz, range, refresh)."""
    with st.sidebar:
        project: dict[str, Any] = {}
        status = st.empty()
        try:
            h = client.health()
            project = client.project()
            status.success(f"API ok - v{h['version']} - {h['utc_now'][:19]}Z")
        except ApiError as exc:
            status.error(f"API down: {exc}")
        st.markdown(f"### {project.get('name', 'TwinMaker')}")
        if project.get("description"):
            st.caption(project["description"])
        st.caption(f"API: {client.base_url}")
        zones = tz_options(project.get("timezone"))
        tz = st.radio("Display time zone", zones, index=0, horizontal=True, key="tz")
        st.caption("Data is stored and exchanged in UTC; only the display changes.")
        choice = st.selectbox("Time range", list(TIME_RANGES), index=3, key="time_range")
        custom = None
        if choice == "custom":
            today = datetime.now(UTC).date()
            d0 = st.date_input("start (UTC)", today - timedelta(days=7), key="custom_start")
            d1 = st.date_input("end (UTC)", today, key="custom_end")
            custom = (
                datetime.combine(d0, time.min, tzinfo=UTC),
                datetime.combine(d1, time.max, tzinfo=UTC),
            )
        start, end = time_range_bounds(choice, custom=custom)
        auto = st.toggle(f"Auto-refresh every {REFRESH_SECONDS} s", value=True, key="auto_refresh")
    return {
        "tz": tz or DEFAULT_TZ,
        "start": start,
        "end": end,
        "auto_refresh": auto,
        "project": project,
    }


def banner(show: bool, slot: Any | None = None) -> None:
    """Persistent banner whenever displayed rows include simulated data."""
    target = slot or st
    if show:
        target.warning(BANNER_TEXT, icon=":material/warning:")


def api_error(exc: ApiError) -> None:
    """Friendly error state (never a traceback)."""
    st.error(f"Could not load data from the API. {exc}")
    st.info("Start it with `make api` (or `make demo`) and check `API_URL` in `.env`.")


def empty_state(text: str) -> None:
    """Friendly empty state."""
    st.info(text)


def run_job(client: ApiClient, kind: str, wait_s: float = 300.0, **args: Any) -> dict[str, Any]:
    """Start an ML job through the API and wait for it, showing progress. Returns the job."""
    if not client.api_key:
        st.error("The dashboard has no API_KEY, so it cannot start jobs. Set it in `.env`.")
        return {"status": "failed"}
    with st.status(f"Running {kind} ...", expanded=False) as box:
        try:
            job = client.start_job(kind, **args)
            deadline = time_module.monotonic() + wait_s
            while job["status"] in ("queued", "running") and time_module.monotonic() < deadline:
                time_module.sleep(1.0)
                job = client.job(job["id"])
        except ApiError as exc:
            box.update(label=f"{kind} could not run", state="error")
            st.error(str(exc))
            return {"status": "failed"}
        if job["status"] == "done":
            box.update(label=f"{kind} finished", state="complete")
            st.json(job["result"], expanded=False)
        elif job["status"] == "failed":
            box.update(label=f"{kind} failed", state="error")
            st.error(job["error"])
        else:
            box.update(label=f"{kind} is still running; check back shortly", state="running")
    return job
