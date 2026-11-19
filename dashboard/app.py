"""Dashboard entry point: ``streamlit run dashboard/app.py`` (or ``make dash``)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

VIEWS = Path(__file__).parent / "views"

st.set_page_config(page_title="TwinMaker", layout="wide")
page = st.navigation(
    [
        st.Page(
            str(VIEWS / "overview.py"), title="Overview", icon=":material/dashboard:", default=True
        ),
        st.Page(str(VIEWS / "live_signals.py"), title="Live signals", icon=":material/show_chart:"),
        st.Page(str(VIEWS / "forecast.py"), title="Forecast", icon=":material/event_upcoming:"),
        st.Page(str(VIEWS / "models.py"), title="Models", icon=":material/leaderboard:"),
        st.Page(str(VIEWS / "import_data.py"), title="Import data", icon=":material/upload_file:"),
    ]
)
page.run()
