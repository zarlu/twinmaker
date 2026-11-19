"""Import data: load a CSV of past readings by mapping its columns to channels."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from connectors.csv_import import DEFAULT_SOURCE, MappingError, post_readings, readings_from_frame
from dashboard.api_client import ApiClient, ApiError
from dashboard.ui import api_error, page_setup, sidebar
from sim.client import SimClient

SKIP = "(skip)"

page_setup(
    "Import data",
    "Load readings you already have. Upload a CSV with one row per time step, say which column "
    "is which, and the rows go through the same checks as live data.",
)
client = ApiClient.from_env()
settings = sidebar(client)

try:
    channels = {c["channel"]: c for c in client.channels()}
    node_ids = [n["node_id"] for n in client.nodes()]
except ApiError as exc:
    api_error(exc)
    st.stop()

upload = st.file_uploader("CSV file", type=["csv"])
if upload is None:
    st.info(
        "Expected layout: a timestamp column, optionally a column naming the node, and one "
        "column per measured channel. For files with one row per value (long layout), use the "
        "command line importer described in docs/INGEST.md."
    )
    st.stop()

try:
    df = pd.read_csv(upload, dtype=str, encoding="utf-8-sig")
except (ValueError, UnicodeDecodeError) as exc:
    st.error(f"This file could not be read as CSV: {exc}")
    st.stop()
if df.empty or len(df.columns) < 2:
    st.error("The file needs a header row, a timestamp column and at least one value column.")
    st.stop()
st.caption(f"{len(df)} rows, {len(df.columns)} columns. First rows:")
st.dataframe(df.head(5), hide_index=True, width="stretch")

columns = list(df.columns)
st.subheader("1. Time and node")
c1, c2, c3 = st.columns(3)
ts_col = c1.selectbox("Timestamp column", columns, key="imp_ts")
zone = c2.text_input(
    "Time zone of the file", settings["project"].get("timezone", "UTC"), key="imp_tz",
    help="Used for timestamps without an offset, e.g. America/Chicago. Stored as UTC.",
)  # fmt: skip
fmt = c3.text_input("Timestamp format (optional)", "", key="imp_fmt",
                    help='Leave empty to detect it. Example: "%m/%d/%Y %H:%M"')  # fmt: skip
n1, n2 = st.columns(2)
node_mode = n1.radio("Node", ["one node for the whole file", "a column names the node"],
                     key="imp_node_mode")  # fmt: skip
if node_mode.startswith("one"):
    node_value = n2.selectbox("Node id", node_ids, accept_new_options=True, key="imp_node")
    node_cfg = {"value": node_value}
    used = {ts_col}
else:
    node_col = n2.selectbox("Node column", [c for c in columns if c != ts_col], key="imp_node_col")
    node_cfg = {"column": node_col}
    used = {ts_col, node_col}

st.subheader("2. Columns to channels")
st.caption(
    "Pick the channel for each column to import. Stored value = file value x scale + offset "
    "(use these to convert units)."
)
value_columns = [c for c in columns if c not in used]
guess = {c: next((ch for ch in channels if ch.lower() == c.strip().lower()), SKIP)
         for c in value_columns}  # fmt: skip
edited = st.data_editor(
    pd.DataFrame({
        "column": value_columns,
        "channel": [guess[c] for c in value_columns],
        "scale": 1.0,
        "offset": 0.0,
    }),
    hide_index=True, width="stretch", key="imp_map", disabled=["column"],
    column_config={
        "channel": st.column_config.SelectboxColumn(options=[SKIP, *channels], required=True),
        "scale": st.column_config.NumberColumn(format="%.6g"),
        "offset": st.column_config.NumberColumn(format="%.6g"),
    },
)  # fmt: skip
mapped = edited[edited["channel"] != SKIP]
for _, row in mapped.iterrows():
    st.caption(f"`{row['column']}` -> `{row['channel']}` in {channels[row['channel']]['unit']}")

st.subheader("3. Import")
source = st.text_input("Source name stored with these rows", DEFAULT_SOURCE, key="imp_source",
                       help="A short tag for where the data came from.")  # fmt: skip
mapping = {
    "format": "wide",
    "timestamp": {"column": ts_col, "timezone": zone or "UTC", "format": fmt or None},
    "node": node_cfg,
    "source": source,
    "columns": {
        row["column"]: {"channel": row["channel"], "scale": row["scale"], "offset": row["offset"]}
        for _, row in mapped.iterrows()
    },
}
if mapped.empty:
    st.info("Map at least one column to a channel.")
    st.stop()
try:
    grouped, problems = readings_from_frame(df, mapping)
except (MappingError, ValueError) as exc:
    st.error(f"This mapping does not work: {exc}")
    st.stop()
total = sum(len(r) for r in grouped.values())
st.markdown(
    f"Ready: **{total} readings** for {len(grouped)} node(s): {', '.join(sorted(grouped))}."
)
for problem in problems:
    st.warning(problem)
if total == 0:
    st.stop()
if st.button(f"Import {total} readings", type="primary"):
    if not client.api_key:
        st.error("The dashboard has no API_KEY, so it cannot write. Set it in `.env`.")
        st.stop()
    with st.spinner("Importing ..."):
        result = post_readings(SimClient(client.base_url, client.api_key, max_tries=2),
                               grouped, source)  # fmt: skip
    if result["failed_requests"]:
        st.error(f"{result['failed_requests']} request(s) failed. Is the API running?")
    st.success(
        f"Stored {result['accepted']} new readings. {result['duplicates']} were already stored "
        f"and {result['rejected']} were rejected."
    )
    for reason, count in result["rejection_examples"].items():
        st.warning(f"Rejected ({count}x): {reason}")
    st.caption("See them on the Live signals page (widen the time range to the file's dates).")
