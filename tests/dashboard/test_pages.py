"""Headless smoke tests of every view with streamlit.testing.v1.AppTest."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from dashboard.helpers import BANNER_TEXT

VIEWS = Path(__file__).resolve().parents[2] / "dashboard" / "views"
ALL_PAGES = sorted(p.name for p in VIEWS.glob("*.py"))
DATA_PAGES = ["live_signals.py", "overview.py"]  # the views that plot raw observations


def _run(page: str, api, monkeypatch: pytest.MonkeyPatch, timeout: float = 60) -> AppTest:
    monkeypatch.setenv("API_URL", api if isinstance(api, str) else api.url)
    monkeypatch.setenv("API_KEY", "" if isinstance(api, str) else api.key)
    at = AppTest.from_file(str(VIEWS / page), default_timeout=timeout)
    at.run()
    return at


def _texts(at: AppTest) -> str:
    chunks = [w.value for w in at.warning] + [w.value for w in at.info]
    chunks += [w.value for w in at.error] + [w.value for w in at.success]
    chunks += [m.value for m in at.markdown] + [c.value for c in at.caption]
    return "\n".join(str(c) for c in chunks)


def test_pages_exist() -> None:
    assert ALL_PAGES == [
        "forecast.py",
        "import_data.py",
        "live_signals.py",
        "models.py",
        "overview.py",
    ]


@pytest.mark.parametrize("page", ALL_PAGES)
def test_page_handles_api_down(page: str, monkeypatch) -> None:
    at = _run(page, "http://127.0.0.1:9", monkeypatch)  # nothing listens on port 9
    assert not at.exception, [e.value for e in at.exception]
    assert "Could not load data from the API" in _texts(at) or "API down" in _texts(at)


@pytest.mark.parametrize("page", ALL_PAGES)
def test_page_runs_on_empty_database_without_exception(page: str, empty_api, monkeypatch) -> None:
    at = _run(page, empty_api, monkeypatch)
    assert not at.exception, [e.value for e in at.exception]
    text = _texts(at)
    assert BANNER_TEXT not in text
    expected = {
        "overview.py": "No model trained yet",
        "live_signals.py": "No rows in this time range",
        "forecast.py": "No model is in use for `temp_3h` yet",
        "models.py": "No backtest yet for this task",
        "import_data.py": "Expected layout",
    }
    assert expected[page] in text


def test_sidebar_shows_the_project_name_and_zone(empty_api, monkeypatch) -> None:
    at = _run("overview.py", empty_api, monkeypatch)
    assert "### Test project" in [m.value for m in at.sidebar.markdown]
    assert at.radio(key="tz").options == ["America/Chicago", "UTC"]


@pytest.mark.parametrize("page", DATA_PAGES)
def test_page_shows_banner_with_simulated_rows(page: str, populated_api, monkeypatch) -> None:
    at = _run(page, populated_api, monkeypatch)
    assert not at.exception, [e.value for e in at.exception]
    assert BANNER_TEXT in _texts(at)


def test_live_signals_resample_and_utc_toggle(populated_api, monkeypatch) -> None:
    at = _run("live_signals.py", populated_api, monkeypatch)
    at.selectbox(key="resample").select("1 h").run()
    assert not at.exception, [e.value for e in at.exception]
    at.radio(key="tz").set_value("UTC").run()
    assert not at.exception
    charts = at.get("plotly_chart")
    assert charts, "no chart rendered"
    assert "time (UTC)" in str(charts[0].proto)
    assert at.dataframe, "data table missing"
    at.sidebar.toggle(key="auto_refresh").set_value(False).run()
    assert not at.exception


def test_overview_cards_list_every_node_with_key_channels(populated_api, monkeypatch) -> None:
    at = _run("overview.py", populated_api, monkeypatch)
    heads = {h.value for h in at.subheader}
    assert {"node-intake", "node-downstream", "node-bench"} <= heads
    assert "`water_temp`" in _texts(at)


def test_forecast_page_shows_the_forecast_and_the_scorecard(trained_api, monkeypatch) -> None:
    at = _run("forecast.py", trained_api, monkeypatch)
    assert not at.exception, [e.value for e in at.exception]
    text = _texts(at)
    assert BANNER_TEXT in text
    labels = [m.label for m in at.metric]
    assert any(label.startswith("Forecast for") for label in labels)
    assert "Chance of at or above 22 degC" in labels
    assert len(at.get("plotly_chart")) == 2  # forecast chart and event-chance chart
    assert "Out-of-sample only" in text and at.dataframe, "scorecard missing"


def test_models_page_shows_backtest_and_registry(trained_api, monkeypatch) -> None:
    at = _run("models.py", trained_api, monkeypatch)
    assert not at.exception, [e.value for e in at.exception]
    text = _texts(at)
    assert BANNER_TEXT in text  # the backtest used simulated rows
    assert "persistence" in text and "SIMULATED DATA" in text
    labels = [m.label for m in at.metric]
    assert "test rows" in labels and "event rows" in labels
    assert len(at.dataframe) >= 3  # scores, event table, registry
    assert "temp_3h-" in text  # model history lists the version


def test_overview_names_the_model_in_use(trained_api, monkeypatch) -> None:
    at = _run("overview.py", trained_api, monkeypatch)
    assert "Task temp_3h: forecasts water_temp 3 h ahead" in _texts(at)
