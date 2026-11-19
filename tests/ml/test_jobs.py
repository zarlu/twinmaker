"""Backtest, train, promote, forecast and scorecard on simulated data (pipeline behavior only)."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest
from sqlalchemy import func, select

from db.models import Model, Prediction
from ml import jobs, report
from ml.config import PERSISTENCE_VERSION
from tests.ml.conftest import NOW, TASK, fill


def test_backtest_scores_every_model_next_to_both_baselines(filled) -> None:
    out = jobs.backtest(filled, TASK, now=NOW)
    result = report.latest(TASK)
    assert out["best_model"] in ("ridge", "random_forest") and result["simulated"] is True
    assert any("SIMULATED DATA" in w for w in result["warnings"])
    for block in ("walk_forward", "holdout"):
        predictors = result["blocks"][block]["predictors"]
        assert list(predictors) == ["persistence", "seasonal_naive", "ridge", "random_forest"]
        assert predictors["persistence"]["skill_vs_persistence"] == 0.0
        for algo in ("ridge", "random_forest"):
            entry = predictors[algo]
            assert set(entry["vs"]) == {"persistence", "seasonal_naive"}
            assert entry["vs"]["persistence"]["sentence"].startswith(algo)
            assert entry["event"]["n"] == entry["n"] > 0
    hold = result["blocks"]["holdout"]["predictors"]["ridge"]
    assert 0.5 <= hold["interval_coverage"] <= 1.0 and hold["interval_half_width"] > 0
    assert result["importance"]["model"] == out["best_model"]
    # Chronology of every fold, as recorded in the report.
    for fold in result["folds"]:
        assert fold["train_last_valid_at"] < fold["test_first_issue"]
    text = (report.latest_folder(TASK) / "report.md").read_text()
    assert text.startswith("# Backtest report: temp_3h") and "SIMULATED DATA" in text
    assert "## Verdict" in text and "(baseline)" in text


def test_backtest_is_reproducible(filled) -> None:
    jobs.backtest(filled, TASK, now=NOW)
    first = report.latest(TASK)["blocks"]
    report.latest_folder(TASK).rename(report.latest_folder(TASK).with_name("0-first"))
    jobs.backtest(filled, TASK, now=NOW)
    assert report.latest(TASK)["blocks"] == first


def test_jobs_explain_what_is_missing(session) -> None:
    with pytest.raises(jobs.JobError, match="no usable rows"):
        jobs.backtest(session, TASK, now=NOW)
    with pytest.raises(jobs.JobError, match="no backtest report"):
        jobs.train(session, TASK, now=NOW)
    with pytest.raises(jobs.JobError, match="no champion"):
        jobs.forecast(session, TASK, now=NOW)
    with pytest.raises(jobs.JobError, match="unknown model version"):
        jobs.promote(session, "nope")
    assert jobs.scorecard(session, TASK)["note"] == "No champion model yet."


def test_train_registers_and_only_an_explicit_promote_changes_the_champion(filled) -> None:
    first = jobs.train(filled, TASK, algorithm="ridge", now=NOW)
    assert first["status"] == "champion" and first["walk_forward"]["n"] > 0
    second = jobs.train(filled, TASK, algorithm="random_forest", now=NOW + timedelta(seconds=1))
    assert second["status"] == "candidate"
    assert jobs.champion(filled, TASK).version == first["version"]
    row = filled.get(Model, second["version"])
    assert "kept until someone promotes" in row.notes and row.train_sources == "simulator"
    assert json.loads(row.features_json)[0].startswith("f_")
    out = jobs.promote(filled, second["version"])
    assert out["retired"] == first["version"]
    assert jobs.champion(filled, TASK).version == second["version"]
    assert filled.get(Model, first["version"]).status == "retired"
    assert jobs.promote(filled, second["version"])["retired"] is None  # already champion


def test_train_uses_the_reports_best_model_when_none_is_named(filled) -> None:
    best = jobs.backtest(filled, TASK, now=NOW)["best_model"]
    assert jobs.train(filled, TASK, now=NOW)["algorithm"] == best


def test_forecast_stores_champion_and_persistence_and_is_idempotent(filled) -> None:
    version = jobs.train(filled, TASK, algorithm="ridge", now=NOW)["version"]
    out = jobs.forecast(filled, TASK, backfill_steps=5, now=NOW)
    assert out["nodes"] == 2 and out["issue_times"] == 6 and out["written"] == 2 * 6 * 2
    again = jobs.forecast(filled, TASK, backfill_steps=5, now=NOW)
    assert again["written"] == 0 and again["already_stored"] == 24
    rows = filled.scalars(select(Prediction).order_by(Prediction.issued_at)).all()
    assert {r.model_version for r in rows} == {version, PERSISTENCE_VERSION}
    newest = [r for r in rows if r.model_version == version][-1]
    assert newest.valid_at - newest.issued_at == timedelta(hours=3) and newest.horizon_h == 3.0
    assert newest.lo < newest.yhat < newest.hi and 0.0 <= newest.p_event <= 1.0
    # The newest issue time is the last complete hour before NOW.
    assert newest.issued_at == NOW - timedelta(hours=1)
    base = [r for r in rows if r.model_version == PERSISTENCE_VERSION][-1]
    assert base.lo is None and base.p_event in (0.0, 1.0)


def test_scorecard_grades_only_outcomes_after_the_training_period(session) -> None:
    fill(session)
    # Train as it would have been done 2 days before NOW, then forecast the days since.
    version = jobs.train(session, TASK, algorithm="ridge", holdback_days=2, now=NOW)["version"]
    train_end = session.get(Model, version).train_end
    assert train_end <= NOW - timedelta(days=2)
    jobs.forecast(session, TASK, backfill_steps=100, now=NOW)
    card = jobs.scorecard(session, TASK, now=NOW)
    model_row = next(r for r in card["rows"] if not r["is_baseline"])
    base_row = next(r for r in card["rows"] if r["is_baseline"])
    assert model_row["n"] == base_row["n"] == card["n_scored"] > 50
    assert card["first_valid"] > train_end.isoformat()
    assert base_row["skill_vs_persistence"] == 0.0 and model_row["interval_coverage"] is not None
    assert "event_recall" in model_row
    stored = session.scalar(
        select(func.count()).select_from(Prediction).where(Prediction.model_version == version)
    )
    assert stored > card["n_scored"]  # in-sample and not-yet-due forecasts are not graded


def test_scorecard_says_when_nothing_is_due_yet(filled) -> None:
    jobs.train(filled, TASK, algorithm="ridge", now=NOW)
    jobs.forecast(filled, TASK, now=NOW)
    card = jobs.scorecard(filled, TASK, now=NOW)
    assert card["rows"] == [] and "none of their outcomes has been measured yet" in card["note"]
