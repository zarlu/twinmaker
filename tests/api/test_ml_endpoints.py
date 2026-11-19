"""ML endpoints and the job runner, through the API (TestClient runs background jobs inline)."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from tests.ml.conftest import TASK, fill


def _job(client: TestClient, auth: dict, kind: str, **args: object) -> dict:
    r = client.post(f"/jobs/{kind}", json=args, headers=auth)
    assert r.status_code == 202, r.text
    return client.get(f"/jobs/{r.json()['id']}").json()


def test_tasks_lists_the_project_tasks(client: TestClient) -> None:
    (task,) = client.get("/tasks").json()
    assert task["name"] == TASK and task["target"] == "water_temp" and task["unit"] == "degC"
    assert task["horizon_h"] == 3.0 and task["event"]["direction"] == "above"
    assert task["models"] == ["ridge", "random_forest"] and task["has_report"] is False


def test_empty_database_reads_are_empty_not_errors(client: TestClient) -> None:
    assert client.get("/models").json() == []
    assert client.get("/predictions").json()["count"] == 0
    assert client.get("/jobs").json() == []
    assert client.get("/scorecard", params={"task": TASK}).json()["rows"] == []
    assert client.get(f"/benchmarks/{TASK}").status_code == 404
    assert client.get("/benchmarks/nope").status_code == 404
    assert client.get("/scorecard", params={"task": "nope"}).status_code == 404
    assert client.get("/jobs/nope").status_code == 404


def test_jobs_need_the_api_key_and_their_arguments(client: TestClient, auth: dict) -> None:
    assert client.post("/jobs/backtest", json={"task": TASK}).status_code == 401
    assert client.post("/jobs/backtest", json={}, headers=auth).status_code == 400
    assert client.post("/jobs/promote", json={"task": TASK}, headers=auth).status_code == 400
    assert client.post("/jobs/retrain", json={"task": TASK}, headers=auth).status_code == 422


def test_a_failing_job_reports_why_and_the_api_keeps_running(
    client: TestClient, auth: dict
) -> None:
    job = _job(client, auth, "backtest", task=TASK)  # empty database
    assert job["status"] == "failed" and "no usable rows" in job["error"]
    job = _job(client, auth, "backtest", task="nope")
    assert job["status"] == "failed" and "unknown task" in job["error"]
    assert client.get("/health").json()["db_ok"] is True


def test_full_loop_through_the_api(client: TestClient, auth: dict) -> None:
    with client.app.state.session_factory() as session:
        fill(session, end=datetime.now(UTC))
    backtest = _job(client, auth, "backtest", task=TASK)
    assert backtest["status"] == "done" and backtest["result"]["best_model"]
    bench = client.get(f"/benchmarks/{TASK}").json()
    assert set(bench["blocks"]) == {"walk_forward", "holdout"} and bench["simulated"] is True
    assert client.get("/tasks").json()[0]["has_report"] is True

    train = _job(client, auth, "train", task=TASK, holdback_days=2)
    assert train["status"] == "done" and train["result"]["status"] == "champion"
    (model,) = client.get("/models", params={"task": TASK}).json()
    assert model["status"] == "champion" and model["metrics"]["mae"] > 0
    assert model["train_end"].endswith(("Z", "+00:00"))

    forecast = _job(client, auth, "forecast", task=TASK, backfill_steps=60)
    assert forecast["status"] == "done" and forecast["result"]["written"] == 2 * 61 * 2
    preds = client.get("/predictions", params={"model_version": model["version"]}).json()
    assert preds["count"] == 2 * 61 and set(preds["target"]) == {"water_temp"}
    one = client.get("/predictions", params={"node_id": "node-intake"}).json()
    assert set(one["node_id"]) == {"node-intake"} and one["valid_at"] == sorted(one["valid_at"])

    card = client.get("/scorecard", params={"task": TASK}).json()
    assert [r["is_baseline"] for r in card["rows"]] == [True, False] and card["n_scored"] > 20

    second = _job(client, auth, "train", task=TASK, algorithm="random_forest")
    assert second["result"]["status"] == "candidate"
    promoted = _job(client, auth, "promote", version=second["result"]["version"])
    assert promoted["result"]["retired"] == model["version"]
    statuses = {m["version"]: m["status"] for m in client.get("/models").json()}
    assert statuses == {second["result"]["version"]: "champion", model["version"]: "retired"}
    assert [j["kind"] for j in client.get("/jobs").json()][:2] == ["promote", "train"]
