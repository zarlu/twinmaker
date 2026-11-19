"""The four explicit ML jobs and the scorecard.

  backtest  score every model of a task against its baselines and save a report
  train     fit one algorithm on all data, save the artifact, add a registry row
  promote   make a registered model the task's champion (the only way a champion changes,
            except that the first model of a task becomes champion at once)
  forecast  store predictions of the champion and of persistence at the same issue times

Models never update themselves: each of these runs only when someone asks for it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from db.models import Model, Node, Prediction
from ml import report
from ml.backtest import run_backtest, walk_forward
from ml.config import PERSISTENCE_VERSION, data_dir, event_hit, load_task
from ml.data import load_grids
from ml.evaluate import event_metrics, regression_metrics, skill
from ml.features import align_columns, build_rows, feature_columns
from ml.models import conformal_quantile, event_probability, make_model
from ml.splits import make_splits


class JobError(Exception):
    """A job cannot run (no data, no report, no champion, ...)."""


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def artifacts_dir() -> Path:
    return data_dir() / "artifacts"


# ------------------------------------------------------------------------------- backtest
def backtest(session: Session, task: str, now: datetime | None = None) -> dict[str, Any]:
    spec = load_task(task)
    grids, sources = load_grids(session, spec, now=now)
    rows = build_rows(grids, spec)
    if rows.empty or rows["y"].notna().sum() == 0:
        raise JobError(
            f"task '{task}': no usable rows for channel '{spec['target']['channel']}'. "
            "Ingest data first (simulator, CSV import or live nodes)."
        )
    result = run_backtest(spec, rows, sources)
    folder = report.save(result)
    return {
        "task": task,
        "report_path": str(folder),
        "n_rows": result["n_rows"],
        "best_model": result["best_model"],
        "warnings": result["warnings"],
    }


# ---------------------------------------------------------------------------------- train
def train(
    session: Session,
    task: str,
    algorithm: str | None = None,
    holdback_days: float = 0.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fit on every labeled row up to ``now - holdback_days`` and register the model."""
    spec = load_task(task)
    now = _utc(now) or datetime.now(UTC)
    cutoff = now - timedelta(days=float(holdback_days))
    algo = algorithm or spec["deploy"].get("algorithm")
    note = "algorithm given explicitly"
    if algo is None:
        latest = report.latest(task)
        if latest is None or not latest.get("best_model"):
            raise JobError(
                f"task '{task}': no backtest report to choose an algorithm from. Run the "
                "backtest first, or set `deploy.algorithm` in the task."
            )
        algo = latest["best_model"]
        note = f"best walk-forward MAE in the report of {latest['created_at']}"
    if algo not in spec["models"]:
        raise JobError(f"task '{task}': algorithm '{algo}' is not listed under `models`")

    grids, sources = load_grids(session, spec, until=cutoff, now=cutoff)
    rows = build_rows(grids, spec)
    labeled = rows[rows["y"].notna()].reset_index(drop=True) if len(rows) else rows
    if len(labeled) < int(spec["split"]["min_train_rows"]):
        raise JobError(
            f"task '{task}': {len(labeled)} labeled rows, fewer than "
            f"split.min_train_rows={spec['split']['min_train_rows']}"
        )
    columns = feature_columns(labeled)
    # Out-of-sample residuals for intervals and event probabilities.
    plan = make_splits(labeled, {**spec["split"], "holdout_frac": 0.01})
    wf = walk_forward(spec, algo, labeled, plan, columns)
    y_wf = labeled["y"].to_numpy(dtype=float)[wf["idx"].to_numpy(dtype=int)]
    residuals = y_wf - wf["pred"].to_numpy(dtype=float)
    wf_rows = labeled.iloc[wf["idx"].to_numpy(dtype=int)]
    metrics = regression_metrics(y_wf, wf["pred"].to_numpy(dtype=float))
    base = regression_metrics(y_wf, wf_rows["y_now"].to_numpy(dtype=float))
    metrics["skill_vs_persistence"] = skill(metrics["mae"], base["mae"])
    metrics["evaluation"] = "walk-forward on the training period"

    model = make_model(algo, spec["models"][algo], int(spec["seed"]))
    model.fit(labeled[columns].to_numpy(dtype=float), labeled["y"].to_numpy(dtype=float))
    created = datetime.now(UTC)
    version = f"{task}-{algo}-{created:%Y%m%d%H%M%S}"
    path = artifacts_dir() / f"{version}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "version": version,
            "spec": spec,
            "algorithm": algo,
            "columns": columns,
            "model": model,
            "residuals": residuals,
            "q": conformal_quantile(residuals, float(spec["intervals"]["level"])),
        },
        path,
    )
    has_champion = session.scalar(
        select(Model).where(Model.task == task, Model.status == "champion")
    )
    status = "candidate" if has_champion else "champion"
    notes = [f"{created:%Y-%m-%dT%H:%M:%SZ} trained ({note})."]
    notes.append(
        "Became champion: first model of this task."
        if status == "champion"
        else f"Candidate: champion {has_champion.version} kept until someone promotes this one."
    )
    session.add(
        Model(
            version=version,
            task=task,
            algorithm=algo,
            target=spec["target"]["channel"],
            horizon_h=spec["horizon_h"],
            features_json=json.dumps(columns),
            train_start=_utc(labeled["t"].min().to_pydatetime()),
            train_end=_utc(labeled["valid_at"].max().to_pydatetime()),
            train_sources=",".join(sources)[:200],
            n_train=int(len(labeled)),
            metrics_json=json.dumps(metrics),
            baseline_metrics_json=json.dumps({"persistence": base}),
            status=status,
            artifact_path=str(path),
            notes=" ".join(notes),
            created_at=created,
        )
    )
    session.commit()
    return {"version": version, "status": status, "algorithm": algo, "n_train": int(len(labeled)),
            "walk_forward": metrics, "persistence": base}  # fmt: skip


# -------------------------------------------------------------------------------- promote
def promote(session: Session, version: str) -> dict[str, Any]:
    """Make ``version`` the champion of its task; the previous champion is retired."""
    model = session.get(Model, version)
    if model is None:
        raise JobError(f"unknown model version '{version}'")
    stamp = f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}"
    retired = None
    for old in session.scalars(
        select(Model).where(Model.task == model.task, Model.status == "champion")
    ):
        if old.version == version:
            return {"version": version, "status": "champion", "retired": None}
        old.status = "retired"
        old.notes = f"{old.notes or ''} {stamp} retired: {version} promoted.".strip()
        retired = old.version
    model.status = "champion"
    model.notes = f"{model.notes or ''} {stamp} promoted to champion by request.".strip()
    session.commit()
    return {"version": version, "status": "champion", "retired": retired}


def champion(session: Session, task: str) -> Model | None:
    return session.scalar(select(Model).where(Model.task == task, Model.status == "champion"))


# ------------------------------------------------------------------------------- forecast
def forecast(
    session: Session, task: str, backfill_steps: int = 0, now: datetime | None = None
) -> dict[str, Any]:
    """Store predictions for the latest issue time of every node (and earlier ones on request).

    The champion and ``persistence-v1`` are written for the same node, issue time and horizon,
    so the scorecard can compare them once the outcome is measured. Re-running changes nothing.
    """
    model = champion(session, task)
    if model is None:
        raise JobError(f"task '{task}' has no champion model. Train one first.")
    bundle = joblib.load(model.artifact_path)
    spec = bundle["spec"]  # the spec the model was trained with, not today's file
    grids, _ = load_grids(session, spec, now=now)
    rows = build_rows(grids, spec)
    if rows.empty:
        raise JobError(f"task '{task}': no data to forecast from")
    keep = int(backfill_steps) + 1
    rows = rows.groupby("node_id", sort=False).tail(keep).reset_index(drop=True)
    X = align_columns(rows, bundle["columns"]).to_numpy(dtype=float)
    pred = np.asarray(bundle["model"].predict(X), dtype=float)
    q = bundle["q"] if np.isfinite(bundle["q"]) else None
    event = spec["event"]
    p_event = (
        event_probability(pred, bundle["residuals"], event["threshold"], event["direction"])
        if event
        else [None] * len(rows)
    )
    for node in rows["node_id"].unique():
        if session.get(Node, node) is None:
            session.add(Node(node_id=node, role="unknown", name=node))
    session.flush()
    records = []
    for i, row in rows.iterrows():
        common = {
            "issued_at": row["t"].to_pydatetime(),
            "valid_at": row["valid_at"].to_pydatetime(),
            "horizon_h": spec["horizon_h"],
            "node_id": row["node_id"],
            "target": spec["target"]["channel"],
        }
        p = None if p_event[i] is None or np.isnan(p_event[i]) else float(p_event[i])
        records.append(
            {
                **common,
                "yhat": float(pred[i]),
                "lo": float(pred[i] - q) if q is not None else None,
                "hi": float(pred[i] + q) if q is not None else None,
                "p_event": p,
                "model_version": model.version,
            }
        )
        now_value = float(row["y_now"])
        records.append(
            {
                **common,
                "yhat": now_value,
                "lo": None,
                "hi": None,
                "p_event": float(bool(event_hit(now_value, spec))) if event else None,
                "model_version": PERSISTENCE_VERSION,
            }
        )
    written = 0
    for start in range(0, len(records), 500):
        stmt = (
            sqlite_insert(Prediction)
            .values(records[start : start + 500])
            .on_conflict_do_nothing(
                index_elements=["model_version", "node_id", "target", "issued_at", "horizon_h"]
            )
        )
        written += session.execute(stmt).rowcount
    session.commit()
    latest = rows.groupby("node_id")["t"].max()
    return {
        "task": task,
        "model_version": model.version,
        "nodes": len(latest),
        "issue_times": int(rows["t"].nunique()),
        "written": int(written),
        "already_stored": int(len(records) - written),
        "latest_issue": latest.max().isoformat(),
    }


# ------------------------------------------------------------------------------ scorecard
def scorecard(session: Session, task: str, now: datetime | None = None) -> dict[str, Any]:
    """Realized error of stored predictions whose outcome has since been measured.

    Only predictions whose outcome falls after the end of the model's training data count, so
    the champion is never graded on an outcome it was fitted to. Persistence is graded on
    exactly the same rows.
    """
    spec = load_task(task)
    model = champion(session, task)
    out: dict[str, Any] = {
        "task": task,
        "target": spec["target"]["channel"],
        "unit": spec["target"]["unit"],
        "model_version": model.version if model else None,
        "rows": [],
        "n_scored": 0,
        "note": "",
    }
    if model is None:
        out["note"] = "No champion model yet."
        return out
    train_end = _utc(model.train_end)
    preds = session.execute(
        select(
            Prediction.model_version,
            Prediction.node_id,
            Prediction.issued_at,
            Prediction.valid_at,
            Prediction.yhat,
            Prediction.lo,
            Prediction.hi,
            Prediction.p_event,
        ).where(
            Prediction.target == spec["target"]["channel"],
            Prediction.horizon_h == spec["horizon_h"],
            Prediction.model_version.in_([model.version, PERSISTENCE_VERSION]),
            Prediction.valid_at > train_end,
        )
    ).all()
    df = pd.DataFrame(
        preds, columns=["version", "node_id", "issued_at", "valid_at", "yhat", "lo", "hi", "p"]
    )
    if df.empty:
        out["note"] = "No stored prediction reaches past the model's training period yet."
        return out
    df["valid_at"] = pd.to_datetime(df["valid_at"], utc=True)
    df["issued_at"] = pd.to_datetime(df["issued_at"], utc=True)
    grids, _ = load_grids(session, spec, now=now)
    target = spec["target"]["channel"]
    actual = pd.concat(
        [g[target].rename("y").reset_index().assign(node_id=n) for n, g in grids.items()]
        or [pd.DataFrame({"t": [], "y": [], "node_id": []})],
        ignore_index=True,
    ).rename(columns={"t": "valid_at"})
    df = df.merge(actual, on=["node_id", "valid_at"], how="inner").dropna(subset=["y"])
    # Same rows for both: keep (node, issue time) pairs that both versions predicted.
    both = df.groupby(["node_id", "issued_at"])["version"].transform("nunique") == 2
    df = df[both]
    out["n_scored"] = int(len(df) // 2)
    if df.empty:
        out["note"] = (
            "Predictions are stored but none of their outcomes has been measured yet. "
            f"Come back after one horizon ({spec['horizon_h']:g} h)."
        )
        return out
    mae_ref = None
    for version in (PERSISTENCE_VERSION, model.version):
        part = df[df["version"] == version]
        m = regression_metrics(part["y"].to_numpy(), part["yhat"].to_numpy())
        if version == PERSISTENCE_VERSION:
            mae_ref = m["mae"]
        row: dict[str, Any] = {
            "model_version": version,
            "is_baseline": version == PERSISTENCE_VERSION,
            "n": m["n"],
            "mae": m["mae"],
            "bias": m["bias"],
            "skill_vs_persistence": skill(m["mae"], mae_ref),
            "interval_coverage": None,
        }
        if part["lo"].notna().any():
            inside = (part["y"] >= part["lo"]) & (part["y"] <= part["hi"])
            row["interval_coverage"] = float(inside[part["lo"].notna()].mean())
        if spec["event"]:
            ev = event_metrics(event_hit(part["y"].to_numpy(), spec), part["p"].to_numpy())
            row.update(
                event_positives=ev["positives"],
                event_precision=ev["precision"],
                event_recall=ev["recall"],
            )
        out["rows"].append(row)
    out["first_issue"] = df["issued_at"].min().isoformat()
    out["last_issue"] = df["issued_at"].max().isoformat()
    out["first_valid"] = df["valid_at"].min().isoformat()
    out["note"] = (
        f"Out-of-sample only: forecasts for times after {train_end:%Y-%m-%d %H:%M} UTC, the end "
        "of the champion's training data."
    )
    return out
