"""Backtest a task: every model next to its baselines, walk-forward and on the final hold-out.

Baselines are always scored: ``persistence`` (the value at issue time carried forward) and, when
the task sets ``seasonal_period_steps``, ``seasonal_naive`` (the value one full cycle before the
valid time). The result is plain data (``metrics.json``); ``ml.report`` turns it into Markdown.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from ml.config import event_hit
from ml.evaluate import (
    event_metrics,
    paired_mae_difference,
    regression_metrics,
    skill,
    verdict_sentence,
)
from ml.features import feature_columns
from ml.models import conformal_quantile, event_probability, make_model
from ml.splits import Fold, SplitPlan, make_splits

BASELINE_MEANING = {
    "persistence": "the latest value at issue time, carried forward unchanged to the valid time",
    "seasonal_naive": "the value one full seasonal cycle before the valid time",
}


def baselines_of(spec: dict[str, Any]) -> list[str]:
    return ["persistence", *(["seasonal_naive"] if spec.get("seasonal_period_steps") else [])]


def baseline_prediction(rows: pd.DataFrame, name: str) -> np.ndarray:
    column = {"persistence": "y_now", "seasonal_naive": "y_seasonal"}[name]
    return rows[column].to_numpy(dtype=float)


def fit_predict(
    spec: dict[str, Any], algo: str, rows: pd.DataFrame, fold: Fold, columns: list[str]
) -> tuple[Any, np.ndarray]:
    """Fit ``algo`` on the fold's training rows, predict its test rows."""
    train, test = rows.iloc[fold.train_idx], rows.iloc[fold.test_idx]
    model = make_model(algo, spec["models"][algo], int(spec["seed"]))
    model.fit(train[columns].to_numpy(dtype=float), train["y"].to_numpy(dtype=float))
    return model, np.asarray(model.predict(test[columns].to_numpy(dtype=float)), dtype=float)


def walk_forward(
    spec: dict[str, Any], algo: str, rows: pd.DataFrame, plan: SplitPlan, columns: list[str]
) -> pd.DataFrame:
    """Out-of-sample predictions of every walk-forward fold: columns ``idx``, ``pred``."""
    parts = []
    for fold in plan.folds:
        _, pred = fit_predict(spec, algo, rows, fold, columns)
        parts.append(pd.DataFrame({"idx": fold.test_idx, "pred": pred}))
    if not parts:
        return pd.DataFrame({"idx": [], "pred": []})
    return pd.concat(parts, ignore_index=True)


def _score_block(
    spec: dict[str, Any],
    test: pd.DataFrame,
    preds: dict[str, np.ndarray],
    residuals: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Metrics of one evaluation block for every predictor in ``preds``."""
    y = test["y"].to_numpy(dtype=float)
    days = test["valid_at"].dt.floor("D").to_numpy()
    unit = spec["target"]["unit"]
    cfg = spec["eval"]
    block: dict[str, Any] = {
        "n": int(len(test)),
        "n_days": int(len(np.unique(days))),
        "first_valid_at": test["valid_at"].min().isoformat() if len(test) else None,
        "last_valid_at": test["valid_at"].max().isoformat() if len(test) else None,
        "predictors": {},
    }
    base_names = baselines_of(spec)
    mae_persistence = regression_metrics(y, preds["persistence"])["mae"]
    actual_event = event_hit(y, spec) if spec["event"] else None
    if spec["event"]:
        block["event_positives"] = int(actual_event.sum())
    for name, pred in preds.items():
        entry: dict[str, Any] = {
            "is_baseline": name in base_names,
            **regression_metrics(y, pred),
        }
        entry["skill_vs_persistence"] = skill(entry["mae"], mae_persistence)
        entry["vs"] = {}
        for ref in base_names:
            if ref == name:
                continue
            res = paired_mae_difference(
                y, pred, preds[ref], days, int(cfg["n_boot"]), float(cfg["ci"]), int(spec["seed"])
            )
            res["sentence"] = verdict_sentence(name, ref, res, unit, float(cfg["ci"]))
            entry["vs"][ref] = res
        if spec["event"]:
            if name in base_names:
                p = np.where(np.isnan(pred), np.nan, event_hit(pred, spec).astype(float))
            else:
                p = event_probability(
                    pred, residuals[name], spec["event"]["threshold"], spec["event"]["direction"]
                )
            entry["event"] = event_metrics(actual_event, p)
        block["predictors"][name] = entry
    return block


def run_backtest(spec: dict[str, Any], rows: pd.DataFrame, sources: list[str]) -> dict[str, Any]:
    """Score every baseline and model of the task. ``rows`` come from ``features.build_rows``."""
    labeled = rows[rows["y"].notna()].reset_index(drop=True)
    columns = feature_columns(labeled)
    plan = make_splits(labeled, spec["split"]) if len(labeled) else SplitPlan()
    level = float(spec["intervals"]["level"])
    result: dict[str, Any] = {
        "task": spec["name"],
        "description": spec["description"],
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "target": spec["target"],
        "resolution": spec["resolution"],
        "horizon_steps": spec["horizon_steps"],
        "horizon_h": spec["horizon_h"],
        "event": spec["event"],
        "sources": sources,
        "simulated": "simulator" in sources,
        "nodes": sorted(labeled["node_id"].unique()) if len(labeled) else [],
        "n_rows": int(len(labeled)),
        "features": columns,
        "baseline_meaning": {b: BASELINE_MEANING[b] for b in baselines_of(spec)},
        "ci": float(spec["eval"]["ci"]),
        "interval_level": level,
        "folds": [],
        "skipped": list(plan.skipped),
        "warnings": [],
        "blocks": {},
        "importance": {},
        "best_model": None,
    }
    if len(labeled):
        result["data_start"] = labeled["t"].min().isoformat()
        result["data_end"] = labeled["valid_at"].max().isoformat()
    for fold in [*plan.folds, *([plan.holdout] if plan.holdout else [])]:
        tr, te = labeled.iloc[fold.train_idx], labeled.iloc[fold.test_idx]
        result["folds"].append(
            {
                "fold": fold.name,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "train_last_valid_at": tr["valid_at"].max().isoformat(),
                "test_first_issue": te["t"].min().isoformat(),
                "test_last_issue": te["t"].max().isoformat(),
            }
        )
    if not plan.folds:
        result["warnings"].append(
            "No walk-forward fold could be built: there is not enough data to backtest. "
            "Collect more data or lower `split.min_train_rows`."
        )
        return result

    # ---- walk-forward: pooled out-of-sample predictions of every fold
    wf_idx = np.concatenate([f.test_idx for f in plan.folds])
    wf_test = labeled.iloc[wf_idx]
    wf_preds = {b: baseline_prediction(wf_test, b) for b in baselines_of(spec)}
    residuals: dict[str, np.ndarray] = {}
    for algo in spec["models"]:
        out = walk_forward(spec, algo, labeled, plan, columns)
        wf_preds[algo] = out["pred"].to_numpy()
        residuals[algo] = wf_test["y"].to_numpy(dtype=float) - wf_preds[algo]
    # Walk-forward event probabilities use each model's own walk-forward residuals, which is
    # mildly optimistic; the hold-out block below is the clean estimate.
    result["blocks"]["walk_forward"] = _score_block(spec, wf_test, wf_preds, residuals)
    wf = result["blocks"]["walk_forward"]["predictors"]
    result["best_model"] = min(spec["models"], key=lambda a: wf[a]["mae"])

    # ---- final hold-out: fitted once on everything before it
    if plan.holdout is not None:
        hold_test = labeled.iloc[plan.holdout.test_idx]
        hold_preds = {b: baseline_prediction(hold_test, b) for b in baselines_of(spec)}
        fitted = {}
        for algo in spec["models"]:
            fitted[algo], hold_preds[algo] = fit_predict(spec, algo, labeled, plan.holdout, columns)
        block = _score_block(spec, hold_test, hold_preds, residuals)
        y_hold = hold_test["y"].to_numpy(dtype=float)
        for algo in spec["models"]:
            q = conformal_quantile(residuals[algo], level)
            inside = np.abs(y_hold - hold_preds[algo]) <= q
            block["predictors"][algo]["interval_half_width"] = q if np.isfinite(q) else None
            block["predictors"][algo]["interval_coverage"] = float(inside.mean())
        result["blocks"]["holdout"] = block
        best = result["best_model"]
        imp = permutation_importance(
            fitted[best],
            hold_test[columns].to_numpy(dtype=float),
            y_hold,
            scoring="neg_mean_absolute_error",
            n_repeats=3,
            random_state=int(spec["seed"]),
        )
        result["importance"] = {
            "model": best,
            "block": "holdout",
            "meaning": "increase in MAE when the feature is shuffled (larger = relied on more)",
            "features": dict(
                sorted(
                    ((c, float(v)) for c, v in zip(columns, imp.importances_mean, strict=True)),
                    key=lambda kv: -kv[1],
                )
            ),
        }
    result["warnings"] += _warnings(spec, result)
    return result


def _warnings(spec: dict[str, Any], result: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if result["simulated"]:
        out.append(
            "SIMULATED DATA: rows with source='simulator' were used. These scores validate the "
            "pipeline only; they say nothing about the real system."
        )
    first = result["folds"][0]["n_train"] if result["folds"] else 0
    if first < 200:
        out.append(f"The first fold trains on only {first} rows; scores will be noisy.")
    for name, block in result["blocks"].items():
        if block["n_days"] < 14:
            out.append(
                f"{name}: the test window covers {block['n_days']} days. Verdicts from so short "
                "a window may not hold in other seasons."
            )
        if spec["event"] and block.get("event_positives", 0) < 10:
            out.append(
                f"{name}: only {block.get('event_positives', 0)} event rows; precision and "
                "recall are unreliable."
            )
        for algo in spec["models"]:
            r2 = block["predictors"][algo]["r2"]
            if r2 is not None and r2 > 0.999:
                out.append(
                    f"{name}: {algo} has R2 {r2:.4f}. Check for a feature that contains the "
                    "answer (leakage) before trusting it."
                )
    return out
