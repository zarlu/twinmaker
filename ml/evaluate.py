"""Metrics, bootstrap intervals and plain-word verdicts.

Sensor errors are autocorrelated, so the bootstrap resamples whole UTC days of test rows instead
of single rows. A model "beats" a baseline only when the interval of the paired MAE difference
excludes zero; otherwise the verdict is "no demonstrable difference".
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int | None]:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    ok = ~np.isnan(pred) & ~np.isnan(y)
    n = int(ok.sum())
    if n == 0:
        return {"n": 0, "mae": None, "rmse": None, "bias": None, "r2": None}
    err = pred[ok] - y[ok]
    ss_tot = float(((y[ok] - y[ok].mean()) ** 2).sum())
    return {
        "n": n,
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "bias": float(err.mean()),
        "r2": float(1 - (err**2).sum() / ss_tot) if ss_tot > 0 else None,
    }


def skill(mae: float | None, mae_ref: float | None) -> float | None:
    """1 - MAE / MAE_reference (positive = better than the reference)."""
    if mae is None or mae_ref is None or mae_ref == 0:
        return None
    return float(1 - mae / mae_ref)


def event_metrics(
    actual: np.ndarray, p: np.ndarray, cutoff: float = 0.5
) -> dict[str, float | int | None]:
    """Threshold-event scores for probabilities ``p`` against boolean ``actual``."""
    actual = np.asarray(actual, dtype=bool)
    p = np.asarray(p, dtype=float)
    ok = ~np.isnan(p)
    actual, p = actual[ok], p[ok]
    n, positives = int(len(actual)), int(actual.sum())
    out: dict[str, float | int | None] = {
        "n": n,
        "positives": positives,
        "base_rate": float(positives / n) if n else None,
        "precision": None,
        "recall": None,
        "f1": None,
        "brier": float(((p - actual) ** 2).mean()) if n else None,
        "tp": None,
        "fp": None,
        "fn": None,
        "tn": None,
    }
    if n == 0:
        return out
    flagged = p >= cutoff
    tp = int((flagged & actual).sum())
    fp = int((flagged & ~actual).sum())
    fn = int((~flagged & actual).sum())
    out.update(tp=tp, fp=fp, fn=fn, tn=int(n - tp - fp - fn))
    out["precision"] = float(tp / (tp + fp)) if tp + fp else None
    out["recall"] = float(tp / (tp + fn)) if tp + fn else None
    if out["precision"] is not None and out["recall"] is not None and tp:
        out["f1"] = float(2 * tp / (2 * tp + fp + fn))
    return out


def paired_mae_difference(
    y: np.ndarray,
    pred: np.ndarray,
    pred_ref: np.ndarray,
    days: np.ndarray,
    n_boot: int = 500,
    ci: float = 0.9,
    seed: int = 7,
) -> dict[str, Any]:
    """MAE(model) - MAE(reference) with a day-block bootstrap interval. Negative = model better."""
    y, pred, pred_ref = (np.asarray(a, dtype=float) for a in (y, pred, pred_ref))
    ok = ~np.isnan(pred) & ~np.isnan(pred_ref) & ~np.isnan(y)
    if ok.sum() == 0:
        return {"diff": None, "lo": None, "hi": None, "verdict": "not_scored", "n_days": 0}
    diff = np.abs(pred[ok] - y[ok]) - np.abs(pred_ref[ok] - y[ok])
    per_day = pd.DataFrame({"day": np.asarray(days)[ok], "d": diff}).groupby("day")["d"]
    sums, counts = per_day.sum().to_numpy(), per_day.count().to_numpy()
    point = float(diff.mean())
    if len(sums) < 5:
        return {"diff": point, "lo": None, "hi": None, "verdict": "too_few_days",
                "n_days": int(len(sums))}  # fmt: skip
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(sums), size=(int(n_boot), len(sums)))
    boot = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    lo, hi = np.quantile(boot, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    verdict = "beats" if hi < 0 else "worse" if lo > 0 else "no_difference"
    return {"diff": point, "lo": float(lo), "hi": float(hi), "verdict": verdict,
            "n_days": int(len(sums))}  # fmt: skip


def verdict_sentence(model: str, reference: str, res: dict[str, Any], unit: str, ci: float) -> str:
    """One sentence, same structure for good and bad news."""
    ref = reference.replace("_", " ")
    if res["verdict"] == "not_scored":
        return f"{model} could not be compared with {ref}: no rows scored by both."
    change = f"MAE difference {res['diff']:+.4g} {unit}"
    if res["verdict"] == "too_few_days":
        return (
            f"{model} vs {ref}: {change}, but the test window covers only {res['n_days']} days, "
            "too few for an interval. No conclusion."
        )
    span = f"{100 * ci:.0f}% interval {res['lo']:+.4g} to {res['hi']:+.4g}"
    if res["verdict"] == "beats":
        return f"{model} beats {ref}: {change} ({span})."
    if res["verdict"] == "worse":
        return f"{model} does NOT beat {ref}, it is worse: {change} ({span})."
    return f"{model} does NOT demonstrably beat {ref}: {change} ({span}, includes zero)."
