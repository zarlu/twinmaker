"""The model ladder behind one constructor, plus intervals and event probabilities.

Algorithms (all scikit-learn, no extra system libraries): ``ridge`` (standardized inputs),
``random_forest`` and ``gradient_boosting`` (histogram gradient boosting). Every setting comes
from the task YAML; there is no hyperparameter search. Missing feature values are replaced by
the training median inside the pipeline, so nothing from a test fold leaks into the fill value.

Uncertainty comes from out-of-sample walk-forward residuals (``y - prediction``):
  * interval: ``prediction +/- q`` where q is the split-conformal quantile of ``|residual|``;
  * event probability: the share of residuals that, added to the prediction, cross the task's
    threshold. Both are only as good as the assumption that future errors look like past ones.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_model(algo: str, params: dict[str, Any], seed: int) -> Pipeline:
    """Imputer (+ scaler for ridge) + estimator."""
    steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if algo == "ridge":
        steps += [("scale", StandardScaler()), ("model", Ridge(**params))]
    elif algo == "random_forest":
        steps.append(("model", RandomForestRegressor(random_state=seed, n_jobs=1, **params)))
    elif algo == "gradient_boosting":
        steps.append(("model", HistGradientBoostingRegressor(random_state=seed, **params)))
    else:
        raise ValueError(f"unknown algorithm {algo!r}")
    return Pipeline(steps)


def conformal_quantile(residuals: np.ndarray, level: float = 0.90) -> float:
    """ceil((n + 1) * level)-th smallest absolute residual; infinite when n is too small."""
    r = np.sort(np.abs(np.asarray(residuals, dtype=float)))
    n = len(r)
    k = math.ceil((n + 1) * level)
    return float(r[k - 1]) if 0 < k <= n else float("inf")


def event_probability(
    pred: np.ndarray, residuals: np.ndarray, threshold: float, direction: str
) -> np.ndarray:
    """P(event) per prediction from the empirical residual distribution."""
    pred = np.asarray(pred, dtype=float)
    r = np.sort(np.asarray(residuals, dtype=float))
    if len(r) == 0:
        return np.full(len(pred), np.nan)
    # y = pred + r crosses the threshold when r is beyond (threshold - pred).
    need = threshold - pred
    if direction == "above":
        return 1.0 - np.searchsorted(r, need, side="left") / len(r)
    return np.searchsorted(r, need, side="right") / len(r)
