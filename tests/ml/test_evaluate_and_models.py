"""Metrics, verdict wording, conformal intervals and event probabilities."""

from __future__ import annotations

import numpy as np
import pytest

from ml.evaluate import (
    event_metrics,
    paired_mae_difference,
    regression_metrics,
    skill,
    verdict_sentence,
)
from ml.models import conformal_quantile, event_probability, make_model


def test_regression_metrics_by_hand() -> None:
    m = regression_metrics(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 3.0, 3.0, 2.0]))
    assert m["n"] == 4 and m["mae"] == pytest.approx(0.75) and m["bias"] == pytest.approx(-0.25)
    assert m["rmse"] == pytest.approx(np.sqrt(5 / 4)) and m["r2"] == pytest.approx(0.0)
    with_nan = regression_metrics(np.array([1.0, 2.0]), np.array([np.nan, 2.5]))
    assert with_nan["n"] == 1 and with_nan["mae"] == 0.5 and with_nan["r2"] is None
    assert regression_metrics(np.array([]), np.array([]))["mae"] is None


def test_skill_is_relative_to_the_reference() -> None:
    assert skill(0.5, 1.0) == 0.5 and skill(1.5, 1.0) == -0.5
    assert skill(None, 1.0) is None and skill(1.0, 0.0) is None


def test_event_metrics_by_hand() -> None:
    actual = np.array([True, True, False, False, True])
    p = np.array([0.9, 0.2, 0.7, 0.1, np.nan])
    m = event_metrics(actual, p)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1) and m["n"] == 4
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["brier"] == pytest.approx((0.01 + 0.64 + 0.49 + 0.01) / 4)
    none = event_metrics(np.array([False, False]), np.array([0.1, 0.2]))
    assert none["precision"] is None and none["recall"] is None and none["positives"] == 0


def _days(n_days: int, per_day: int) -> np.ndarray:
    return np.repeat(np.arange(n_days), per_day)


def test_paired_difference_verdicts() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 300)
    days = _days(30, 10)
    good, poor = y + rng.normal(0, 0.1, 300), y + rng.normal(0, 1.0, 300)
    assert paired_mae_difference(y, good, poor, days)["verdict"] == "beats"
    assert paired_mae_difference(y, poor, good, days)["verdict"] == "worse"
    # Better on even days, equally worse on odd days: no demonstrable difference overall.
    swing = np.where(days % 2 == 0, 0.9, 1.1)
    same = paired_mae_difference(y, y + swing, y + 1.0, days)
    assert same["diff"] == pytest.approx(0.0, abs=1e-9)
    assert same["verdict"] == "no_difference" and same["lo"] < 0 < same["hi"]
    short = paired_mae_difference(y[:40], good[:40], poor[:40], _days(4, 10))
    assert short["verdict"] == "too_few_days" and short["lo"] is None
    assert paired_mae_difference(y, np.full(300, np.nan), poor, days)["verdict"] == "not_scored"


def test_verdict_sentences_have_the_same_shape_for_good_and_bad_news() -> None:
    base = {"diff": -0.2, "lo": -0.3, "hi": -0.1, "n_days": 30}
    win = verdict_sentence("ridge", "persistence", {**base, "verdict": "beats"}, "degC", 0.9)
    tied = {**base, "hi": 0.1, "verdict": "no_difference"}
    tie = verdict_sentence("ridge", "seasonal_naive", tied, "degC", 0.9)
    loss = verdict_sentence(
        "ridge", "persistence", {"diff": 0.2, "lo": 0.1, "hi": 0.3, "verdict": "worse"}, "degC", 0.9
    )
    assert win == "ridge beats persistence: MAE difference -0.2 degC (90% interval -0.3 to -0.1)."
    assert tie.startswith("ridge does NOT demonstrably beat seasonal naive:") and "zero" in tie
    assert loss.startswith("ridge does NOT beat persistence, it is worse:")


def test_conformal_quantile_and_its_small_sample_limit() -> None:
    r = np.arange(1, 101, dtype=float)  # |residuals| 1..100
    assert conformal_quantile(r, 0.90) == 91.0  # ceil(101 * 0.9) = 91st smallest
    assert conformal_quantile(-r, 0.90) == 91.0
    assert conformal_quantile(np.array([1.0, 2.0]), 0.90) == float("inf")


def test_event_probability_is_the_share_of_residuals_that_cross() -> None:
    residuals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    pred = np.array([9.0, 10.0, 13.0])
    above = event_probability(pred, residuals, threshold=10.0, direction="above")
    assert above.tolist() == [0.4, 0.6, 1.0]  # y = pred + r >= 10
    below = event_probability(pred, residuals, threshold=10.0, direction="below")
    assert below.tolist() == [0.8, 0.6, 0.0]  # y = pred + r <= 10
    assert np.isnan(event_probability(pred, np.array([]), 10.0, "above")).all()


@pytest.mark.parametrize("algo", ["ridge", "random_forest", "gradient_boosting"])
def test_every_algorithm_fits_with_missing_values_and_is_reproducible(algo: str) -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (200, 4))
    y = 2 * X[:, 0] - X[:, 1] + rng.normal(0, 0.1, 200)
    X[::7, 2] = np.nan
    X[:, 3] = np.nan  # a channel that never reported
    params = {"n_estimators": 20} if algo == "random_forest" else {}
    a = make_model(algo, params, seed=7).fit(X[:150], y[:150]).predict(X[150:])
    b = make_model(algo, params, seed=7).fit(X[:150], y[:150]).predict(X[150:])
    assert np.array_equal(a, b)
    assert np.abs(a - y[150:]).mean() < np.abs(y[150:] - y[:150].mean()).mean()


def test_unknown_algorithm_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown algorithm"):
        make_model("deep_magic", {}, 1)
