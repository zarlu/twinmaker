"""The two rules that keep a backtest honest: no future in features, no future in training."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.config import load_task
from ml.data import load_grids, to_grids
from ml.features import align_columns, build_rows, feature_columns
from ml.splits import Fold, SplitError, assert_chronological, make_splits
from tests.ml.conftest import NOW, TASK


def _grid(n: int = 200) -> dict[str, pd.DataFrame]:
    index = pd.date_range("2026-10-01", periods=n, freq="1h", tz="UTC", name="t")
    rng = np.random.default_rng(0)
    return {
        "a": pd.DataFrame(
            {
                "water_temp": 20 + rng.normal(0, 1, n).cumsum() * 0.1,
                "air_temp": rng.normal(22, 3, n),
            },
            index=index,
        )
    }  # fmt: skip


def test_label_is_the_target_h_steps_later_and_baselines_use_the_past() -> None:
    spec = load_task(TASK)  # horizon 3 steps, seasonal period 24
    grids = _grid()
    rows = build_rows(grids, spec)
    series = grids["a"]["water_temp"]
    row = rows.iloc[100]
    t = row["t"]
    assert row["valid_at"] == t + pd.Timedelta(hours=3)
    assert row["y"] == series[t + pd.Timedelta(hours=3)]
    assert row["y_now"] == series[t] == row["f_target_lag0"]
    assert row["f_target_lag24"] == series[t - pd.Timedelta(hours=24)]
    assert row["y_seasonal"] == series[t + pd.Timedelta(hours=3 - 24)]
    assert row["f_air_temp_lag3"] == grids["a"]["air_temp"][t - pd.Timedelta(hours=3)]
    assert rows["y"].isna().sum() == 3  # the newest 3 issue times have no outcome yet


def test_changing_the_future_never_changes_a_feature() -> None:
    """Leakage guard: features at issue time t must not depend on any value after t."""
    spec = load_task(TASK)
    grids = _grid()
    before = build_rows(grids, spec)
    cut = grids["a"].index[120]
    tampered = {"a": grids["a"].copy()}
    tampered["a"].loc[tampered["a"].index > cut, :] = 999.0
    after = build_rows(tampered, spec)
    cols = feature_columns(before)
    past = before["t"] <= cut
    pd.testing.assert_frame_equal(before.loc[past, cols], after.loc[past, cols])
    assert not before.loc[~past, cols].equals(after.loc[~past, cols])  # the check can fail


def test_partial_bucket_is_left_out_and_short_gaps_are_filled_from_the_past() -> None:
    spec = load_task(TASK)
    ts = pd.to_datetime(
        ["2026-10-01T00:10Z", "2026-10-01T01:10Z", "2026-10-01T04:10Z", "2026-10-01T05:10Z"]
    )
    rows = pd.DataFrame({"ts": ts, "node_id": "a", "channel": "water_temp",
                         "value": [1.0, 2.0, 5.0, 6.0], "source": "sensor"})  # fmt: skip
    grid = to_grids(rows, spec, now=pd.Timestamp("2026-10-01T05:30Z"))["a"]
    assert grid.index[-1] == pd.Timestamp("2026-10-01T04:00Z")  # the 05:00 bucket is still open
    # 02:00 and 03:00 are missing: max_gap_steps=2 carries 01:00 forward, never 04:00 backward.
    assert grid["water_temp"].tolist() == [1.0, 2.0, 2.0, 2.0, 5.0]


def test_only_good_rows_and_no_labels_feed_features(filled) -> None:
    spec = load_task(TASK)
    grids, sources = load_grids(filled, spec, now=NOW)
    assert sources == ["simulator"] and set(grids) == {"node-downstream", "node-intake"}
    # Injected 20 degC spikes are flagged suspect by QC and must not reach the grid.
    assert grids["node-intake"]["water_temp"].max() < 35


def test_splits_are_chronological_with_a_horizon_gap(filled) -> None:
    spec = load_task(TASK)
    rows = build_rows(load_grids(filled, spec, now=NOW)[0], spec)
    labeled = rows[rows["y"].notna()].reset_index(drop=True)
    plan = make_splits(labeled, spec["split"])
    assert [f.name for f in plan.folds] == ["fold1", "fold2", "fold3"] and plan.holdout
    hold_times = set(labeled.iloc[plan.holdout.test_idx]["t"])
    for fold in [*plan.folds, plan.holdout]:
        train, test = labeled.iloc[fold.train_idx], labeled.iloc[fold.test_idx]
        assert train["valid_at"].max() < test["t"].min()
        if fold.name != "holdout":
            assert not hold_times & set(test["t"])  # hold-out rows are never a walk-forward test
    share = labeled.iloc[plan.holdout.test_idx]["t"].nunique() / labeled["t"].nunique()
    assert 0.19 < share < 0.22


def test_a_fold_that_trains_on_the_future_is_refused() -> None:
    rows = pd.DataFrame({"t": pd.date_range("2026-10-01", periods=10, freq="1h", tz="UTC")})
    rows["valid_at"] = rows["t"] + pd.Timedelta(hours=3)
    leaky = Fold("leaky", np.array([0, 1, 2, 3]), np.array([5, 6]), rows["t"][5])
    with pytest.raises(SplitError, match="at or after a test issue time"):
        assert_chronological(rows, leaky)  # row 3's label is dated 06:00, test starts 05:00
    assert_chronological(rows, Fold("ok", np.array([0, 1]), np.array([5, 6]), rows["t"][5]))


def test_too_little_data_is_skipped_not_guessed() -> None:
    rows = build_rows(_grid(60), load_task(TASK))
    labeled = rows[rows["y"].notna()].reset_index(drop=True)
    plan = make_splits(labeled, {"n_folds": 3, "holdout_frac": 0.2, "min_train_rows": 50})
    assert plan.holdout is None and not plan.folds and len(plan.skipped) == 4


def test_align_columns_keeps_a_stored_models_layout() -> None:
    rows = pd.DataFrame({"f_a": [1.0], "f_node_x": [1.0]})
    out = align_columns(rows, ["f_node_y", "f_a", "f_b"])
    assert list(out.columns) == ["f_node_y", "f_a", "f_b"]
    assert out.iloc[0]["f_node_y"] == 0.0 and np.isnan(out.iloc[0]["f_b"])
