"""Chronological splits only: walk-forward folds, then one final hold-out.

Units are distinct issue times, so all nodes at one instant are on the same side.
  * Final hold-out: the last ``holdout_frac`` of issue times, evaluated once per run.
  * Walk-forward: the earlier issue times are cut into ``n_folds + 1`` consecutive blocks; fold k
    tests block k + 1 and trains on what came before.
A training row is allowed only when its label was already known at the first test issue time
(``valid_at < test start``), which leaves a gap of one horizon between train and test.
There is no random split anywhere in this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


class SplitError(Exception):
    """A split breaks chronology."""


@dataclass(frozen=True)
class Fold:
    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_start: pd.Timestamp


@dataclass
class SplitPlan:
    folds: list[Fold] = field(default_factory=list)
    holdout: Fold | None = None
    skipped: list[str] = field(default_factory=list)


def _fold(name: str, rows: pd.DataFrame, test_mask: np.ndarray) -> Fold:
    test_start = rows.loc[test_mask, "t"].min()
    train_mask = (rows["valid_at"] < test_start).to_numpy()
    return Fold(name, np.flatnonzero(train_mask), np.flatnonzero(test_mask), test_start)


def make_splits(rows: pd.DataFrame, cfg: dict[str, Any]) -> SplitPlan:
    """Build the plan for labeled rows (needs a RangeIndex and columns ``t``, ``valid_at``)."""
    plan = SplitPlan()
    times = np.sort(rows["t"].unique())
    min_train = int(cfg.get("min_train_rows", 1))
    n_hold = int(np.ceil(float(cfg["holdout_frac"]) * len(times)))
    if len(times) < 4 or n_hold < 1 or n_hold >= len(times):
        plan.skipped.append(f"only {len(times)} distinct issue times: nothing to split")
        return plan
    hold_start = pd.Timestamp(times[-n_hold])
    in_hold = (rows["t"] >= hold_start).to_numpy()
    hold = _fold("holdout", rows, in_hold)
    if len(hold.train_idx) < min_train:
        plan.skipped.append(
            f"hold-out: {len(hold.train_idx)} training rows, fewer than min_train_rows={min_train}"
        )
    else:
        plan.holdout = hold

    pre_times = times[:-n_hold]
    blocks = np.array_split(pre_times, int(cfg["n_folds"]) + 1)
    for k, block in enumerate(blocks[1:], start=1):
        if len(block) == 0:
            continue
        mask = rows["t"].isin(block).to_numpy()
        fold = _fold(f"fold{k}", rows, mask)
        if len(fold.train_idx) < min_train:
            plan.skipped.append(
                f"fold{k}: {len(fold.train_idx)} training rows, fewer than "
                f"min_train_rows={min_train}"
            )
            continue
        plan.folds.append(fold)
    for f in [*plan.folds, *([plan.holdout] if plan.holdout else [])]:
        assert_chronological(rows, f)
    return plan


def assert_chronological(rows: pd.DataFrame, fold: Fold) -> None:
    """Raise unless every training label was known before the first test issue time."""
    train, test = rows.iloc[fold.train_idx], rows.iloc[fold.test_idx]
    if train.empty or test.empty:
        raise SplitError(f"{fold.name}: empty train or test side")
    if train["valid_at"].max() >= test["t"].min():
        raise SplitError(f"{fold.name}: a training label is dated at or after a test issue time")
    if set(fold.train_idx) & set(fold.test_idx):
        raise SplitError(f"{fold.name}: a row is on both sides")
