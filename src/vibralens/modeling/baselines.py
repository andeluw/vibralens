"""Simple, explicit RUL reference models."""

from __future__ import annotations

from typing import Dict

import numpy as np


class BaselineError(ValueError):
    """Raised when an age-only baseline cannot be fitted or evaluated."""


def _numeric_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise BaselineError("{} must be a non-empty one-dimensional array".format(name))
    if not np.isfinite(array).all():
        raise BaselineError("{} contains non-finite values".format(name))
    return array


class AgeOnlyRegressor:
    """Estimate RUL from elapsed age and median train lifetime per condition."""

    def __init__(self) -> None:
        self.lifetime_by_condition: Dict[int, float] = {}

    def fit(
        self,
        ages_minutes: np.ndarray,
        condition_ids: np.ndarray,
        y: np.ndarray,
        bearing_ids: np.ndarray,
    ) -> "AgeOnlyRegressor":
        ages = _numeric_vector(ages_minutes, "ages_minutes")
        conditions = np.asarray(condition_ids, dtype=np.int64)
        targets = _numeric_vector(y, "y")
        bearings = np.asarray(bearing_ids, dtype=str)
        if conditions.ndim != 1 or bearings.ndim != 1:
            raise BaselineError("condition_ids and bearing_ids must be vectors")
        if len({ages.size, conditions.size, targets.size, bearings.size}) != 1:
            raise BaselineError("baseline input arrays must have matching lengths")
        if np.any(ages < 0.0) or np.any(targets < 0.0):
            raise BaselineError("ages and RUL targets cannot be negative")

        lifetimes_by_condition: Dict[int, list] = {}
        for bearing_id in sorted(set(bearings.tolist())):
            mask = bearings == bearing_id
            bearing_conditions = set(conditions[mask].tolist())
            if len(bearing_conditions) != 1:
                raise BaselineError(
                    "bearing {} has multiple conditions".format(bearing_id)
                )
            lifetime_values = ages[mask] + targets[mask]
            if not np.allclose(
                lifetime_values,
                lifetime_values[0],
                rtol=0.0,
                atol=1e-9,
            ):
                raise BaselineError(
                    "bearing {} has inconsistent lifetime labels".format(bearing_id)
                )
            condition_id = int(next(iter(bearing_conditions)))
            lifetimes_by_condition.setdefault(condition_id, []).append(
                float(lifetime_values[0])
            )

        self.lifetime_by_condition = {
            condition_id: float(np.median(lifetimes))
            for condition_id, lifetimes in sorted(lifetimes_by_condition.items())
        }
        return self

    def predict(
        self,
        ages_minutes: np.ndarray,
        condition_ids: np.ndarray,
    ) -> np.ndarray:
        if not self.lifetime_by_condition:
            raise BaselineError("age-only baseline has not been fitted")
        ages = _numeric_vector(ages_minutes, "ages_minutes")
        conditions = np.asarray(condition_ids, dtype=np.int64)
        if conditions.ndim != 1 or conditions.size != ages.size:
            raise BaselineError("condition_ids must match ages_minutes")
        if np.any(ages < 0.0):
            raise BaselineError("ages_minutes cannot be negative")

        predictions = np.empty(ages.size, dtype=np.float64)
        for index, (age, condition_id) in enumerate(zip(ages, conditions)):
            condition = int(condition_id)
            if condition not in self.lifetime_by_condition:
                raise BaselineError("unsupported condition {}".format(condition))
            predictions[index] = max(
                self.lifetime_by_condition[condition] - float(age),
                0.0,
            )
        return predictions
