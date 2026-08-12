"""Bearing-aware RUL evaluation metrics."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import numpy as np


class ModelingMetricError(ValueError):
    """Raised when metric inputs are invalid."""


def _validated_array(
    values: Iterable[object],
    name: str,
    *,
    numeric: bool = True,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64 if numeric else str)
    if array.ndim != 1 or array.size == 0:
        raise ModelingMetricError("{} must be a non-empty one-dimensional array".format(name))
    if numeric and not np.isfinite(array).all():
        raise ModelingMetricError("{} contains non-finite values".format(name))
    return array


def _validate_matching_lengths(arrays: Mapping[str, np.ndarray]) -> None:
    lengths = {array.size for array in arrays.values()}
    if len(lengths) != 1:
        raise ModelingMetricError("metric arrays must have matching lengths")


def evaluate_point_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bearing_ids: np.ndarray,
    ages_minutes: np.ndarray,
) -> Dict[str, object]:
    """Return snapshot and macro-per-bearing point-prediction metrics."""
    truth = _validated_array(y_true, "y_true")
    prediction = _validated_array(y_pred, "y_pred")
    bearings = _validated_array(bearing_ids, "bearing_ids", numeric=False)
    ages = _validated_array(ages_minutes, "ages_minutes")
    _validate_matching_lengths(
        {
            "y_true": truth,
            "y_pred": prediction,
            "bearing_ids": bearings,
            "ages_minutes": ages,
        }
    )
    if np.any(truth < 0.0) or np.any(prediction < 0.0) or np.any(ages < 0.0):
        raise ModelingMetricError("RUL predictions, targets, and ages cannot be negative")

    absolute_errors = np.abs(prediction - truth)
    per_bearing: Dict[str, Dict[str, Optional[float]]] = {}
    bearing_maes = []
    bearing_nmaes = []
    for bearing_id in sorted(set(bearings.tolist())):
        mask = bearings == bearing_id
        mae = float(np.mean(absolute_errors[mask]))
        lifetime = float(np.max(ages[mask] + truth[mask]))
        nmae = None if lifetime == 0.0 else mae / lifetime
        per_bearing[bearing_id] = {
            "mae_minutes": mae,
            "nmae": nmae,
            "observed_lifetime_minutes": lifetime,
            "snapshots": int(np.sum(mask)),
        }
        bearing_maes.append(mae)
        if nmae is not None:
            bearing_nmaes.append(nmae)

    return {
        "snapshot_mae": float(np.mean(absolute_errors)),
        "macro_bearing_mae": float(np.mean(bearing_maes)),
        "median_bearing_mae": float(np.median(bearing_maes)),
        "macro_bearing_nmae": (
            float(np.mean(bearing_nmaes)) if bearing_nmaes else None
        ),
        "per_bearing": per_bearing,
    }


def evaluate_interval_predictions(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    bearing_ids: np.ndarray,
) -> Dict[str, object]:
    """Return empirical interval coverage and width by bearing."""
    truth = _validated_array(y_true, "y_true")
    lower_values = _validated_array(lower, "lower")
    upper_values = _validated_array(upper, "upper")
    bearings = _validated_array(bearing_ids, "bearing_ids", numeric=False)
    _validate_matching_lengths(
        {
            "y_true": truth,
            "lower": lower_values,
            "upper": upper_values,
            "bearing_ids": bearings,
        }
    )
    if np.any(truth < 0.0) or np.any(lower_values < 0.0) or np.any(upper_values < 0.0):
        raise ModelingMetricError("RUL targets and interval bounds cannot be negative")
    if np.any(lower_values > upper_values):
        raise ModelingMetricError("lower interval bounds cannot exceed upper bounds")

    covered = (truth >= lower_values) & (truth <= upper_values)
    widths = upper_values - lower_values
    per_bearing: Dict[str, Dict[str, float]] = {}
    bearing_coverages = []
    for bearing_id in sorted(set(bearings.tolist())):
        mask = bearings == bearing_id
        coverage = float(np.mean(covered[mask]))
        per_bearing[bearing_id] = {
            "coverage": coverage,
            "mean_width_minutes": float(np.mean(widths[mask])),
            "snapshots": int(np.sum(mask)),
        }
        bearing_coverages.append(coverage)

    return {
        "snapshot_coverage": float(np.mean(covered)),
        "macro_bearing_coverage": float(np.mean(bearing_coverages)),
        "mean_width_minutes": float(np.mean(widths)),
        "per_bearing": per_bearing,
    }
