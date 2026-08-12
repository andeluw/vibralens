"""Validated feature-table loading and model matrix construction."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from vibralens.features.vibration import FEATURE_COLUMN_NAMES


VALID_SPLITS = frozenset(("train", "validation", "test"))


class ModelingDataError(ValueError):
    """Raised when a feature artifact violates the modeling contract."""


@dataclass(frozen=True)
class FeatureTable:
    vibration: np.ndarray
    rul_minutes: np.ndarray
    age_minutes: np.ndarray
    condition_ids: np.ndarray
    bearing_ids: np.ndarray
    snapshot_indices: np.ndarray
    splits: np.ndarray
    feature_names: Tuple[str, ...]


@dataclass(frozen=True)
class ModelMatrix:
    values: np.ndarray
    targets: np.ndarray
    bearing_ids: np.ndarray
    condition_ids: np.ndarray
    ages_minutes: np.ndarray
    feature_names: Tuple[str, ...]


def _parse_int(row: dict, field: str, row_number: int) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ModelingDataError(
            "row {} has invalid {}".format(row_number, field)
        ) from error
    return value


def _parse_float(row: dict, field: str, row_number: int) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ModelingDataError(
            "row {} has invalid {}".format(row_number, field)
        ) from error
    if not np.isfinite(value):
        raise ModelingDataError(
            "row {} has non-finite {}".format(row_number, field)
        )
    return value


def load_feature_table(path: Path) -> FeatureTable:
    """Load and validate the deterministic feature artifact."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = tuple(reader.fieldnames or ())
            rows = list(reader)
    except OSError as error:
        raise ModelingDataError("cannot read feature table: {}".format(path)) from error

    required = {
        "condition_id",
        "bearing_id",
        "snapshot_index",
        "split",
        "snapshot_interval_minutes",
        "rul_minutes",
        *FEATURE_COLUMN_NAMES,
    }
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ModelingDataError("feature table is missing columns: {}".format(missing))
    if not rows:
        raise ModelingDataError("feature table must contain at least one row")

    vibration_rows: List[List[float]] = []
    rul_values: List[float] = []
    age_values: List[float] = []
    condition_values: List[int] = []
    bearing_values: List[str] = []
    snapshot_values: List[int] = []
    split_values: List[str] = []
    identities = set()
    bearing_splits = {}
    bearing_conditions = {}

    for row_number, row in enumerate(rows, start=2):
        condition_id = _parse_int(row, "condition_id", row_number)
        snapshot_index = _parse_int(row, "snapshot_index", row_number)
        interval = _parse_float(row, "snapshot_interval_minutes", row_number)
        rul = _parse_float(row, "rul_minutes", row_number)
        bearing_id = row.get("bearing_id", "").strip()
        split = row.get("split", "").strip()

        if condition_id < 1:
            raise ModelingDataError("condition_id must be positive")
        if not bearing_id:
            raise ModelingDataError("bearing_id cannot be empty")
        if snapshot_index < 1:
            raise ModelingDataError("snapshot_index must be at least one")
        if interval <= 0.0:
            raise ModelingDataError("snapshot_interval_minutes must be positive")
        if rul < 0.0:
            raise ModelingDataError("rul_minutes cannot be negative")
        if split not in VALID_SPLITS:
            raise ModelingDataError("unknown split {!r}".format(split))

        identity = (bearing_id, snapshot_index)
        if identity in identities:
            raise ModelingDataError("duplicate bearing/snapshot identity: {}".format(identity))
        identities.add(identity)
        previous_split = bearing_splits.setdefault(bearing_id, split)
        if previous_split != split:
            raise ModelingDataError(
                "bearing {} appears in multiple splits".format(bearing_id)
            )
        previous_condition = bearing_conditions.setdefault(bearing_id, condition_id)
        if previous_condition != condition_id:
            raise ModelingDataError(
                "bearing {} appears in multiple conditions".format(bearing_id)
            )

        features = [
            _parse_float(row, feature_name, row_number)
            for feature_name in FEATURE_COLUMN_NAMES
        ]
        vibration_rows.append(features)
        rul_values.append(rul)
        age_values.append((snapshot_index - 1) * interval)
        condition_values.append(condition_id)
        bearing_values.append(bearing_id)
        snapshot_values.append(snapshot_index)
        split_values.append(split)

    return FeatureTable(
        vibration=np.asarray(vibration_rows, dtype=np.float64),
        rul_minutes=np.asarray(rul_values, dtype=np.float64),
        age_minutes=np.asarray(age_values, dtype=np.float64),
        condition_ids=np.asarray(condition_values, dtype=np.int64),
        bearing_ids=np.asarray(bearing_values, dtype=str),
        snapshot_indices=np.asarray(snapshot_values, dtype=np.int64),
        splits=np.asarray(split_values, dtype=str),
        feature_names=tuple(FEATURE_COLUMN_NAMES),
    )


def build_model_matrix(
    table: FeatureTable,
    row_mask: np.ndarray,
    *,
    feature_set: str,
    include_age: bool,
) -> ModelMatrix:
    """Build an ordered numeric matrix for a declared channel/age variant."""
    mask = np.asarray(row_mask)
    if mask.dtype != np.bool_ or mask.ndim != 1 or mask.size != table.rul_minutes.size:
        raise ModelingDataError("row_mask must be a one-dimensional boolean array")
    if not np.any(mask):
        raise ModelingDataError("row_mask selects no rows")
    if feature_set == "horizontal":
        selected_indices = [
            index
            for index, name in enumerate(table.feature_names)
            if name.startswith("horizontal_")
        ]
    elif feature_set == "vertical":
        selected_indices = [
            index
            for index, name in enumerate(table.feature_names)
            if name.startswith("vertical_")
        ]
    elif feature_set == "dual":
        selected_indices = list(range(len(table.feature_names)))
    else:
        raise ModelingDataError("unknown feature_set {!r}".format(feature_set))

    columns = [table.condition_ids[mask].astype(np.float64)]
    feature_names = ["condition_id"]
    if include_age:
        columns.append(table.age_minutes[mask])
        feature_names.append("age_minutes")
    columns.extend(table.vibration[mask, index] for index in selected_indices)
    feature_names.extend(table.feature_names[index] for index in selected_indices)
    values = np.column_stack(columns)
    if not np.isfinite(values).all():
        raise ModelingDataError("model matrix contains non-finite values")

    return ModelMatrix(
        values=values,
        targets=table.rul_minutes[mask],
        bearing_ids=table.bearing_ids[mask],
        condition_ids=table.condition_ids[mask],
        ages_minutes=table.age_minutes[mask],
        feature_names=tuple(feature_names),
    )
