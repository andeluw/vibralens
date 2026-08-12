"""Versioned, self-describing RUL inference bundles."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Tuple

import joblib
import numpy as np


BUNDLE_FORMAT_VERSION = 1


class BundleError(RuntimeError):
    """Base class for model artifact failures."""


class BundleCompatibilityError(BundleError):
    """Raised when an artifact does not match the runtime contract."""


@dataclass(frozen=True)
class RulInterval:
    pessimistic: float
    median: float
    optimistic: float

    def __post_init__(self) -> None:
        values = (self.pessimistic, self.median, self.optimistic)
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise BundleCompatibilityError(
                "RUL interval values must be finite and non-negative"
            )
        if not self.pessimistic <= self.median <= self.optimistic:
            raise BundleCompatibilityError("RUL interval values must be ordered")

    def to_dict(self) -> Mapping[str, float]:
        return {
            "pessimistic": self.pessimistic,
            "median": self.median,
            "optimistic": self.optimistic,
        }


@dataclass
class RulModelBundle:
    bundle_format_version: int
    model_version: str
    estimator: object
    interval_radius_minutes: float
    feature_set: str
    include_age: bool
    feature_names: Tuple[str, ...]
    supported_condition_ids: Tuple[int, ...]
    metadata: Mapping[str, object]

    def validate(self) -> None:
        if self.bundle_format_version != BUNDLE_FORMAT_VERSION:
            raise BundleCompatibilityError(
                "unsupported bundle format version {}".format(
                    self.bundle_format_version
                )
            )
        if not self.model_version:
            raise BundleCompatibilityError("model_version cannot be empty")
        if self.feature_set not in ("horizontal", "vertical", "dual"):
            raise BundleCompatibilityError("bundle has invalid feature_set")
        if not self.feature_names or self.feature_names[0] != "condition_id":
            raise BundleCompatibilityError("bundle feature schema must start with condition_id")
        if self.include_age != ("age_minutes" in self.feature_names):
            raise BundleCompatibilityError("bundle age schema is inconsistent")
        if not self.supported_condition_ids:
            raise BundleCompatibilityError("bundle has no supported conditions")
        if (
            not math.isfinite(self.interval_radius_minutes)
            or self.interval_radius_minutes < 0.0
        ):
            raise BundleCompatibilityError("bundle interval radius is invalid")
        if not hasattr(self.estimator, "predict"):
            raise BundleCompatibilityError("bundle estimator cannot predict")
        expected_feature_count = getattr(self.estimator, "n_features_in_", None)
        if (
            expected_feature_count is not None
            and int(expected_feature_count) != len(self.feature_names)
        ):
            raise BundleCompatibilityError(
                "bundle feature count disagrees with fitted estimator"
            )
        metadata_version = self.metadata.get("bundle_format_version")
        if metadata_version != self.bundle_format_version:
            raise BundleCompatibilityError("bundle metadata format version disagrees")
        if self.metadata.get("model_version") != self.model_version:
            raise BundleCompatibilityError("bundle metadata model version disagrees")
        if tuple(self.metadata.get("feature_names", ())) != self.feature_names:
            raise BundleCompatibilityError("bundle metadata feature schema disagrees")

    def predict(
        self,
        feature_values: Mapping[str, float],
        *,
        age_minutes: float,
        condition_id: int,
    ) -> RulInterval:
        """Predict one RUL point and apply the frozen empirical interval radius."""
        self.validate()
        if condition_id not in self.supported_condition_ids:
            raise BundleCompatibilityError(
                "unsupported condition {}".format(condition_id)
            )
        if not math.isfinite(age_minutes) or age_minutes < 0.0:
            raise BundleCompatibilityError("age_minutes must be finite and non-negative")

        row = []
        for name in self.feature_names:
            if name == "condition_id":
                value = float(condition_id)
            elif name == "age_minutes":
                value = float(age_minutes)
            else:
                try:
                    value = float(feature_values[name])
                except (KeyError, TypeError, ValueError) as error:
                    raise BundleCompatibilityError(
                        "missing or invalid feature {}".format(name)
                    ) from error
            if not math.isfinite(value):
                raise BundleCompatibilityError(
                    "feature {} must be finite".format(name)
                )
            row.append(value)

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            raw_prediction = np.asarray(
                self.estimator.predict(np.asarray([row], dtype=np.float64)),
                dtype=np.float64,
            )
        if raw_prediction.shape != (1,) or not np.isfinite(raw_prediction[0]):
            raise BundleCompatibilityError("model produced an invalid prediction")
        median = max(float(raw_prediction[0]), 0.0)
        return RulInterval(
            pessimistic=max(median - self.interval_radius_minutes, 0.0),
            median=median,
            optimistic=median + self.interval_radius_minutes,
        )


def save_bundle(
    bundle: RulModelBundle,
    path: Path,
    metadata_path: Path,
) -> None:
    """Validate and atomically write the binary bundle and readable metadata."""
    bundle.validate()
    path = Path(path)
    metadata_path = Path(metadata_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    try:
        joblib.dump(bundle, temporary_path)
        with temporary_metadata.open("w", encoding="utf-8") as stream:
            json.dump(bundle.metadata, stream, indent=2, sort_keys=True)
            stream.write("\n")
        temporary_metadata.replace(metadata_path)
        temporary_path.replace(path)
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
        raise BundleError("cannot save model bundle") from error


def load_bundle(path: Path) -> RulModelBundle:
    """Load a bundle and reject corrupt or incompatible artifacts."""
    path = Path(path)
    try:
        bundle = joblib.load(path)
    except Exception as error:
        raise BundleError("cannot load model bundle: {}".format(path)) from error
    if not isinstance(bundle, RulModelBundle):
        raise BundleCompatibilityError("artifact is not a RulModelBundle")
    bundle.validate()
    return bundle
