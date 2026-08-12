"""Canonical inference path shared by command-line and HTTP adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

from vibralens.data.xjtu_sy import SIGNAL_ROWS_PER_SNAPSHOT
from vibralens.decision import assess_planned_break
from vibralens.features.vibration import (
    FeatureValidationError,
    extract_snapshot_features,
)
from vibralens.modeling.bundle import RulInterval, RulModelBundle


EMPIRICAL_INTERVAL_LIMITATION = (
    "The RUL interval is empirical and is not a formal safety guarantee."
)


class InferenceValidationError(ValueError):
    """Raised when a prediction request cannot be safely evaluated."""


@dataclass(frozen=True)
class PredictionRequest:
    snapshot_path: Path
    bearing_age_minutes: float
    condition_id: int
    planned_break_minutes: float


@dataclass(frozen=True)
class PredictionResponse:
    model_version: str
    rul_minutes: Optional[RulInterval]
    planned_break_minutes: float
    break_risk: str
    advisory: str
    support_status: str
    limitations: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_version": self.model_version,
            "rul_minutes": (
                None if self.rul_minutes is None else dict(self.rul_minutes.to_dict())
            ),
            "planned_break_minutes": self.planned_break_minutes,
            "break_risk": self.break_risk,
            "advisory": self.advisory,
            "support_status": self.support_status,
            "limitations": list(self.limitations),
        }


class PredictionService:
    def __init__(
        self,
        bundle: RulModelBundle,
        *,
        expected_signal_rows: int = SIGNAL_ROWS_PER_SNAPSHOT,
    ) -> None:
        if expected_signal_rows < 2:
            raise ValueError("expected_signal_rows must be at least two")
        self.bundle = bundle
        self.expected_signal_rows = expected_signal_rows
        raw_limitations = bundle.metadata.get("limitations", ())
        if isinstance(raw_limitations, str) or not isinstance(
            raw_limitations, (list, tuple)
        ):
            raise ValueError("bundle limitations metadata is invalid")
        limitations = tuple(str(value) for value in raw_limitations)
        if not any(
            "empirical" in value.lower() and "interval" in value.lower()
            for value in limitations
        ):
            limitations = (*limitations, EMPIRICAL_INTERVAL_LIMITATION)
        self.limitations = limitations

    @staticmethod
    def _non_negative_finite(value: object, name: str) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise InferenceValidationError(
                "{} must be a finite non-negative number".format(name)
            ) from error
        if not math.isfinite(numeric) or numeric < 0.0:
            raise InferenceValidationError(
                "{} must be a finite non-negative number".format(name)
            )
        return numeric

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        age = self._non_negative_finite(
            request.bearing_age_minutes,
            "bearing_age_minutes",
        )
        horizon = self._non_negative_finite(
            request.planned_break_minutes,
            "planned_break_minutes",
        )
        if isinstance(request.condition_id, bool) or not isinstance(
            request.condition_id, int
        ):
            raise InferenceValidationError("condition_id must be an integer")

        try:
            features: Mapping[str, float] = extract_snapshot_features(
                Path(request.snapshot_path),
                expected_rows=self.expected_signal_rows,
            )
        except FeatureValidationError as error:
            raise InferenceValidationError(str(error)) from error

        if request.condition_id not in self.bundle.supported_condition_ids:
            return PredictionResponse(
                model_version=self.bundle.model_version,
                rul_minutes=None,
                planned_break_minutes=horizon,
                break_risk="unknown",
                advisory="inspect_first",
                support_status="unsupported",
                limitations=self.limitations,
            )

        interval = self.bundle.predict(
            features,
            age_minutes=age,
            condition_id=request.condition_id,
        )
        assessment = assess_planned_break(interval, horizon)
        return PredictionResponse(
            model_version=self.bundle.model_version,
            rul_minutes=interval,
            planned_break_minutes=horizon,
            break_risk=assessment.break_risk,
            advisory=assessment.advisory,
            support_status="supported",
            limitations=self.limitations,
        )
