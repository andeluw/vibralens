from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from typing import Mapping, Optional

from vibralens.features.vibration import extract_snapshot_features
from vibralens.inference import (
    InferenceValidationError,
    PredictionRequest,
    PredictionService,
)
from vibralens.modeling.bundle import RulInterval


FIXTURE = Path("tests/fixtures/xjtu_sy_snapshot.csv")


class FakeBundle:
    model_version = "vibralens-rul-0.1.0"
    supported_condition_ids = (1, 2, 3)
    metadata = {
        "limitations": [
            "The RUL interval is empirical and is not a formal safety guarantee."
        ]
    }

    def __init__(self) -> None:
        self.received_features: Optional[Mapping[str, float]] = None
        self.predict_calls = 0

    def predict(
        self,
        feature_values: Mapping[str, float],
        *,
        age_minutes: float,
        condition_id: int,
    ) -> RulInterval:
        self.predict_calls += 1
        self.received_features = feature_values
        return RulInterval(pessimistic=25.0, median=42.0, optimistic=68.0)


class PredictionServiceTests(unittest.TestCase):
    @staticmethod
    def _request(**overrides: object) -> PredictionRequest:
        values = {
            "snapshot_path": FIXTURE,
            "bearing_age_minutes": 100.0,
            "condition_id": 1,
            "planned_break_minutes": 60.0,
        }
        values.update(overrides)
        return PredictionRequest(**values)  # type: ignore[arg-type]

    def test_returns_canonical_supported_prediction(self) -> None:
        service = PredictionService(FakeBundle(), expected_signal_rows=4)

        response = service.predict(self._request())

        self.assertEqual("vibralens-rul-0.1.0", response.model_version)
        self.assertIsNotNone(response.rul_minutes)
        self.assertEqual(42.0, response.rul_minutes.median)  # type: ignore[union-attr]
        self.assertEqual("uncertain", response.break_risk)
        self.assertEqual("inspect_first", response.advisory)
        self.assertEqual("supported", response.support_status)
        self.assertEqual(
            {
                "model_version": "vibralens-rul-0.1.0",
                "rul_minutes": {
                    "pessimistic": 25.0,
                    "median": 42.0,
                    "optimistic": 68.0,
                },
                "planned_break_minutes": 60.0,
                "break_risk": "uncertain",
                "advisory": "inspect_first",
                "support_status": "supported",
                "limitations": [
                    "The RUL interval is empirical and is not a formal safety guarantee."
                ],
            },
            response.to_dict(),
        )

    def test_passes_production_feature_mapping_to_predictor_unchanged(self) -> None:
        bundle = FakeBundle()
        service = PredictionService(bundle, expected_signal_rows=4)

        service.predict(self._request())

        self.assertEqual(
            extract_snapshot_features(FIXTURE, expected_rows=4),
            bundle.received_features,
        )

    def test_abstains_for_well_formed_unsupported_condition(self) -> None:
        bundle = FakeBundle()
        service = PredictionService(bundle, expected_signal_rows=4)

        response = service.predict(self._request(condition_id=9))

        self.assertIsNone(response.rul_minutes)
        self.assertEqual("unknown", response.break_risk)
        self.assertEqual("inspect_first", response.advisory)
        self.assertEqual("unsupported", response.support_status)
        self.assertEqual(0, bundle.predict_calls)

    def test_rejects_invalid_scalar_inputs(self) -> None:
        service = PredictionService(FakeBundle(), expected_signal_rows=4)
        invalid_requests = (
            self._request(bearing_age_minutes=-1.0),
            self._request(bearing_age_minutes=math.nan),
            self._request(planned_break_minutes=-1.0),
            self._request(planned_break_minutes=math.inf),
            self._request(condition_id=True),
        )

        for request in invalid_requests:
            with self.subTest(request=request):
                with self.assertRaises(InferenceValidationError):
                    service.predict(request)

    def test_rejects_invalid_snapshot(self) -> None:
        service = PredictionService(FakeBundle(), expected_signal_rows=4)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "invalid.csv"
            path.write_text("wrong,header\n1,2\n", encoding="utf-8")

            with self.assertRaisesRegex(InferenceValidationError, "header"):
                service.predict(self._request(snapshot_path=path))


if __name__ == "__main__":
    unittest.main()
