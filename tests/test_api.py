from __future__ import annotations

import unittest
from pathlib import Path
from typing import Optional
from unittest import mock

from fastapi.testclient import TestClient

from vibralens.api import create_app
from vibralens.inference import (
    InferenceValidationError,
    PredictionResponse,
)
from vibralens.modeling.bundle import BundleError, RulInterval


SNAPSHOT_BYTES = Path("tests/fixtures/xjtu_sy_snapshot.csv").read_bytes()


class FakeBundle:
    model_version = "vibralens-rul-0.1.0"
    supported_condition_ids = (1, 2, 3)
    feature_set = "vertical"
    include_age = True
    feature_names = ("condition_id", "age_minutes", "vertical_rms")
    metadata = {
        "limitations": [
            "The RUL interval is empirical and is not a formal safety guarantee."
        ]
    }


class FakeService:
    def __init__(self) -> None:
        self.bundle = FakeBundle()
        self.calls = 0
        self.snapshot_path: Optional[Path] = None
        self.snapshot_existed_during_call = False
        self.error: Optional[Exception] = None
        self.response = PredictionResponse(
            model_version="vibralens-rul-0.1.0",
            rul_minutes=RulInterval(25.0, 42.0, 68.0),
            planned_break_minutes=60.0,
            break_risk="uncertain",
            advisory="inspect_first",
            support_status="supported",
            limitations=tuple(self.bundle.metadata["limitations"]),
        )

    def predict(self, request: object) -> PredictionResponse:
        self.calls += 1
        self.snapshot_path = Path(request.snapshot_path)  # type: ignore[attr-defined]
        self.snapshot_existed_during_call = self.snapshot_path.exists()
        if self.error is not None:
            raise self.error
        return self.response


def prediction_form() -> dict[str, str]:
    return {
        "bearing_age_minutes": "100",
        "condition_id": "1",
        "planned_break_minutes": "60",
    }


class InferenceApiTests(unittest.TestCase):
    def test_health_and_model_describe_the_loaded_artifact(self) -> None:
        service = FakeService()
        client = TestClient(create_app(Path("unused.joblib"), service=service))

        health = client.get("/health")
        model = client.get("/model")

        self.assertEqual(200, health.status_code)
        self.assertEqual(
            {"status": "ready", "model_version": "vibralens-rul-0.1.0"},
            health.json(),
        )
        self.assertEqual(200, model.status_code)
        self.assertEqual(
            {
                "model_version": "vibralens-rul-0.1.0",
                "supported_condition_ids": [1, 2, 3],
                "feature_set": "vertical",
                "include_age": True,
                "feature_names": [
                    "condition_id",
                    "age_minutes",
                    "vertical_rms",
                ],
                "limitations": [
                    "The RUL interval is empirical and is not a formal safety guarantee."
                ],
            },
            model.json(),
        )

    def test_predict_returns_service_response_and_removes_upload(self) -> None:
        service = FakeService()
        client = TestClient(create_app(Path("unused.joblib"), service=service))

        response = client.post(
            "/predict",
            data=prediction_form(),
            files={"snapshot": ("snapshot.csv", SNAPSHOT_BYTES, "text/csv")},
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual(service.response.to_dict(), response.json())
        self.assertEqual(1, service.calls)
        self.assertTrue(service.snapshot_existed_during_call)
        self.assertIsNotNone(service.snapshot_path)
        self.assertFalse(service.snapshot_path.exists())  # type: ignore[union-attr]

    def test_invalid_csv_and_scalar_input_return_422(self) -> None:
        service = FakeService()
        service.error = InferenceValidationError("snapshot has invalid header")
        client = TestClient(create_app(Path("unused.joblib"), service=service))

        invalid_csv = client.post(
            "/predict",
            data=prediction_form(),
            files={"snapshot": ("bad.csv", b"wrong,header\n1,2\n", "text/csv")},
        )
        invalid_scalar = client.post(
            "/predict",
            data={**prediction_form(), "bearing_age_minutes": "-1"},
            files={"snapshot": ("snapshot.csv", SNAPSHOT_BYTES, "text/csv")},
        )

        self.assertEqual(422, invalid_csv.status_code)
        self.assertIn("invalid header", invalid_csv.json()["detail"])
        self.assertEqual(422, invalid_scalar.status_code)
        self.assertIn("greater than or equal to 0", str(invalid_scalar.json()))

    def test_unavailable_artifact_never_reports_ready(self) -> None:
        with mock.patch(
            "vibralens.api.load_bundle",
            side_effect=BundleError("cannot load model"),
        ):
            client = TestClient(create_app(Path("missing.joblib")))

        health = client.get("/health")
        model = client.get("/model")

        self.assertEqual(503, health.status_code)
        self.assertEqual(503, model.status_code)
        self.assertEqual("unavailable", health.json()["status"])

    def test_unexpected_failure_returns_500_without_prediction_body(self) -> None:
        service = FakeService()
        service.error = RuntimeError("internal implementation detail")
        client = TestClient(
            create_app(Path("unused.joblib"), service=service),
            raise_server_exceptions=False,
        )

        response = client.post(
            "/predict",
            data=prediction_form(),
            files={"snapshot": ("snapshot.csv", SNAPSHOT_BYTES, "text/csv")},
        )

        self.assertEqual(500, response.status_code)
        self.assertEqual({"detail": "prediction failed"}, response.json())
        self.assertNotIn("rul_minutes", response.json())


if __name__ == "__main__":
    unittest.main()
