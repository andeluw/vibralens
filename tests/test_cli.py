from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from vibralens.cli import main
from vibralens.inference import InferenceValidationError, PredictionResponse
from vibralens.modeling.bundle import BundleError, RulInterval


COMMAND = [
    "--snapshot",
    "tests/fixtures/xjtu_sy_snapshot.csv",
    "--bearing-age-minutes",
    "100",
    "--condition-id",
    "1",
    "--planned-break-minutes",
    "60",
    "--model",
    "artifacts/models/vibralens_rul_v0_1.joblib",
]


class PredictionCliTests(unittest.TestCase):
    @staticmethod
    def _response() -> PredictionResponse:
        return PredictionResponse(
            model_version="vibralens-rul-0.1.0",
            rul_minutes=RulInterval(25.0, 42.0, 68.0),
            planned_break_minutes=60.0,
            break_risk="uncertain",
            advisory="inspect_first",
            support_status="supported",
            limitations=("Empirical interval",),
        )

    def test_prints_the_canonical_response_as_json(self) -> None:
        service = mock.Mock()
        service.predict.return_value = self._response()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch("vibralens.cli.load_bundle", return_value=object()), mock.patch(
            "vibralens.cli.PredictionService", return_value=service
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(COMMAND)

        self.assertEqual(0, exit_code)
        self.assertEqual(self._response().to_dict(), json.loads(stdout.getvalue()))
        self.assertEqual("", stderr.getvalue())
        request = service.predict.call_args.args[0]
        self.assertEqual(Path(COMMAND[1]), request.snapshot_path)
        self.assertEqual(100.0, request.bearing_age_minutes)
        self.assertEqual(1, request.condition_id)
        self.assertEqual(60.0, request.planned_break_minutes)

    def test_validation_error_uses_exit_code_two_and_no_stdout(self) -> None:
        service = mock.Mock()
        service.predict.side_effect = InferenceValidationError("invalid snapshot")
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch("vibralens.cli.load_bundle", return_value=object()), mock.patch(
            "vibralens.cli.PredictionService", return_value=service
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(COMMAND)

        self.assertEqual(2, exit_code)
        self.assertEqual("", stdout.getvalue())
        self.assertIn("invalid snapshot", stderr.getvalue())

    def test_artifact_error_uses_exit_code_three_and_no_stdout(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch(
            "vibralens.cli.load_bundle",
            side_effect=BundleError("cannot load artifact"),
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(COMMAND)

        self.assertEqual(3, exit_code)
        self.assertEqual("", stdout.getvalue())
        self.assertIn("cannot load artifact", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
