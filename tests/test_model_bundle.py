from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import joblib
from sklearn.dummy import DummyRegressor

from vibralens.modeling.bundle import (
    BundleCompatibilityError,
    BundleError,
    RulInterval,
    RulModelBundle,
    load_bundle,
    save_bundle,
)


class NonFiniteRegressor:
    n_features_in_ = 3

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.full(values.shape[0], np.nan)


class ModelBundleTests(unittest.TestCase):
    @staticmethod
    def _bundle() -> RulModelBundle:
        estimator = DummyRegressor(strategy="constant", constant=42.0)
        estimator.fit(np.array([[1.0, 3.0, 7.0]]), np.array([42.0]))
        metadata = {
            "bundle_format_version": 1,
            "model_version": "vibralens-rul-0.1.0",
            "feature_set": "vertical",
            "include_age": True,
            "feature_names": ["condition_id", "age_minutes", "vertical_rms"],
            "supported_condition_ids": [1, 2, 3],
            "interval_radius_minutes": 10.0,
            "limitations": ["Empirical interval"],
        }
        return RulModelBundle(
            bundle_format_version=1,
            model_version="vibralens-rul-0.1.0",
            estimator=estimator,
            interval_radius_minutes=10.0,
            feature_set="vertical",
            include_age=True,
            feature_names=("condition_id", "age_minutes", "vertical_rms"),
            supported_condition_ids=(1, 2, 3),
            metadata=metadata,
        )

    def test_round_trips_and_predicts_an_ordered_non_negative_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            bundle_path = Path(temporary_directory) / "model.joblib"
            metadata_path = Path(temporary_directory) / "model.json"
            bundle = self._bundle()

            save_bundle(bundle, bundle_path, metadata_path)
            loaded = load_bundle(bundle_path)
            interval = loaded.predict(
                {"vertical_rms": 7.0},
                age_minutes=3.0,
                condition_id=1,
            )

            self.assertEqual("vibralens-rul-0.1.0", loaded.model_version)
            self.assertEqual(bundle.feature_names, loaded.feature_names)
            self.assertEqual(RulInterval(32.0, 42.0, 52.0), interval)
            self.assertEqual(
                loaded.metadata,
                json.loads(metadata_path.read_text(encoding="utf-8")),
            )

    def test_rejects_unsupported_condition_before_estimator_prediction(self) -> None:
        with self.assertRaisesRegex(BundleCompatibilityError, "unsupported condition"):
            self._bundle().predict(
                {"vertical_rms": 7.0},
                age_minutes=3.0,
                condition_id=9,
            )

    def test_rejects_missing_feature_and_unsupported_bundle_format(self) -> None:
        bundle = self._bundle()
        with self.assertRaisesRegex(BundleCompatibilityError, "vertical_rms"):
            bundle.predict({}, age_minutes=3.0, condition_id=1)

        bundle.bundle_format_version = 2
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "model.joblib"
            joblib.dump(bundle, path)
            with self.assertRaisesRegex(BundleCompatibilityError, "format version"):
                load_bundle(path)

    def test_rejects_missing_and_corrupt_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "missing.joblib"
            with self.assertRaisesRegex(BundleError, "cannot load model bundle"):
                load_bundle(missing)

            corrupt = Path(temporary_directory) / "corrupt.joblib"
            corrupt.write_bytes(b"this is not a joblib artifact")
            with self.assertRaisesRegex(BundleError, "cannot load model bundle"):
                load_bundle(corrupt)

    def test_rejects_estimator_feature_count_mismatch(self) -> None:
        bundle = self._bundle()
        bundle.feature_names = ("condition_id", "age_minutes")
        bundle.metadata = {**bundle.metadata, "feature_names": list(bundle.feature_names)}

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(BundleCompatibilityError, "feature count"):
                save_bundle(
                    bundle,
                    Path(temporary_directory) / "model.joblib",
                    Path(temporary_directory) / "model.json",
                )

    def test_rejects_non_finite_model_output(self) -> None:
        bundle = self._bundle()
        bundle.estimator = NonFiniteRegressor()

        with self.assertRaisesRegex(BundleCompatibilityError, "invalid prediction"):
            bundle.predict(
                {"vertical_rms": 7.0},
                age_minutes=3.0,
                condition_id=1,
            )


if __name__ == "__main__":
    unittest.main()
