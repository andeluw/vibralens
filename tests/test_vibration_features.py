from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from vibralens.features.vibration import (
    FeatureValidationError,
    build_feature_artifact,
    extract_channel_features,
    extract_snapshot_features,
)


class VibrationFeatureTests(unittest.TestCase):
    def test_extracts_hand_checked_time_and_frequency_features(self) -> None:
        features = extract_channel_features(
            [1.0, -1.0, 1.0, -1.0], sampling_rate_hz=25_600
        )

        self.assertAlmostEqual(0.0, features["mean"])
        self.assertAlmostEqual(1.0, features["std"])
        self.assertAlmostEqual(1.0, features["rms"])
        self.assertAlmostEqual(1.0, features["abs_peak"])
        self.assertAlmostEqual(2.0, features["peak_to_peak"])
        self.assertAlmostEqual(0.0, features["skewness"])
        self.assertAlmostEqual(1.0, features["kurtosis"])
        self.assertAlmostEqual(1.0, features["crest_factor"])
        self.assertAlmostEqual(12_800.0, features["dominant_frequency_hz"])
        self.assertAlmostEqual(12_800.0, features["spectral_centroid_hz"])
        self.assertAlmostEqual(0.0, features["spectral_entropy"])
        self.assertAlmostEqual(0.0, features["band_energy_low_fraction"])
        self.assertAlmostEqual(0.0, features["band_energy_mid_fraction"])
        self.assertAlmostEqual(1.0, features["band_energy_high_fraction"])

    def test_rejects_non_finite_signal_values(self) -> None:
        with self.assertRaisesRegex(FeatureValidationError, "non-finite"):
            extract_channel_features([1.0, math.nan, 2.0])

    def test_rejects_snapshot_with_wrong_channel_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad.csv"
            path.write_text(
                "Horizontal_vibration_signals,Vertical_vibration_signals\n"
                "1.0,2.0,3.0\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FeatureValidationError, "two columns"):
                extract_snapshot_features(path, expected_rows=1)

    def test_rejects_snapshot_with_wrong_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad-header.csv"
            path.write_text(
                "vertical,horizontal\n1.0,2.0\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FeatureValidationError, "header"):
                extract_snapshot_features(path, expected_rows=1)

    def test_builds_feature_artifact_without_losing_manifest_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            dataset_root = temporary_path / "dataset"
            snapshot_path = dataset_root / "35Hz12kN/Bearing1_1/1.csv"
            snapshot_path.parent.mkdir(parents=True)
            snapshot_path.write_text(
                "Horizontal_vibration_signals,Vertical_vibration_signals\n"
                "1.0,2.0\n-1.0,-2.0\n1.0,2.0\n-1.0,-2.0\n",
                encoding="utf-8",
            )
            manifest_path = temporary_path / "manifest.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "dataset",
                        "condition_id",
                        "bearing_id",
                        "snapshot_index",
                        "relative_path",
                        "split",
                        "n_signal_samples",
                        "sampling_rate_hz",
                        "rul_minutes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "dataset": "XJTU-SY",
                        "condition_id": 1,
                        "bearing_id": "Bearing1_1",
                        "snapshot_index": 1,
                        "relative_path": "35Hz12kN/Bearing1_1/1.csv",
                        "split": "train",
                        "n_signal_samples": 4,
                        "sampling_rate_hz": 25_600,
                        "rul_minutes": 0,
                    }
                )

            feature_path, audit_path = build_feature_artifact(
                dataset_root,
                manifest_path,
                temporary_path / "output",
                workers=1,
            )

            with feature_path.open(newline="", encoding="utf-8") as stream:
                feature_rows = list(csv.DictReader(stream))
            with audit_path.open(encoding="utf-8") as stream:
                audit = json.load(stream)

            self.assertEqual(1, len(feature_rows))
            self.assertEqual("Bearing1_1", feature_rows[0]["bearing_id"])
            self.assertEqual("train", feature_rows[0]["split"])
            self.assertEqual("0", feature_rows[0]["rul_minutes"])
            self.assertAlmostEqual(1.0, float(feature_rows[0]["horizontal_rms"]))
            self.assertAlmostEqual(2.0, float(feature_rows[0]["vertical_rms"]))
            self.assertEqual("passed", audit["status"])
            self.assertEqual(1, audit["snapshots"])
            self.assertEqual(28, audit["feature_columns"])
            self.assertEqual(
                {"min": -1.0, "max": 1.0}, audit["raw_ranges"]["horizontal"]
            )
            self.assertEqual(
                {"min": -2.0, "max": 2.0}, audit["raw_ranges"]["vertical"]
            )


if __name__ == "__main__":
    unittest.main()
