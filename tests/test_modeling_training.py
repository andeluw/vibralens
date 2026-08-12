from __future__ import annotations

import unittest
import csv
import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

from vibralens.features.vibration import FEATURE_COLUMN_NAMES
from vibralens.modeling.table import FeatureTable
from vibralens.modeling.training import (
    ExperimentConfig,
    TrainingError,
    build_ridge_pipeline,
    bearing_balanced_quantile,
    bearing_balanced_sample_weights,
    load_experiment_config,
    passes_vibration_gate,
    postprocess_quantiles,
    select_quantile_candidate,
    select_ridge_candidate,
    run_selection,
    main,
)


class ModelingTrainingTests(unittest.TestCase):
    @staticmethod
    def _small_table() -> FeatureTable:
        rows = 9
        base = np.arange(rows, dtype=np.float64)
        return FeatureTable(
            vibration=np.repeat(base[:, None], len(FEATURE_COLUMN_NAMES), axis=1),
            rul_minutes=np.array(
                [10.0, 9.0, 20.0, 19.0, 30.0, 29.0, 8.0, 18.0, 28.0]
            ),
            age_minutes=np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 2.0, 2.0]),
            condition_ids=np.array([1, 1, 2, 2, 3, 3, 1, 2, 3]),
            bearing_ids=np.array(
                ["T1", "T1", "T2", "T2", "T3", "T3", "V1", "V2", "V3"]
            ),
            snapshot_indices=np.array([1, 2, 1, 2, 1, 2, 3, 3, 3]),
            splits=np.array(
                [
                    "train",
                    "train",
                    "train",
                    "train",
                    "train",
                    "train",
                    "validation",
                    "validation",
                    "validation",
                ]
            ),
            feature_names=tuple(FEATURE_COLUMN_NAMES),
        )

    @staticmethod
    def _small_config() -> ExperimentConfig:
        return ExperimentConfig(
            schema_version=1,
            model_version="test",
            random_state=2026,
            quantiles=(0.1, 0.5, 0.9),
            feature_sets=("dual",),
            include_age_options=(True,),
            ridge_alphas=(1.0,),
            minimum_macro_mae_improvement_fraction=0.05,
            minimum_validation_bearings_improved=2,
            empirical_interval_coverage=0.8,
            gradient_boosting_candidates=(
                {
                    "learning_rate": 0.1,
                    "max_iter": 20,
                    "max_leaf_nodes": 5,
                    "min_samples_leaf": 2,
                    "l2_regularization": 1.0,
                },
            ),
        )

    def test_loads_the_frozen_experiment_configuration(self) -> None:
        config = load_experiment_config(
            Path("configs/models/xjtu_sy_rul_v0_1.json")
        )

        self.assertEqual((0.1, 0.5, 0.9), config.quantiles)
        self.assertEqual(2026, config.random_state)
        self.assertEqual(("horizontal", "vertical", "dual"), config.feature_sets)

    def test_ridge_preprocessing_is_fitted_only_on_supplied_training_rows(self) -> None:
        train_values = np.array(
            [
                [1.0, 1.0, 10.0],
                [2.0, 3.0, 30.0],
            ]
        )
        pipeline = build_ridge_pipeline(alpha=1.0, number_of_columns=3)
        pipeline.fit(train_values, np.array([5.0, 1.0]))

        numeric_scaler = pipeline.named_steps["preprocessor"].named_transformers_[
            "numeric"
        ]
        np.testing.assert_allclose([2.0, 20.0], numeric_scaler.mean_)

    def test_bearing_weights_give_each_run_equal_total_influence(self) -> None:
        weights = bearing_balanced_sample_weights(
            np.array(["A", "A", "A", "B"])
        )

        np.testing.assert_allclose([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0], weights)
        self.assertAlmostEqual(1.0, float(np.sum(weights[:3])))
        self.assertAlmostEqual(1.0, float(weights[3]))

    def test_bearing_balanced_quantile_does_not_let_long_runs_dominate(self) -> None:
        value = bearing_balanced_quantile(
            np.array([1.0, 2.0, 3.0, 100.0]),
            np.array(["A", "A", "A", "B"]),
            0.8,
        )

        self.assertEqual(100.0, value)

    def test_orders_and_clips_quantile_predictions(self) -> None:
        lower, median, upper = postprocess_quantiles(
            np.array([-2.0, 8.0]),
            np.array([5.0, 4.0]),
            np.array([3.0, 10.0]),
        )

        np.testing.assert_allclose([0.0, 4.0], lower)
        np.testing.assert_allclose([3.0, 8.0], median)
        np.testing.assert_allclose([5.0, 10.0], upper)

    def test_requires_both_macro_and_per_bearing_improvement(self) -> None:
        age_metrics = {
            "macro_bearing_mae": 100.0,
            "per_bearing": {
                "A": {"mae_minutes": 100.0},
                "B": {"mae_minutes": 100.0},
                "C": {"mae_minutes": 100.0},
            },
        }
        candidate_metrics = {
            "macro_bearing_mae": 94.0,
            "per_bearing": {
                "A": {"mae_minutes": 90.0},
                "B": {"mae_minutes": 92.0},
                "C": {"mae_minutes": 100.0},
            },
        }

        passed = passes_vibration_gate(
            age_metrics,
            candidate_metrics,
            minimum_improvement_fraction=0.05,
            minimum_bearings_improved=2,
        )
        self.assertTrue(passed["passed"])
        self.assertEqual(2, passed["bearings_improved"])

        failed_macro = passes_vibration_gate(
            age_metrics,
            {**candidate_metrics, "macro_bearing_mae": 96.0},
            minimum_improvement_fraction=0.05,
            minimum_bearings_improved=2,
        )
        self.assertFalse(failed_macro["passed"])

        failed_bearings = passes_vibration_gate(
            age_metrics,
            {
                **candidate_metrics,
                "per_bearing": {
                    "A": {"mae_minutes": 90.0},
                    "B": {"mae_minutes": 100.0},
                    "C": {"mae_minutes": 100.0},
                },
            },
            minimum_improvement_fraction=0.05,
            minimum_bearings_improved=2,
        )
        self.assertFalse(failed_bearings["passed"])

    def test_selects_ridge_and_quantile_candidates_on_validation_bearings(self) -> None:
        table = self._small_table()
        config = self._small_config()

        ridge_best, ridge_results = select_ridge_candidate(table, config)
        quantile_best, quantile_results = select_quantile_candidate(table, config)

        self.assertEqual(1, len(ridge_results))
        self.assertEqual("ridge", ridge_best.estimator_family)
        self.assertEqual(1, len(quantile_results))
        self.assertEqual("hist_gradient_boosting_quantile", quantile_best.estimator_family)
        self.assertIn(
            "macro_bearing_coverage",
            quantile_best.validation_interval_metrics,
        )

    def test_selection_report_contains_no_held_out_test_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            feature_path = root / "features.csv"
            fieldnames = [
                "condition_id",
                "bearing_id",
                "snapshot_index",
                "split",
                "snapshot_interval_minutes",
                "rul_minutes",
                *FEATURE_COLUMN_NAMES,
            ]
            with feature_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                for row_index in range(9):
                    condition_id = row_index % 3 + 1
                    split = "train" if row_index < 6 else "validation"
                    bearing = "T{}".format(condition_id) if split == "train" else "V{}".format(condition_id)
                    snapshot_index = row_index // 3 + 1
                    lifetime = float(condition_id * 10)
                    age = float(snapshot_index - 1)
                    writer.writerow(
                        {
                            "condition_id": condition_id,
                            "bearing_id": bearing,
                            "snapshot_index": snapshot_index,
                            "split": split,
                            "snapshot_interval_minutes": 1,
                            "rul_minutes": max(lifetime - age, 0.0),
                            **{
                                name: float(row_index + feature_index)
                                for feature_index, name in enumerate(FEATURE_COLUMN_NAMES)
                            },
                        }
                    )
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps({"status": "passed"}), encoding="utf-8")
            manifest_path = root / "manifest.csv"
            manifest_path.write_text("bearing_id,split\nT1,train\n", encoding="utf-8")
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model_version": "test-model",
                        "random_state": 2026,
                        "quantiles": [0.1, 0.5, 0.9],
                        "feature_sets": ["dual"],
                        "include_age_options": [True],
                        "ridge_alphas": [1.0],
                        "minimum_macro_mae_improvement_fraction": 0.0,
                        "minimum_validation_bearings_improved": 1,
                        "empirical_interval_coverage": 0.8,
                        "gradient_boosting_candidates": [
                            {
                                "learning_rate": 0.1,
                                "max_iter": 10,
                                "max_leaf_nodes": 5,
                                "min_samples_leaf": 2,
                                "l2_regularization": 1.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = run_selection(
                feature_path,
                audit_path,
                manifest_path,
                config_path,
            )

            self.assertEqual("not_evaluated", report["test_metrics_status"])
            self.assertNotIn("test_metrics", report)
            self.assertEqual(64, len(report["fingerprints"]["features_sha256"]))
            self.assertEqual(
                "ridge_empirical_interval",
                report["selected_candidate"]["estimator_family"],
            )
            self.assertIn(
                "interval_radius_minutes",
                report["selected_candidate"]["parameters"],
            )

    def test_selection_rejects_feature_digest_that_disagrees_with_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            feature_path = root / "features.csv"
            feature_path.write_text("not-the-audited-content\n", encoding="utf-8")
            audit_path = root / "audit.json"
            audit_path.write_text(
                json.dumps({"feature_artifact_sha256": "0" * 64}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(TrainingError, "digest"):
                run_selection(
                    feature_path,
                    audit_path,
                    root / "manifest.csv",
                    Path("configs/models/xjtu_sy_rul_v0_1.json"),
                )

    def test_select_command_writes_the_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "nested" / "selection.json"
            with mock.patch(
                "vibralens.modeling.training.run_selection",
                return_value={
                    "test_metrics_status": "not_evaluated",
                    "vibration_gate": {"passed": True},
                },
            ):
                exit_code = main(
                    [
                        "select",
                        "--features",
                        "features.csv",
                        "--feature-audit",
                        "audit.json",
                        "--manifest",
                        "manifest.csv",
                        "--config",
                        "config.json",
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(0, exit_code)
            self.assertEqual(
                {
                    "test_metrics_status": "not_evaluated",
                    "vibration_gate": {"passed": True},
                },
                json.loads(output.read_text(encoding="utf-8")),
            )

    def test_finalize_command_exports_the_frozen_selection(self) -> None:
        with mock.patch(
            "vibralens.modeling.training.finalize_model",
            return_value={"held_out_bearings": ["A", "B"]},
        ) as finalize:
            exit_code = main(
                [
                    "finalize",
                    "--features",
                    "features.csv",
                    "--feature-audit",
                    "audit.json",
                    "--manifest",
                    "manifest.csv",
                    "--config",
                    "config.json",
                    "--selection",
                    "selection.json",
                    "--bundle",
                    "model.joblib",
                    "--metadata",
                    "model.json",
                    "--test-report",
                    "test.json",
                ]
            )

        self.assertEqual(0, exit_code)
        finalize.assert_called_once_with(
            Path("features.csv"),
            Path("audit.json"),
            Path("manifest.csv"),
            Path("config.json"),
            Path("selection.json"),
            Path("model.joblib"),
            Path("model.json"),
            Path("test.json"),
        )

    def test_verify_command_does_not_require_a_test_report(self) -> None:
        with mock.patch(
            "vibralens.modeling.training.verify_model",
            return_value={"model_version": "test-model"},
        ) as verify:
            exit_code = main(
                [
                    "verify",
                    "--features",
                    "features.csv",
                    "--feature-audit",
                    "audit.json",
                    "--manifest",
                    "manifest.csv",
                    "--config",
                    "config.json",
                    "--selection",
                    "selection.json",
                    "--bundle",
                    "model.joblib",
                    "--metadata",
                    "model.json",
                ]
            )

        self.assertEqual(0, exit_code)
        verify.assert_called_once()


if __name__ == "__main__":
    unittest.main()
