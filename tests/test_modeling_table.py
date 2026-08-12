from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from vibralens.features.vibration import FEATURE_COLUMN_NAMES
from vibralens.modeling.table import (
    ModelingDataError,
    build_model_matrix,
    load_feature_table,
)


class ModelingTableTests(unittest.TestCase):
    @staticmethod
    def _write_rows(path: Path, rows: list) -> None:
        fieldnames = [
            "condition_id",
            "bearing_id",
            "snapshot_index",
            "split",
            "snapshot_interval_minutes",
            "rul_minutes",
            *FEATURE_COLUMN_NAMES,
        ]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        **row,
                        **{
                            name: feature_index
                            for feature_index, name in enumerate(
                                FEATURE_COLUMN_NAMES
                            )
                        },
                    }
                )

    def test_loads_valid_rows_and_builds_dual_channel_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "features.csv"
            fieldnames = [
                "condition_id",
                "bearing_id",
                "snapshot_index",
                "split",
                "snapshot_interval_minutes",
                "rul_minutes",
                *FEATURE_COLUMN_NAMES,
            ]
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                for bearing_id, snapshot_index, split, rul, offset in (
                    ("Bearing1_1", 1, "train", 9, 0.0),
                    ("Bearing1_3", 2, "validation", 8, 1.0),
                ):
                    writer.writerow(
                        {
                            "condition_id": 1,
                            "bearing_id": bearing_id,
                            "snapshot_index": snapshot_index,
                            "split": split,
                            "snapshot_interval_minutes": 1,
                            "rul_minutes": rul,
                            **{
                                name: feature_index + offset
                                for feature_index, name in enumerate(
                                    FEATURE_COLUMN_NAMES
                                )
                            },
                        }
                    )

            table = load_feature_table(path)
            self.assertEqual((2, 28), table.vibration.shape)
            self.assertEqual([0.0, 1.0], table.age_minutes.tolist())
            self.assertEqual([9.0, 8.0], table.rul_minutes.tolist())
            self.assertEqual(["train", "validation"], table.splits.tolist())

            matrix = build_model_matrix(
                table,
                table.splits == "train",
                feature_set="dual",
                include_age=True,
            )
            self.assertEqual((1, 30), matrix.values.shape)
            self.assertEqual("condition_id", matrix.feature_names[0])
            self.assertEqual("age_minutes", matrix.feature_names[1])

    def test_rejects_one_bearing_assigned_to_multiple_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "features.csv"
            self._write_rows(
                path,
                [
                    {
                        "condition_id": 1,
                        "bearing_id": "Bearing1_1",
                        "snapshot_index": 1,
                        "split": "train",
                        "snapshot_interval_minutes": 1,
                        "rul_minutes": 1,
                    },
                    {
                        "condition_id": 2,
                        "bearing_id": "Bearing1_1",
                        "snapshot_index": 2,
                        "split": "train",
                        "snapshot_interval_minutes": 1,
                        "rul_minutes": 0,
                    },
                ],
            )

            with self.assertRaisesRegex(ModelingDataError, "multiple conditions"):
                load_feature_table(path)


if __name__ == "__main__":
    unittest.main()
