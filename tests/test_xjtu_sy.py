from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from vibralens.data.xjtu_sy import (
    DatasetValidationError,
    build_manifest,
    split_for_bearing,
    write_artifacts,
)


EXPECTED_HEADER = "Horizontal_vibration_signals,Vertical_vibration_signals\n"


class XjtuSyManifestTests(unittest.TestCase):
    def _write_snapshot(
        self,
        root: Path,
        condition: str,
        bearing: str,
        index: int,
        *,
        rows: int = 3,
        header: str = EXPECTED_HEADER,
    ) -> None:
        path = root / condition / bearing / f"{index}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        values = "".join(f"{i / 10},{i / 20}\n" for i in range(rows))
        path.write_text(header + values, encoding="utf-8")

    def test_assigns_crule_split_by_complete_bearing(self) -> None:
        self.assertEqual("train", split_for_bearing("Bearing1_1"))
        self.assertEqual("train", split_for_bearing("Bearing2_2"))
        self.assertEqual("validation", split_for_bearing("Bearing3_3"))
        self.assertEqual("test", split_for_bearing("Bearing1_4"))
        self.assertEqual("test", split_for_bearing("Bearing2_5"))

    def test_builds_ordered_manifest_with_absolute_rul_minutes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            condition = "35Hz12kN"
            bearing = "Bearing1_1"
            self._write_snapshot(root, condition, bearing, 1)
            self._write_snapshot(root, condition, bearing, 2)

            rows, audit = build_manifest(
                root,
                expected_counts={f"{condition}/{bearing}": 2},
                expected_signal_rows=3,
                verify_rows=True,
            )

            self.assertEqual([1, 2], [row.snapshot_index for row in rows])
            self.assertEqual([1, 0], [row.rul_minutes for row in rows])
            self.assertEqual(["train", "train"], [row.split for row in rows])
            self.assertEqual(
                [
                    "35Hz12kN/Bearing1_1/1.csv",
                    "35Hz12kN/Bearing1_1/2.csv",
                ],
                [row.relative_path for row in rows],
            )
            self.assertEqual(2, audit.csv_files)
            self.assertEqual(1, audit.bearings)
            self.assertTrue(audit.rows_verified)

    def test_rejects_a_gap_in_snapshot_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            condition = "35Hz12kN"
            bearing = "Bearing1_1"
            self._write_snapshot(root, condition, bearing, 1)
            self._write_snapshot(root, condition, bearing, 3)

            with self.assertRaisesRegex(
                DatasetValidationError, "expected snapshot indices"
            ):
                build_manifest(
                    root,
                    expected_counts={f"{condition}/{bearing}": 2},
                    expected_signal_rows=3,
                    verify_rows=True,
                )

    def test_rejects_an_invalid_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            condition = "35Hz12kN"
            bearing = "Bearing1_1"
            self._write_snapshot(
                root,
                condition,
                bearing,
                1,
                header="vertical,horizontal\n",
            )

            with self.assertRaisesRegex(DatasetValidationError, "header"):
                build_manifest(
                    root,
                    expected_counts={f"{condition}/{bearing}": 1},
                    expected_signal_rows=3,
                    verify_rows=True,
                )

    def test_rejects_an_invalid_signal_row_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            condition = "35Hz12kN"
            bearing = "Bearing1_1"
            self._write_snapshot(root, condition, bearing, 1, rows=2)

            with self.assertRaisesRegex(DatasetValidationError, "signal rows"):
                build_manifest(
                    root,
                    expected_counts={f"{condition}/{bearing}": 1},
                    expected_signal_rows=3,
                    verify_rows=True,
                )

    def test_writes_stable_csv_and_json_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "dataset"
            output = Path(temporary_directory) / "artifacts"
            condition = "35Hz12kN"
            bearing = "Bearing1_1"
            self._write_snapshot(root, condition, bearing, 1)

            rows, audit = build_manifest(
                root,
                expected_counts={f"{condition}/{bearing}": 1},
                expected_signal_rows=3,
                verify_rows=True,
            )
            manifest_path, audit_path = write_artifacts(rows, audit, output)

            with manifest_path.open(newline="", encoding="utf-8") as stream:
                manifest_rows = list(csv.DictReader(stream))
            with audit_path.open(encoding="utf-8") as stream:
                audit_data = json.load(stream)

            self.assertEqual("0", manifest_rows[0]["rul_minutes"])
            self.assertEqual("Bearing1_1", manifest_rows[0]["bearing_id"])
            self.assertEqual(1, audit_data["bearings"])
            self.assertEqual(1, audit_data["csv_files"])
            self.assertEqual("passed", audit_data["status"])


if __name__ == "__main__":
    unittest.main()
