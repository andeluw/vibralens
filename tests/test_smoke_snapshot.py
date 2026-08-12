from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from scripts.generate_smoke_snapshot import write_smoke_snapshot
from vibralens.data.xjtu_sy import (
    EXPECTED_HEADER,
    SIGNAL_ROWS_PER_SNAPSHOT,
)
from vibralens.features.vibration import (
    FEATURE_COLUMN_NAMES,
    extract_snapshot_features,
)


class SmokeSnapshotTests(unittest.TestCase):
    def test_generates_a_valid_deterministic_production_shape_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            first = Path(temporary_directory) / "first.csv"
            second = Path(temporary_directory) / "second.csv"

            write_smoke_snapshot(first)
            write_smoke_snapshot(second)

            lines = first.read_text(encoding="utf-8").splitlines()
            self.assertEqual(EXPECTED_HEADER, lines[0])
            self.assertEqual(SIGNAL_ROWS_PER_SNAPSHOT, len(lines) - 1)
            self.assertEqual(first.read_bytes(), second.read_bytes())

            features = extract_snapshot_features(
                first,
                expected_rows=SIGNAL_ROWS_PER_SNAPSHOT,
            )
            self.assertEqual(set(FEATURE_COLUMN_NAMES), set(features))
            self.assertTrue(all(math.isfinite(value) for value in features.values()))


if __name__ == "__main__":
    unittest.main()
