from __future__ import annotations

import json
import unittest
from pathlib import Path

from vibralens.data.xjtu_sy import EXPECTED_SNAPSHOT_COUNTS, split_for_bearing


class SplitConfigTests(unittest.TestCase):
    def test_every_bearing_has_exactly_one_crule_role(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "splits"
            / "xjtu_sy_crule.json"
        )
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)

        assignments = {
            bearing: role
            for role, bearings in config["assignments"].items()
            for bearing in bearings
        }
        expected_bearings = {
            relative_path.split("/", 1)[1]
            for relative_path in EXPECTED_SNAPSHOT_COUNTS
        }

        self.assertEqual(15, sum(map(len, config["assignments"].values())))
        self.assertEqual(expected_bearings, set(assignments))
        self.assertEqual(
            {bearing: split_for_bearing(bearing) for bearing in expected_bearings},
            assignments,
        )


if __name__ == "__main__":
    unittest.main()
