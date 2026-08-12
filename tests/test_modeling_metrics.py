from __future__ import annotations

import unittest

import numpy as np

from vibralens.modeling.metrics import (
    evaluate_interval_predictions,
    evaluate_point_predictions,
)


class ModelingMetricTests(unittest.TestCase):
    def test_reports_snapshot_and_macro_bearing_errors_separately(self) -> None:
        report = evaluate_point_predictions(
            y_true=np.array([0.0, 2.0, 4.0, 10.0]),
            y_pred=np.array([1.0, 1.0, 5.0, 14.0]),
            bearing_ids=np.array(["A", "A", "A", "B"]),
            ages_minutes=np.array([4.0, 2.0, 0.0, 0.0]),
        )

        self.assertAlmostEqual(2.5, report["macro_bearing_mae"])
        self.assertAlmostEqual(1.75, report["snapshot_mae"])
        self.assertAlmostEqual(0.325, report["macro_bearing_nmae"])
        self.assertEqual({"A", "B"}, set(report["per_bearing"]))

    def test_reports_empirical_interval_coverage_and_width(self) -> None:
        report = evaluate_interval_predictions(
            y_true=np.array([0.0, 2.0, 4.0, 10.0]),
            lower=np.array([0.0, 1.0, 3.0, 11.0]),
            upper=np.array([2.0, 3.0, 5.0, 13.0]),
            bearing_ids=np.array(["A", "A", "A", "B"]),
        )

        self.assertAlmostEqual(0.75, report["snapshot_coverage"])
        self.assertAlmostEqual(0.5, report["macro_bearing_coverage"])
        self.assertAlmostEqual(2.0, report["mean_width_minutes"])


if __name__ == "__main__":
    unittest.main()
