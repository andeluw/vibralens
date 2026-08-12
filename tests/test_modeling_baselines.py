from __future__ import annotations

import unittest

import numpy as np

from vibralens.modeling.baselines import AgeOnlyRegressor, BaselineError


class AgeOnlyBaselineTests(unittest.TestCase):
    def test_predicts_from_median_training_lifetime_by_condition(self) -> None:
        model = AgeOnlyRegressor().fit(
            ages_minutes=np.array([0.0, 2.0, 0.0, 4.0]),
            condition_ids=np.array([1, 1, 2, 2]),
            y=np.array([10.0, 8.0, 20.0, 16.0]),
            bearing_ids=np.array(["A", "A", "B", "B"]),
        )

        self.assertEqual({1: 10.0, 2: 20.0}, model.lifetime_by_condition)
        np.testing.assert_allclose(
            [7.0, 0.0],
            model.predict(
                ages_minutes=np.array([3.0, 25.0]),
                condition_ids=np.array([1, 2]),
            ),
        )

    def test_rejects_inconsistent_lifetime_labels_within_a_bearing(self) -> None:
        with self.assertRaisesRegex(BaselineError, "inconsistent lifetime"):
            AgeOnlyRegressor().fit(
                ages_minutes=np.array([0.0, 1.0]),
                condition_ids=np.array([1, 1]),
                y=np.array([10.0, 8.0]),
                bearing_ids=np.array(["A", "A"]),
            )

    def test_rejects_unseen_condition_at_prediction_time(self) -> None:
        model = AgeOnlyRegressor().fit(
            ages_minutes=np.array([0.0]),
            condition_ids=np.array([1]),
            y=np.array([10.0]),
            bearing_ids=np.array(["A"]),
        )

        with self.assertRaisesRegex(BaselineError, "unsupported condition"):
            model.predict(np.array([1.0]), np.array([2]))


if __name__ == "__main__":
    unittest.main()
