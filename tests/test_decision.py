from __future__ import annotations

import math
import unittest

from vibralens.decision import assess_planned_break
from vibralens.modeling.bundle import RulInterval


class PlannedBreakDecisionTests(unittest.TestCase):
    def test_uses_interval_boundaries_literally(self) -> None:
        interval = RulInterval(pessimistic=25.0, median=42.0, optimistic=68.0)

        self.assertEqual(
            ("low", "safe_to_wait"),
            tuple(assess_planned_break(interval, 25.0)),
        )
        self.assertEqual(
            ("uncertain", "inspect_first"),
            tuple(assess_planned_break(interval, 25.01)),
        )
        self.assertEqual(
            ("uncertain", "inspect_first"),
            tuple(assess_planned_break(interval, 68.0)),
        )
        self.assertEqual(
            ("high", "maintenance_urgent"),
            tuple(assess_planned_break(interval, 68.01)),
        )

    def test_rejects_negative_and_non_finite_horizons(self) -> None:
        interval = RulInterval(pessimistic=25.0, median=42.0, optimistic=68.0)

        for horizon in (-1.0, math.nan, math.inf, -math.inf):
            with self.subTest(horizon=horizon):
                with self.assertRaisesRegex(ValueError, "planned break"):
                    assess_planned_break(interval, horizon)

    def test_rejects_malformed_interval_even_if_construction_was_bypassed(self) -> None:
        malformed = object.__new__(RulInterval)
        object.__setattr__(malformed, "pessimistic", 50.0)
        object.__setattr__(malformed, "median", 40.0)
        object.__setattr__(malformed, "optimistic", 60.0)

        with self.assertRaisesRegex(ValueError, "RUL interval"):
            assess_planned_break(malformed, 20.0)


if __name__ == "__main__":
    unittest.main()
