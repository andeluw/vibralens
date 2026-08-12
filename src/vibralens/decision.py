"""Transparent maintenance advice derived from a predicted RUL interval."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

from vibralens.modeling.bundle import RulInterval


@dataclass(frozen=True)
class DecisionAssessment:
    break_risk: str
    advisory: str

    def __iter__(self) -> Iterator[str]:
        yield self.break_risk
        yield self.advisory


def assess_planned_break(
    interval: RulInterval,
    horizon_minutes: float,
) -> DecisionAssessment:
    """Compare a planned maintenance horizon with the complete RUL interval."""
    try:
        horizon = float(horizon_minutes)
    except (TypeError, ValueError) as error:
        raise ValueError("planned break must be a finite non-negative number") from error
    if not math.isfinite(horizon) or horizon < 0.0:
        raise ValueError("planned break must be a finite non-negative number")

    try:
        values = (
            float(interval.pessimistic),
            float(interval.median),
            float(interval.optimistic),
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("RUL interval is malformed") from error
    if (
        not all(math.isfinite(value) and value >= 0.0 for value in values)
        or not values[0] <= values[1] <= values[2]
    ):
        raise ValueError("RUL interval is malformed")

    if horizon <= values[0]:
        return DecisionAssessment("low", "safe_to_wait")
    if horizon <= values[2]:
        return DecisionAssessment("uncertain", "inspect_first")
    return DecisionAssessment("high", "maintenance_urgent")
