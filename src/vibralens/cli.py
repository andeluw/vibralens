"""JSON command-line adapter for VibraLens inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from vibralens.inference import (
    InferenceValidationError,
    PredictionRequest,
    PredictionService,
)
from vibralens.modeling.bundle import BundleError, load_bundle


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict bearing RUL from one XJTU-SY-format snapshot."
    )
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--bearing-age-minutes", required=True, type=float)
    parser.add_argument("--condition-id", required=True, type=int)
    parser.add_argument("--planned-break-minutes", required=True, type=float)
    parser.add_argument("--model", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    try:
        bundle = load_bundle(args.model)
        service = PredictionService(bundle)
        response = service.predict(
            PredictionRequest(
                snapshot_path=args.snapshot,
                bearing_age_minutes=args.bearing_age_minutes,
                condition_id=args.condition_id,
                planned_break_minutes=args.planned_break_minutes,
            )
        )
    except InferenceValidationError as error:
        print("validation error: {}".format(error), file=sys.stderr)
        return 2
    except BundleError as error:
        print("artifact error: {}".format(error), file=sys.stderr)
        return 3

    print(json.dumps(response.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
