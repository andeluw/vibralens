"""Generate a deterministic, synthetic production-shape vibration snapshot."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Sequence

from vibralens.data.xjtu_sy import (
    EXPECTED_HEADER,
    SAMPLING_RATE_HZ,
    SIGNAL_ROWS_PER_SNAPSHOT,
)


def write_smoke_snapshot(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(EXPECTED_HEADER + "\n")
        for index in range(SIGNAL_ROWS_PER_SNAPSHOT):
            time_seconds = index / SAMPLING_RATE_HZ
            horizontal = math.sin(2.0 * math.pi * 120.0 * time_seconds)
            vertical = 0.5 * math.cos(2.0 * math.pi * 240.0 * time_seconds)
            stream.write("{:.12f},{:.12f}\n".format(horizontal, vertical))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate one synthetic VibraLens smoke-test CSV."
    )
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    write_smoke_snapshot(args.output)
    print("Smoke snapshot written to {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
