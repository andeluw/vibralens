"""Build a leakage-safe manifest for the XJTU-SY bearing dataset."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DATASET_NAME = "XJTU-SY"
EXPECTED_HEADER = (
    "Horizontal_vibration_signals,Vertical_vibration_signals"
)
SAMPLING_RATE_HZ = 25_600
SIGNAL_ROWS_PER_SNAPSHOT = 32_768
SNAPSHOT_DURATION_SECONDS = 1.28
SNAPSHOT_INTERVAL_MINUTES = 1

CONDITIONS = {
    "35Hz12kN": {"condition_id": 1, "shaft_speed_rpm": 2100, "radial_load_kn": 12.0},
    "37.5Hz11kN": {"condition_id": 2, "shaft_speed_rpm": 2250, "radial_load_kn": 11.0},
    "40Hz10kN": {"condition_id": 3, "shaft_speed_rpm": 2400, "radial_load_kn": 10.0},
}

EXPECTED_SNAPSHOT_COUNTS = {
    "35Hz12kN/Bearing1_1": 123,
    "35Hz12kN/Bearing1_2": 161,
    "35Hz12kN/Bearing1_3": 158,
    "35Hz12kN/Bearing1_4": 122,
    "35Hz12kN/Bearing1_5": 52,
    "37.5Hz11kN/Bearing2_1": 491,
    "37.5Hz11kN/Bearing2_2": 161,
    "37.5Hz11kN/Bearing2_3": 533,
    "37.5Hz11kN/Bearing2_4": 42,
    "37.5Hz11kN/Bearing2_5": 339,
    "40Hz10kN/Bearing3_1": 2538,
    "40Hz10kN/Bearing3_2": 2496,
    "40Hz10kN/Bearing3_3": 371,
    "40Hz10kN/Bearing3_4": 1515,
    "40Hz10kN/Bearing3_5": 114,
}

_BEARING_PATTERN = re.compile(r"^Bearing([1-3])_([1-5])$")


class DatasetValidationError(ValueError):
    """Raised when the local XJTU-SY copy violates the expected schema."""


@dataclass(frozen=True)
class ManifestRow:
    dataset: str
    condition_id: int
    condition_name: str
    shaft_speed_rpm: int
    radial_load_kn: float
    bearing_id: str
    bearing_index: int
    snapshot_index: int
    relative_path: str
    split: str
    n_signal_samples: int
    sampling_rate_hz: int
    snapshot_duration_seconds: float
    snapshot_interval_minutes: int
    rul_minutes: int


@dataclass(frozen=True)
class AuditSummary:
    dataset: str
    status: str
    conditions: int
    bearings: int
    csv_files: int
    signal_rows_per_csv: int
    expected_header: str
    rows_verified: bool
    bearing_counts_by_split: Dict[str, int]
    snapshot_counts_by_split: Dict[str, int]


def split_for_bearing(bearing_id: str) -> str:
    """Return the fixed CRULE development role for a complete bearing."""
    match = _BEARING_PATTERN.fullmatch(bearing_id)
    if match is None:
        raise DatasetValidationError("invalid bearing identifier: {}".format(bearing_id))

    bearing_index = int(match.group(2))
    if bearing_index <= 2:
        return "train"
    if bearing_index == 3:
        return "validation"
    return "test"


def _condition_metadata(condition_name: str) -> Mapping[str, object]:
    try:
        return CONDITIONS[condition_name]
    except KeyError as error:
        raise DatasetValidationError(
            "unknown operating-condition directory: {}".format(condition_name)
        ) from error


def _bearing_index(bearing_id: str) -> int:
    match = _BEARING_PATTERN.fullmatch(bearing_id)
    if match is None:
        raise DatasetValidationError("invalid bearing identifier: {}".format(bearing_id))
    return int(match.group(2))


def _verify_snapshot(path: Path, expected_signal_rows: int) -> None:
    with path.open("rb") as stream:
        raw_header = stream.readline()
        try:
            header = raw_header.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError as error:
            raise DatasetValidationError(
                "{} has a non-UTF-8 header".format(path)
            ) from error

        if header != EXPECTED_HEADER:
            raise DatasetValidationError(
                "{} has invalid header {!r}".format(path, header)
            )

        signal_rows = 0
        final_byte = b""
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            signal_rows += block.count(b"\n")
            final_byte = block[-1:]
        if final_byte and final_byte != b"\n":
            signal_rows += 1

    if signal_rows != expected_signal_rows:
        raise DatasetValidationError(
            "{} has {} signal rows; expected {}".format(
                path, signal_rows, expected_signal_rows
            )
        )


def _sort_bearing_path(relative_bearing_path: str) -> Tuple[int, int]:
    condition_name, bearing_id = relative_bearing_path.split("/", 1)
    condition_id = int(_condition_metadata(condition_name)["condition_id"])
    return condition_id, _bearing_index(bearing_id)


def _validate_expected_bearings(
    dataset_root: Path, expected_counts: Mapping[str, int]
) -> None:
    actual = {
        str(path.relative_to(dataset_root))
        for path in dataset_root.glob("*/*")
        if path.is_dir()
    }
    expected = set(expected_counts)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise DatasetValidationError(
            "bearing directories do not match the specification; "
            "missing={}, unexpected={}".format(missing, unexpected)
        )


def build_manifest(
    dataset_root: Path,
    *,
    expected_counts: Optional[Mapping[str, int]] = None,
    expected_signal_rows: int = SIGNAL_ROWS_PER_SNAPSHOT,
    verify_rows: bool = False,
) -> Tuple[List[ManifestRow], AuditSummary]:
    """Validate the dataset and return one deterministic row per snapshot."""
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        raise DatasetValidationError(
            "dataset root does not exist: {}".format(dataset_root)
        )

    counts = EXPECTED_SNAPSHOT_COUNTS if expected_counts is None else expected_counts
    if not counts:
        raise DatasetValidationError("expected bearing counts cannot be empty")
    _validate_expected_bearings(dataset_root, counts)

    rows: List[ManifestRow] = []
    bearing_counts_by_split = {"train": 0, "validation": 0, "test": 0}
    snapshot_counts_by_split = {"train": 0, "validation": 0, "test": 0}
    condition_names = set()

    for relative_bearing_path in sorted(counts, key=_sort_bearing_path):
        expected_count = counts[relative_bearing_path]
        condition_name, bearing_id = relative_bearing_path.split("/", 1)
        condition = _condition_metadata(condition_name)
        condition_id = int(condition["condition_id"])
        match = _BEARING_PATTERN.fullmatch(bearing_id)
        if match is None or int(match.group(1)) != condition_id:
            raise DatasetValidationError(
                "bearing {} does not belong to condition {}".format(
                    bearing_id, condition_name
                )
            )

        bearing_directory = dataset_root / relative_bearing_path
        snapshot_paths = list(bearing_directory.glob("*.csv"))
        try:
            snapshot_paths.sort(key=lambda path: int(path.stem))
        except ValueError as error:
            raise DatasetValidationError(
                "non-numeric CSV filename in {}".format(bearing_directory)
            ) from error

        actual_indices = [int(path.stem) for path in snapshot_paths]
        expected_indices = list(range(1, expected_count + 1))
        if actual_indices != expected_indices:
            raise DatasetValidationError(
                "{} expected snapshot indices 1..{}; found {} files".format(
                    relative_bearing_path, expected_count, len(actual_indices)
                )
            )

        split = split_for_bearing(bearing_id)
        bearing_counts_by_split[split] += 1
        snapshot_counts_by_split[split] += expected_count
        condition_names.add(condition_name)

        for snapshot_path, snapshot_index in zip(snapshot_paths, actual_indices):
            if verify_rows:
                _verify_snapshot(snapshot_path, expected_signal_rows)
            rows.append(
                ManifestRow(
                    dataset=DATASET_NAME,
                    condition_id=condition_id,
                    condition_name=condition_name,
                    shaft_speed_rpm=int(condition["shaft_speed_rpm"]),
                    radial_load_kn=float(condition["radial_load_kn"]),
                    bearing_id=bearing_id,
                    bearing_index=_bearing_index(bearing_id),
                    snapshot_index=snapshot_index,
                    relative_path=str(snapshot_path.relative_to(dataset_root)),
                    split=split,
                    n_signal_samples=expected_signal_rows,
                    sampling_rate_hz=SAMPLING_RATE_HZ,
                    snapshot_duration_seconds=SNAPSHOT_DURATION_SECONDS,
                    snapshot_interval_minutes=SNAPSHOT_INTERVAL_MINUTES,
                    rul_minutes=(expected_count - snapshot_index)
                    * SNAPSHOT_INTERVAL_MINUTES,
                )
            )

    audit = AuditSummary(
        dataset=DATASET_NAME,
        status="passed",
        conditions=len(condition_names),
        bearings=len(counts),
        csv_files=len(rows),
        signal_rows_per_csv=expected_signal_rows,
        expected_header=EXPECTED_HEADER,
        rows_verified=verify_rows,
        bearing_counts_by_split=bearing_counts_by_split,
        snapshot_counts_by_split=snapshot_counts_by_split,
    )
    return rows, audit


def write_artifacts(
    rows: Iterable[ManifestRow], audit: AuditSummary, output_directory: Path
) -> Tuple[Path, Path]:
    """Write deterministic CSV and JSON audit artifacts."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "xjtu_sy_manifest.csv"
    audit_path = output_directory / "xjtu_sy_audit.json"

    materialized_rows = list(rows)
    if not materialized_rows:
        raise DatasetValidationError("cannot write an empty manifest")

    fieldnames = list(asdict(materialized_rows[0]).keys())
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in materialized_rows)

    with audit_path.open("w", encoding="utf-8") as stream:
        json.dump(asdict(audit), stream, indent=2, sort_keys=True)
        stream.write("\n")

    return manifest_path, audit_path


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate XJTU-SY and build its leakage-safe manifest."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/data"),
    )
    parser.add_argument(
        "--skip-row-verification",
        action="store_true",
        help="Only validate paths and counts; do not read every CSV body.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    rows, audit = build_manifest(
        args.dataset_root,
        verify_rows=not args.skip_row_verification,
    )
    manifest_path, audit_path = write_artifacts(
        rows, audit, args.output_directory
    )
    print(
        "Validated {} CSV files across {} bearings; manifest={}; audit={}".format(
            audit.csv_files, audit.bearings, manifest_path, audit_path
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
