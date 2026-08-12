"""Deterministic time- and frequency-domain vibration features."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from vibralens.data.xjtu_sy import EXPECTED_HEADER


DEFAULT_SAMPLING_RATE_HZ = 25_600
LOW_BAND_LIMIT_HZ = 1_000.0
MID_BAND_LIMIT_HZ = 5_000.0

CHANNEL_FEATURE_NAMES = (
    "mean",
    "std",
    "rms",
    "abs_peak",
    "peak_to_peak",
    "skewness",
    "kurtosis",
    "crest_factor",
    "dominant_frequency_hz",
    "spectral_centroid_hz",
    "spectral_entropy",
    "band_energy_low_fraction",
    "band_energy_mid_fraction",
    "band_energy_high_fraction",
)
FEATURE_COLUMN_NAMES = tuple(
    "{}_{}".format(channel, feature)
    for channel in ("horizontal", "vertical")
    for feature in CHANNEL_FEATURE_NAMES
)


class FeatureValidationError(ValueError):
    """Raised when a vibration snapshot cannot be safely processed."""


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def extract_channel_features(
    signal: Iterable[float], sampling_rate_hz: int = DEFAULT_SAMPLING_RATE_HZ
) -> Dict[str, float]:
    """Extract fixed, non-learned features from one vibration channel."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise FeatureValidationError("signal must be a one-dimensional non-empty series")
    if sampling_rate_hz <= 0:
        raise FeatureValidationError("sampling rate must be positive")
    if not np.isfinite(values).all():
        raise FeatureValidationError("signal contains non-finite values")

    mean = float(np.mean(values))
    centered = values - mean
    std = float(np.sqrt(np.mean(centered * centered)))
    rms = float(np.sqrt(np.mean(values * values)))
    abs_peak = float(np.max(np.abs(values)))
    peak_to_peak = float(np.max(values) - np.min(values))

    if std == 0.0:
        skewness = 0.0
        kurtosis = 0.0
    else:
        standardized = centered / std
        skewness = float(np.mean(standardized**3))
        kurtosis = float(np.mean(standardized**4))

    power = np.abs(np.fft.rfft(centered)) ** 2
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / sampling_rate_hz)
    total_power = float(np.sum(power))

    if total_power == 0.0:
        dominant_frequency_hz = 0.0
        spectral_centroid_hz = 0.0
        spectral_entropy = 0.0
        low_fraction = 0.0
        mid_fraction = 0.0
        high_fraction = 0.0
    else:
        probabilities = power / total_power
        dominant_frequency_hz = float(frequencies[int(np.argmax(power))])
        spectral_centroid_hz = float(np.sum(frequencies * probabilities))
        positive_probabilities = probabilities[probabilities > 0.0]
        entropy_denominator = math.log(probabilities.size)
        spectral_entropy = (
            float(-np.sum(positive_probabilities * np.log(positive_probabilities)))
            / entropy_denominator
            if entropy_denominator > 0.0
            else 0.0
        )
        low_fraction = float(
            np.sum(power[(frequencies >= 0.0) & (frequencies < LOW_BAND_LIMIT_HZ)])
            / total_power
        )
        mid_fraction = float(
            np.sum(
                power[
                    (frequencies >= LOW_BAND_LIMIT_HZ)
                    & (frequencies < MID_BAND_LIMIT_HZ)
                ]
            )
            / total_power
        )
        high_fraction = float(
            np.sum(power[frequencies >= MID_BAND_LIMIT_HZ]) / total_power
        )

    features = {
        "mean": mean,
        "std": std,
        "rms": rms,
        "abs_peak": abs_peak,
        "peak_to_peak": peak_to_peak,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "crest_factor": _safe_ratio(abs_peak, rms),
        "dominant_frequency_hz": dominant_frequency_hz,
        "spectral_centroid_hz": spectral_centroid_hz,
        "spectral_entropy": spectral_entropy,
        "band_energy_low_fraction": low_fraction,
        "band_energy_mid_fraction": mid_fraction,
        "band_energy_high_fraction": high_fraction,
    }
    if not all(math.isfinite(value) for value in features.values()):
        raise FeatureValidationError("feature extraction produced non-finite values")
    return features


def _load_snapshot(path: Path, expected_rows: Optional[int]) -> np.ndarray:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            header = stream.readline().rstrip("\r\n")
    except (OSError, UnicodeDecodeError) as error:
        raise FeatureValidationError(
            "{} does not have a readable UTF-8 header".format(path)
        ) from error
    if header != EXPECTED_HEADER:
        raise FeatureValidationError(
            "{} has invalid header {!r}".format(path, header)
        )

    try:
        values = np.loadtxt(
            path,
            delimiter=",",
            skiprows=1,
            dtype=np.float64,
            ndmin=2,
        )
    except (OSError, ValueError) as error:
        raise FeatureValidationError(
            "{} contains malformed numeric data".format(path)
        ) from error

    if values.ndim != 2 or values.shape[1] != 2:
        raise FeatureValidationError(
            "{} must contain exactly two columns".format(path)
        )
    if expected_rows is not None and values.shape[0] != expected_rows:
        raise FeatureValidationError(
            "{} has {} signal rows; expected {}".format(
                path, values.shape[0], expected_rows
            )
        )
    if not np.isfinite(values).all():
        raise FeatureValidationError(
            "{} contains non-finite signal values".format(path)
        )
    return values


def _features_for_values(values: np.ndarray, sampling_rate_hz: int) -> Dict[str, float]:
    features: Dict[str, float] = {}
    for column_index, channel in enumerate(("horizontal", "vertical")):
        channel_features = extract_channel_features(
            values[:, column_index], sampling_rate_hz=sampling_rate_hz
        )
        features.update(
            {
                "{}_{}".format(channel, name): value
                for name, value in channel_features.items()
            }
        )
    return features


def extract_snapshot_features(
    path: Path,
    expected_rows: Optional[int] = None,
    sampling_rate_hz: int = DEFAULT_SAMPLING_RATE_HZ,
) -> Dict[str, float]:
    """Load one two-channel CSV and return 28 deterministic features."""
    values = _load_snapshot(Path(path), expected_rows)
    return _features_for_values(values, sampling_rate_hz)


def _process_manifest_row(
    dataset_root: Path, row: Mapping[str, str]
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    try:
        expected_rows = int(row["n_signal_samples"])
        sampling_rate_hz = int(row["sampling_rate_hz"])
        relative_path = row["relative_path"]
    except (KeyError, ValueError) as error:
        raise FeatureValidationError(
            "manifest row is missing valid path, row-count, or sampling-rate fields"
        ) from error

    values = _load_snapshot(dataset_root / relative_path, expected_rows)
    features = _features_for_values(values, sampling_rate_hz)
    ranges = {
        "horizontal": (
            float(np.min(values[:, 0])),
            float(np.max(values[:, 0])),
        ),
        "vertical": (
            float(np.min(values[:, 1])),
            float(np.max(values[:, 1])),
        ),
    }
    return features, ranges


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def build_feature_artifact(
    dataset_root: Path,
    manifest_path: Path,
    output_directory: Path,
    *,
    workers: int = 1,
) -> Tuple[Path, Path]:
    """Extract all manifest snapshots and write deterministic CSV/JSON artifacts."""
    dataset_root = Path(dataset_root)
    manifest_path = Path(manifest_path)
    output_directory = Path(output_directory)
    if workers < 1:
        raise FeatureValidationError("workers must be at least one")
    if not dataset_root.is_dir():
        raise FeatureValidationError(
            "dataset root does not exist: {}".format(dataset_root)
        )

    try:
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            manifest_fieldnames = reader.fieldnames
            manifest_rows = list(reader)
    except OSError as error:
        raise FeatureValidationError(
            "cannot read manifest: {}".format(manifest_path)
        ) from error

    if not manifest_fieldnames or not manifest_rows:
        raise FeatureValidationError("manifest must contain at least one row")

    def process(row: Mapping[str, str]):
        return _process_manifest_row(dataset_root, row)

    if workers == 1:
        results = [process(row) for row in manifest_rows]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(process, manifest_rows))

    output_directory.mkdir(parents=True, exist_ok=True)
    feature_path = output_directory / "xjtu_sy_features.csv"
    audit_path = output_directory / "xjtu_sy_feature_audit.json"
    fieldnames = list(manifest_fieldnames) + list(FEATURE_COLUMN_NAMES)

    raw_minimums = {"horizontal": math.inf, "vertical": math.inf}
    raw_maximums = {"horizontal": -math.inf, "vertical": -math.inf}
    bearings = set()
    split_counts: Dict[str, int] = {}

    with feature_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row, (features, ranges) in zip(manifest_rows, results):
            writer.writerow(dict(row, **features))
            bearings.add(row.get("bearing_id", ""))
            split = row.get("split", "")
            split_counts[split] = split_counts.get(split, 0) + 1
            for channel in ("horizontal", "vertical"):
                raw_minimums[channel] = min(raw_minimums[channel], ranges[channel][0])
                raw_maximums[channel] = max(raw_maximums[channel], ranges[channel][1])

    audit = {
        "dataset": "XJTU-SY",
        "status": "passed",
        "snapshots": len(manifest_rows),
        "bearings": len(bearings),
        "feature_columns": len(FEATURE_COLUMN_NAMES),
        "feature_names": list(FEATURE_COLUMN_NAMES),
        "non_finite_values": 0,
        "raw_ranges": {
            channel: {
                "min": raw_minimums[channel],
                "max": raw_maximums[channel],
            }
            for channel in ("horizontal", "vertical")
        },
        "snapshots_by_split": dict(sorted(split_counts.items())),
        "feature_artifact_sha256": _sha256(feature_path),
    }
    with audit_path.open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, sort_keys=True)
        stream.write("\n")

    return feature_path, audit_path


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract deterministic XJTU-SY vibration features."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/data/xjtu_sy_manifest.csv"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/features"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    feature_path, audit_path = build_feature_artifact(
        args.dataset_root,
        args.manifest,
        args.output_directory,
        workers=args.workers,
    )
    print("Features written to {}; audit={}".format(feature_path, audit_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
