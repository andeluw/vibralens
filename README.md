# VibraLens

VibraLens provides a reproducible pipeline for validating XJTU-SY rolling-bearing vibration data and converting it into a leakage-aware feature dataset for prognostics experiments.

This repository contains:

- a full structural audit of the 15 XJTU-SY bearing runs and 9,216 vibration snapshots;
- deterministic train, validation, and test assignments at the complete-bearing level;
- absolute remaining-life labels in dataset minutes;
- strict two-channel signal validation;
- 28 deterministic time- and frequency-domain features;
- automated tests for dataset, split, and feature contracts.

Raw vibration data is intentionally kept outside Git.

## Quick setup

Requirements:

- Git;
- [uv](https://docs.astral.sh/uv/);
- all six XJTU-SY multipart RAR volumes;
- approximately 20 GB of available local storage for the archives, extracted dataset, and working headroom.

Clone the project and install its locked environment:

```bash
git clone https://github.com/andeluw/vibralens.git
cd vibralens
uv sync --locked
uv run python -m unittest discover -s tests -v
```

Keep the raw dataset outside the repository. The recommended layout is:

```text
workspace/
├── vibralens/
└── XJTU-SY_Bearing_Datasets/
    └── Data/
        ├── XJTU-SY_Bearing_Datasets.part01.rar
        ├── ... part02 through part06 ...
        └── XJTU-SY_Bearing_Datasets/
            ├── 35Hz12kN/
            ├── 37.5Hz11kN/
            └── 40Hz10kN/
```

Place all six RAR volumes together and extract only `part01.rar` once with WinRAR or another compatible RAR application. The extractor automatically reads parts 02-06. Do not run each volume separately, and do not use `tar` for these RAR files.

Then run the full audit and feature extraction from `vibralens/`:

```bash
uv run vibralens-build-manifest \
  --dataset-root ../XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets \
  --output-directory artifacts/data

uv run vibralens-extract-features \
  --dataset-root ../XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets \
  --manifest artifacts/data/xjtu_sy_manifest.csv \
  --output-directory artifacts/features \
  --workers 8
```

See [docs/setup.md](docs/setup.md) for detailed macOS/Linux and Windows instructions, expected outputs, and troubleshooting.

## Verified outputs

- one row per XJTU-SY vibration snapshot;
- complete-bearing train, validation, and test assignments;
- absolute remaining life in observation intervals/minutes;
- full header and row-count validation for every CSV;
- deterministic time- and frequency-domain features for both vibration channels;
- finite-value validation and hand-checked feature calculations;
- no raw vibration data copied into Git.

The generated audit report is in `artifacts/data/xjtu_sy_audit.json`. The generated snapshot manifest is in `artifacts/data/xjtu_sy_manifest.csv`.

The feature contract is documented in `docs/feature-protocol.md`. Its audit report is tracked at `artifacts/features/xjtu_sy_feature_audit.json`; the reproducible feature CSV remains a local generated artifact.

## Reproduce the data audit

From this repository:

```bash
uv run vibralens-build-manifest \
  --dataset-root ../XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets \
  --output-directory artifacts/data
```

The command reads all CSV bodies by default. Use `--skip-row-verification` only for a faster path/count check; the abbreviated check does not provide full audit evidence.

## Run tests

```bash
uv run python -m unittest discover -s tests -v
```

## Extract vibration features

```bash
uv run vibralens-extract-features \
  --dataset-root ../XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets \
  --manifest artifacts/data/xjtu_sy_manifest.csv \
  --output-directory artifacts/features \
  --workers 8
```

## Dataset boundary

The raw dataset is external to this repository and must not be committed. Manifest paths are relative to the supplied dataset root so the audit is portable across machines.

See [docs/data-protocol.md](docs/data-protocol.md) for label semantics and split constraints, and [docs/feature-protocol.md](docs/feature-protocol.md) for exact feature definitions and limitations.
