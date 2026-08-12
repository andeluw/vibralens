# VibraLens

VibraLens is an uncertainty-aware maintenance decision-support project for rolling-element bearings. The first milestone is a leakage-safe, reproducible XJTU-SY data layer; raw data is intentionally kept outside this repository.

## Current verified milestone

- one row per XJTU-SY vibration snapshot;
- complete-bearing train, validation, and test assignments;
- absolute remaining life in observation intervals/minutes;
- full header and row-count validation for every CSV;
- no raw vibration data copied into Git.

The generated audit report is in `artifacts/data/xjtu_sy_audit.json`. The generated snapshot manifest is in `artifacts/data/xjtu_sy_manifest.csv`.

## Reproduce the data audit

From this repository:

```bash
uv run vibralens-build-manifest \
  --dataset-root ../XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets \
  --output-directory artifacts/data
```

The command reads all CSV bodies by default. Use `--skip-row-verification` only for a faster path/count check; do not use the abbreviated check as final evidence.

## Run tests

```bash
uv run python -m unittest discover -s tests -v
```

## Dataset boundary

The raw dataset is external to this repository and must not be committed. Manifest paths are relative to the supplied dataset root, so another team member can reproduce the audit without using the original machine's absolute path.

See [docs/data-protocol.md](docs/data-protocol.md) for label semantics, split constraints, and limitations.
