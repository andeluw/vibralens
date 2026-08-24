# VibraLens

VibraLens turns an XJTU-SY-format bearing vibration snapshot into an estimated remaining-useful-life (RUL) interval and a transparent planned-maintenance advisory. It includes the reproducible data and model pipeline, a versioned evaluated model, a JSON CLI, and a small HTTP API.

This repository contains:

- a full structural audit of the 15 XJTU-SY bearing runs and 9,216 vibration snapshots;
- deterministic train, validation, and test assignments at the complete-bearing level;
- absolute remaining-life labels in dataset minutes;
- strict two-channel signal validation;
- 28 deterministic time- and frequency-domain features;
- bearing-balanced model selection without test-bearing leakage;
- a versioned Ridge RUL model with an empirical interval;
- one inference service shared by the CLI and API.

Raw vibration data is intentionally kept outside Git.

## Quick setup

Requirements:

- Git;
- Docker with Compose;
- [uv](https://docs.astral.sh/uv/);

Clone the project and install its locked environment:

```bash
git clone https://github.com/andeluw/vibralens.git
cd vibralens
uv sync --locked
```

The committed model can be served without downloading the raw research dataset:

```bash
docker compose up --build
```

In another terminal:

```bash
uv run python scripts/generate_smoke_snapshot.py /tmp/vibralens-smoke.csv

curl --fail -X POST http://localhost:8000/predict \
  -F 'snapshot=@/tmp/vibralens-smoke.csv;type=text/csv' \
  -F 'bearing_age_minutes=100' \
  -F 'condition_id=1' \
  -F 'planned_break_minutes=60'
```

Stop it with `docker compose down`. See [docs/api.md](docs/api.md) for the HTTP and CLI contracts and [docs/model.md](docs/model.md) for the model evidence and limitations.

## Run the frontend

With the API running on port 8000, start the web workspace in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3003`, add a valid two-channel CSV snapshot, enter the
bearing context, and run the remaining-life estimate. To use a different API
host, copy `frontend/.env.example` to `frontend/.env.local` and change the URL.

## Reproduce the data and model pipeline

Full reproduction additionally requires all six XJTU-SY multipart RAR volumes and approximately 20 GB of storage for the archives, extracted dataset, and working headroom.

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
- a frozen vertical-feature Ridge model selected on three validation bearings;
- held-out evaluation on six untouched test bearings;
- finite-value validation and deterministic feature calculations;
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

The reported results are experimental evidence from XJTU-SY run-to-failure tests. They are not factory certification or a safety guarantee; unsupported operating conditions return an explicit abstention.
