# VibraLens setup guide

## 1. Prerequisites

Install:

- Git;
- `uv`, the Python environment and package manager;
- WinRAR or another extractor compatible with this multipart RAR archive.

The project supports Python 3.9 or newer. `uv` selects and manages the environment from `pyproject.toml` and `uv.lock`; a separately managed virtual environment is not required.

Allow approximately:

- 4.1 GB for the six compressed archive volumes;
- 11 GB for one extracted dataset tree;
- additional headroom for installation and generated artifacts.

Twenty gigabytes of free space is a practical minimum. Do not extract every archive volume independently: doing so creates repeated copies of the same 11 GB dataset.

### Install uv

Follow the official installation instructions at <https://docs.astral.sh/uv/getting-started/installation/>.

Confirm installation:

```bash
uv --version
```

## 2. Clone and install

```bash
git clone https://github.com/andeluw/vibralens.git
cd vibralens
uv sync --locked
```

`uv sync --locked` creates `.venv/` and installs the exact locked project dependencies without changing `uv.lock`.

## 3. Prepare XJTU-SY

Keep raw data outside the Git repository. A sibling directory makes the documented commands portable:

```text
workspace/
├── vibralens/
└── XJTU-SY_Bearing_Datasets/
    ├── Important References/
    ├── Photographs/
    └── Data/
        ├── XJTU-SY_Bearing_Datasets.part01.rar
        ├── XJTU-SY_Bearing_Datasets.part02.rar
        ├── XJTU-SY_Bearing_Datasets.part03.rar
        ├── XJTU-SY_Bearing_Datasets.part04.rar
        ├── XJTU-SY_Bearing_Datasets.part05.rar
        ├── XJTU-SY_Bearing_Datasets.part06.rar
        └── XJTU-SY_Bearing_Datasets/
```

To extract:

1. Put parts 01-06 in the same `Data/` directory.
2. Open or extract only `XJTU-SY_Bearing_Datasets.part01.rar`.
3. Let the application automatically consume parts 02-06.
4. Keep one unnumbered extracted `XJTU-SY_Bearing_Datasets/` directory.

The extracted dataset should contain exactly these condition directories:

```text
35Hz12kN/
37.5Hz11kN/
40Hz10kN/
```

Each condition must contain five bearing directories. If extraction creates directories ending in `-1`, `-2`, and so on, the archive was extracted repeatedly. Verify them before deletion and keep the unnumbered copy.

## 4. Build and validate the manifest

### macOS or Linux

From `vibralens/`:

```bash
dataset_root="../XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets"

uv run vibralens-build-manifest \
  --dataset-root "$dataset_root" \
  --output-directory artifacts/data
```

### Windows PowerShell

From `vibralens/`:

```powershell
$DatasetRoot = "..\XJTU-SY_Bearing_Datasets\Data\XJTU-SY_Bearing_Datasets"

uv run vibralens-build-manifest `
  --dataset-root $DatasetRoot `
  --output-directory artifacts/data
```

The successful full audit reports:

```text
Validated 9216 CSV files across 15 bearings
```

It creates:

- `artifacts/data/xjtu_sy_manifest.csv`;
- `artifacts/data/xjtu_sy_audit.json`.

The command reads every CSV body and verifies its header and 32,768 signal rows. `--skip-row-verification` is available for quick path checks but does not produce full audit evidence.

## 5. Extract deterministic features

### macOS or Linux

```bash
uv run vibralens-extract-features \
  --dataset-root "$dataset_root" \
  --manifest artifacts/data/xjtu_sy_manifest.csv \
  --output-directory artifacts/features \
  --workers 8
```

### Windows PowerShell

```powershell
uv run vibralens-extract-features `
  --dataset-root $DatasetRoot `
  --manifest artifacts/data/xjtu_sy_manifest.csv `
  --output-directory artifacts/features `
  --workers 8
```

Reduce `--workers` if the computer has fewer CPU cores or experiences resource pressure.

The command creates:

- `artifacts/features/xjtu_sy_features.csv` - 9,216 rows and 28 deterministic features;
- `artifacts/features/xjtu_sy_feature_audit.json` - counts, raw ranges, and artifact digest.

The feature CSV is derived data and intentionally ignored by Git. It can always be recreated from the external raw dataset and committed source code.

## 6. Verify the generated outputs

Expected verified dataset totals:

| Item | Count |
|---|---:|
| Operating conditions | 3 |
| Bearings | 15 |
| CSV snapshots | 9,216 |
| Samples per snapshot | 32,768 |
| Raw channels | 2 |
| Extracted features | 28 |

## 7. Reproduce model selection

The split is fixed by complete bearing: bearing indices 1-2 train, index 3 validates model choice, and indices 4-5 remain held out for the one-time final evaluation. Never choose a candidate from held-out results.

Run validation-only selection:

```bash
uv run vibralens-train-rul select \
  --features artifacts/features/xjtu_sy_features.csv \
  --feature-audit artifacts/features/xjtu_sy_feature_audit.json \
  --manifest artifacts/data/xjtu_sy_manifest.csv \
  --config configs/models/xjtu_sy_rul_v0_1.json \
  --output artifacts/evaluation/xjtu_sy_rul_selection.json
```

The frozen selection report records `test_metrics_status: not_evaluated`. Finalization refuses to overwrite existing outputs. On a clean reproduction, run it once only after committing or otherwise freezing the passing selection:

```bash
uv run vibralens-train-rul finalize \
  --features artifacts/features/xjtu_sy_features.csv \
  --feature-audit artifacts/features/xjtu_sy_feature_audit.json \
  --manifest artifacts/data/xjtu_sy_manifest.csv \
  --config configs/models/xjtu_sy_rul_v0_1.json \
  --selection artifacts/evaluation/xjtu_sy_rul_selection.json \
  --bundle artifacts/models/vibralens_rul_v0_1.joblib \
  --metadata artifacts/models/vibralens_rul_v0_1.json \
  --test-report artifacts/evaluation/xjtu_sy_rul_test.json
```

Verify fingerprints, bundle compatibility, and reproduced validation metrics without reading or rewriting the held-out report:

```bash
uv run vibralens-train-rul verify \
  --features artifacts/features/xjtu_sy_features.csv \
  --feature-audit artifacts/features/xjtu_sy_feature_audit.json \
  --manifest artifacts/data/xjtu_sy_manifest.csv \
  --config configs/models/xjtu_sy_rul_v0_1.json \
  --selection artifacts/evaluation/xjtu_sy_rul_selection.json \
  --bundle artifacts/models/vibralens_rul_v0_1.joblib \
  --metadata artifacts/models/vibralens_rul_v0_1.json
```

See [model.md](model.md) for the selected model, metrics, and limitations.

## 8. Run inference

Generate a synthetic production-shape input, then use either the CLI or API:

```bash
uv run python scripts/generate_smoke_snapshot.py /tmp/vibralens-smoke.csv

uv run vibralens-predict \
  --snapshot /tmp/vibralens-smoke.csv \
  --bearing-age-minutes 100 \
  --condition-id 1 \
  --planned-break-minutes 60 \
  --model artifacts/models/vibralens_rul_v0_1.joblib
```

Or start the API:

```bash
docker compose up --build
curl --fail http://localhost:8000/health
docker compose down
```

See [api.md](api.md) for request fields, response semantics, and status codes.

## Troubleshooting

### “Dataset root does not exist”

Pass the extracted directory containing `35Hz12kN`, `37.5Hz11kN`, and `40Hz10kN`, not the parent containing the RAR files.

### “Bearing directories do not match the specification”

The wrong directory was supplied, extraction is incomplete, or an unexpected directory exists inside the dataset tree. Confirm all six RAR volumes were together during extraction.

### Numbered extracted directories appear

The multipart archive was extracted more than once. Every extraction produces the complete dataset, not one sixth of it. Keep one verified copy.

### `tar` reports a block error

The files are multipart RAR archives, not tar archives. Use WinRAR or a compatible RAR extractor and start from `part01.rar`.

### Feature extraction is slow

The command parses more than 300 million two-channel sample rows and performs two FFTs per snapshot. One to several minutes can be normal depending on CPU and storage speed.

### Raw data appears in Git status

Raw data should not live inside the repository. Move it to a sibling directory. The root-level `data/` path is ignored only as an additional safeguard.
