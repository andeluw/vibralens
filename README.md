# VibraLens

VibraLens helps a maintenance supervisor assess whether a bearing is likely to
reach the next planned production stop. It converts one two-channel vibration
snapshot into an estimated remaining-useful-life (RUL) interval and a clear
maintenance advisory for human review.

The system is decision support, not autonomous machine control. Its evidence
comes from accelerated XJTU-SY bearing experiments and is not a factory safety
certification.

## What the MVP does

1. Accepts a two-channel vibration CSV plus the bearing age, operating
   condition, and time until the next planned stop.
2. Validates the complete signal and extracts 28 deterministic time- and
   frequency-domain features.
3. Runs a versioned Ridge model using condition, age, and 14 vertical-channel
   features.
4. Returns a pessimistic, median, and optimistic RUL estimate.
5. Compares the interval with the planned stop and recommends waiting,
   inspecting first, or prioritizing maintenance.
6. Abstains from estimating RUL when the operating condition is outside the
   evaluated model boundary.

```text
Browser workspace (:3003)
        |
        | multipart request via Axios
        v
FastAPI inference service (:8000)
        |
        +-- CSV and operating-context validation
        +-- deterministic vibration features
        +-- versioned Ridge model and empirical interval
        +-- planned-stop decision rule
        v
RUL interval + advisory + limitations
```

The frontend is a thin client: the Python service owns validation, inference,
abstention, interval handling, and advisory logic.

## Run the complete MVP locally

Requirements:

- Git;
- Docker with Compose;
- Node.js 22.13 or newer with npm;
- [uv](https://docs.astral.sh/uv/).

Clone the repository:

```bash
git clone https://github.com/andeluw/vibralens.git
cd vibralens
```

Start the API and load the committed model:

```bash
docker compose up --build
```

Confirm the model is ready at <http://localhost:8000/health>. The interactive
OpenAPI page is available at <http://localhost:8000/docs>.

In a second terminal, start the frontend:

```bash
cd frontend
npm ci
npm run dev
```

Open <http://localhost:3003>.

To generate a valid synthetic CSV for a pipeline check, run this from the
repository root:

```bash
uv sync --locked
uv run python scripts/generate_smoke_snapshot.py /tmp/vibralens-smoke.csv
```

Upload `/tmp/vibralens-smoke.csv` in the web interface. This sine/cosine file
checks the real validation, feature, model, API, and UI path, but its prediction
is not evidence about a physical bearing. Use an original XJTU-SY snapshot for
a data-backed demonstration.

Stop the API with:

```bash
docker compose down
```

See [the setup guide](docs/setup.md) for full dataset and model reproduction,
and [the API guide](docs/api.md) for the HTTP and CLI contracts.

## Dataset and operating conditions

VibraLens uses the public XJTU-SY accelerated-life bearing dataset provided by
Xi'an Jiaotong University and Changxing Sumyoung Technology. The dataset has 15
complete bearing runs across three experimental regimes:

| Operating condition | Shaft speed | Radial load |
|---|---:|---:|
| 1 | 2,100 rpm | 12 kN |
| 2 | 2,250 rpm | 11 kN |
| 3 | 2,400 rpm | 10 kN |

Download the raw files from the
[official XJTU-SY repository](https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets)
and cite the originating paper:

> B. Wang, Y. Lei, N. Li, and N. Li, "A Hybrid Prognostics Approach for
> Estimating Remaining Useful Life of Rolling Element Bearings," IEEE
> Transactions on Reliability, 69(1), 401-412, 2020.
> <https://doi.org/10.1109/TR.2018.2882682>

The provider describes the data as publicly available for prognostics research
and requests citation, but does not publish a standard license file in the
official repository. Raw data are therefore not redistributed here; users
should obtain them from the provider and follow the provider's terms.

## Evaluation protocol and evidence

All snapshots from a physical bearing remain in one role. Within each operating
condition, bearing indices 1-2 train the model, index 3 is used for validation
and model selection, and indices 4-5 form the held-out test set. This is one
fixed complete-bearing holdout, not rotating cross-validation.

The selected model passed a validation gate against an age-only baseline before
the six test bearings were evaluated:

| Metric | Result |
|---|---:|
| Validation age-only macro-bearing MAE | 775.916 min |
| Validation selected-model macro-bearing MAE | 317.883 min |
| Validation improvement | 59.0% |
| Held-out macro-bearing MAE | 347.010 min |
| Held-out macro-bearing interval coverage | 62.42% |
| Held-out mean interval width | 873.644 min |

The interval is an empirical validation-residual interval, not a calibrated
deployment guarantee. Its held-out coverage is lower than its 80.21% validation
coverage, and the final `Bearing1_4` snapshot is an important failure case: the
actual dataset RUL is 0 minutes while the model predicts 778.100 minutes. Full
aggregate and per-bearing evidence is retained in machine-readable artifacts.

## Reproduce the data and model pipeline

Full reproduction requires all six XJTU-SY multipart RAR volumes and about
20 GB of free storage. Keep the raw dataset outside this repository:

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

Place all six archive volumes together and extract `part01.rar` once. Then:

```bash
uv sync --locked

uv run vibralens-build-manifest \
  --dataset-root ../XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets \
  --output-directory artifacts/data

uv run vibralens-extract-features \
  --dataset-root ../XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets \
  --manifest artifacts/data/xjtu_sy_manifest.csv \
  --output-directory artifacts/features \
  --workers 8
```

The detailed selection, finalization, and verification commands are in
[docs/setup.md](docs/setup.md).

## Repository evidence

- `artifacts/data/xjtu_sy_audit.json`: full dataset-structure audit;
- `artifacts/data/xjtu_sy_manifest.csv`: portable snapshot identities, labels,
  and complete-bearing assignments;
- `artifacts/features/xjtu_sy_feature_audit.json`: feature counts, ranges, and
  digest;
- `artifacts/evaluation/xjtu_sy_rul_selection.json`: validation-only model
  selection evidence;
- `artifacts/evaluation/xjtu_sy_rul_test.json`: frozen held-out results and
  per-bearing diagnostics;
- `artifacts/models/vibralens_rul_v0_1.joblib` and `.json`: serving bundle and
  human-readable metadata.

## Technical documentation

- [Setup and full reproduction](docs/setup.md)
- [HTTP API and JSON CLI](docs/api.md)
- [Dataset, labels, and split](docs/data-protocol.md)
- [Vibration feature definitions](docs/feature-protocol.md)
- [Model selection, results, and limitations](docs/model.md)

## Scope and limitations

- Results come from 15 accelerated experimental bearing runs, not production
  factory deployments.
- Dataset minutes are observation intervals under the experiment and should not
  be treated as transferable factory operating minutes without validation.
- The fixed empirical interval does not provide a formal coverage or safety
  guarantee.
- The model supports only the three evaluated speed/load regimes and abstains
  outside them.
- Every advisory requires qualified human inspection and operating context.
