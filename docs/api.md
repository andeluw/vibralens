# VibraLens inference interfaces

The CLI and HTTP API call the same `PredictionService`, so feature extraction, abstention, interval handling, and planned-break advice have one implementation.

## JSON CLI

```bash
uv run vibralens-predict \
  --snapshot /tmp/vibralens-smoke.csv \
  --bearing-age-minutes 100 \
  --condition-id 1 \
  --planned-break-minutes 60 \
  --model artifacts/models/vibralens_rul_v0_1.joblib
```

The command writes one JSON object to standard output. Exit codes are:

| Code | Meaning |
|---:|---|
| 0 | Prediction or explicit unsupported-condition abstention |
| 2 | Invalid arguments, scalars, or snapshot CSV |
| 3 | Missing, corrupt, or incompatible model artifact |

Expected failures write to standard error and leave standard output empty.

## Start the HTTP API

```bash
docker compose up --build
```

The service listens at `http://localhost:8000`. Stop it with:

```bash
docker compose down
```

For local development without Docker:

```bash
VIBRALENS_MODEL_PATH=artifacts/models/vibralens_rul_v0_1.joblib \
  uv run uvicorn vibralens.api:app --reload
```

## `GET /health`

Returns HTTP 200 when the bundle is loaded:

```json
{"status":"ready","model_version":"vibralens-rul-0.1.0"}
```

If loading fails, it returns HTTP 503 with `status: "unavailable"` and never reports ready.

## `GET /model`

Returns the model version, supported condition IDs, selected feature set, age inclusion, exact ordered feature schema, and limitations. It returns HTTP 503 if the artifact is unavailable.

## `POST /predict`

Send multipart form data:

| Field | Type | Meaning |
|---|---|---|
| `snapshot` | CSV file | Exact two-channel, 32,768-row snapshot |
| `bearing_age_minutes` | number >= 0 | Observed age at the snapshot |
| `condition_id` | integer | XJTU-SY condition ID |
| `planned_break_minutes` | number >= 0 | Time until the next planned stop |

Example:

```bash
curl --fail -X POST http://localhost:8000/predict \
  -F 'snapshot=@/tmp/vibralens-smoke.csv;type=text/csv' \
  -F 'bearing_age_minutes=100' \
  -F 'condition_id=1' \
  -F 'planned_break_minutes=60'
```

A supported response has this shape:

```json
{
  "model_version": "vibralens-rul-0.1.0",
  "rul_minutes": {
    "pessimistic": 0.0,
    "median": 303.46,
    "optimistic": 757.96
  },
  "planned_break_minutes": 60.0,
  "break_risk": "uncertain",
  "advisory": "inspect_first",
  "support_status": "supported",
  "limitations": ["..."]
}
```

For a well-formed snapshot with an unsupported condition, HTTP 200 is deliberate: `rul_minutes` is `null`, risk is `unknown`, advisory is `inspect_first`, and support status is `unsupported`. The estimator is not called.

| Status | Meaning |
|---:|---|
| 200 | Prediction or explicit abstention |
| 422 | Invalid form value or snapshot schema/content |
| 503 | Model unavailable or incompatible |
| 500 | Unexpected inference failure; no prediction body is returned |

The service has no authentication and is intended for local evaluation. Add deployment controls before exposing it to a network.

## Browser access

The API allows the local VibraLens frontend at `http://localhost:3000` by
default. Set `VIBRALENS_ALLOWED_ORIGINS` to a comma-separated list of exact
origins when the frontend runs elsewhere.
