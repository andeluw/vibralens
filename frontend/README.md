# VibraLens frontend

Operator-facing web interface for the VibraLens bearing remaining-life API.

## Run locally

Start the backend from the repository root:

```bash
docker compose up --build
```

Then start the frontend in a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open `http://localhost:3003`. The frontend expects the API at
`http://localhost:8000`; override `NEXT_PUBLIC_VIBRALENS_API_URL` when needed.

The status control reports whether the API and committed model are ready. If it
shows that the service is unavailable, confirm
`curl --fail http://localhost:8000/health` succeeds.

Generate a valid synthetic upload from the repository root with:

```bash
uv run python scripts/generate_smoke_snapshot.py /tmp/vibralens-smoke.csv
```

The synthetic snapshot verifies the complete software path but is not a
physical bearing example.

## Verify

```bash
npm run build
```

The UI accepts the same exact two-channel, 32,768-row CSV snapshot as the CLI
and API. Uploaded snapshots are processed by the backend's temporary-file flow
and removed after each request.
