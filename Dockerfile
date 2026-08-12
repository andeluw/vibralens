FROM ghcr.io/astral-sh/uv:0.11.15 AS uv
FROM python:3.11-slim

COPY --from=uv /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    VIBRALENS_MODEL_PATH=/app/artifacts/models/vibralens_rul_v0_1.joblib

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev

COPY artifacts/models ./artifacts/models

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "vibralens.api:app", "--host", "0.0.0.0", "--port", "8000"]
