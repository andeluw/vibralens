"""Thin synchronous HTTP adapter for canonical VibraLens inference."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from vibralens.inference import (
    InferenceValidationError,
    PredictionRequest,
    PredictionService,
)
from vibralens.modeling.bundle import BundleError, load_bundle


DEFAULT_MODEL_PATH = Path("artifacts/models/vibralens_rul_v0_1.joblib")


def create_app(
    model_path: Path,
    *,
    service: Optional[PredictionService] = None,
) -> FastAPI:
    application = FastAPI(title="VibraLens", version="0.1.0")
    allowed_origins = [
        origin.strip()
        for origin in os.environ.get(
            "VIBRALENS_ALLOWED_ORIGINS",
            "http://localhost:3000",
        ).split(",")
        if origin.strip()
    ]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    active_service = service
    load_error: Optional[str] = None
    if active_service is None:
        try:
            active_service = PredictionService(load_bundle(Path(model_path)))
        except (BundleError, ValueError) as error:
            load_error = str(error)

    @application.get("/health")
    def health() -> object:
        if active_service is None:
            return JSONResponse(
                status_code=503,
                content={"status": "unavailable"},
            )
        return {
            "status": "ready",
            "model_version": active_service.bundle.model_version,
        }

    @application.get("/model")
    def model() -> object:
        if active_service is None:
            raise HTTPException(status_code=503, detail="model unavailable")
        bundle = active_service.bundle
        raw_limitations = bundle.metadata.get("limitations", ())
        return {
            "model_version": bundle.model_version,
            "supported_condition_ids": list(bundle.supported_condition_ids),
            "feature_set": bundle.feature_set,
            "include_age": bundle.include_age,
            "feature_names": list(bundle.feature_names),
            "limitations": list(raw_limitations),
        }

    @application.post("/predict")
    async def predict(
        snapshot: UploadFile = File(...),
        bearing_age_minutes: float = Form(..., ge=0.0),
        condition_id: int = Form(...),
        planned_break_minutes: float = Form(..., ge=0.0),
    ) -> object:
        if active_service is None:
            raise HTTPException(status_code=503, detail="model unavailable")

        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".csv",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                while True:
                    chunk = await snapshot.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)

            response = active_service.predict(
                PredictionRequest(
                    snapshot_path=temporary_path,
                    bearing_age_minutes=bearing_age_minutes,
                    condition_id=condition_id,
                    planned_break_minutes=planned_break_minutes,
                )
            )
            return response.to_dict()
        except InferenceValidationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except BundleError as error:
            raise HTTPException(status_code=503, detail="model unavailable") from error
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(status_code=500, detail="prediction failed") from error
        finally:
            await snapshot.close()
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    application.state.model_load_error = load_error
    return application


app = create_app(
    Path(os.environ.get("VIBRALENS_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
)
