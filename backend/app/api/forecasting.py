"""Fixed-origin multi-step forecasting endpoint."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4
import joblib
from typing import Optional

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from ..models.forecasting import DirectTimeSeriesForecaster
from ..config.settings import backend_settings
from ..export.model_exporter import ModelExporter
from .state import DATASETS, JOBS, PROJECT_ROOT

router = APIRouter()


class ForecastRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    dataset_id: str
    target_column: str
    datetime_column: Optional[str] = None
    frequency: Optional[str] = None
    horizon: int = Field(default=7, ge=1, le=90)
    n_trials: int = Field(default=10, ge=1, le=50)
    cv_folds: int = Field(default=3, ge=2, le=8)
    random_seed: int = 42
    model_family: str = Field(default="ridge")


@router.post("/forecast/run")
def run_forecast(request: ForecastRequest) -> dict:
    dataset = DATASETS.get(request.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    try:
        frame = pd.read_csv(Path(dataset["path"]))
        forecaster = DirectTimeSeriesForecaster(
            horizon=request.horizon,
            n_trials=request.n_trials,
            cv_folds=request.cv_folds,
            random_state=request.random_seed,
            model_family=request.model_family,
        )
        result = forecaster.fit_evaluate(
            frame,
            target=request.target_column,
            datetime_column=request.datetime_column,
            frequency=request.frequency,
        )
        model_id = uuid4().hex
        artifact_path = ModelExporter(str(PROJECT_ROOT / backend_settings.model_directory)).export(
            pipeline=forecaster,
            model_name=f"DirectTimeSeriesForecaster({request.model_family.upper()})",
            filename=f"forecast_{model_id}",
            metadata={
                "artifact_type": "direct_time_series_forecaster",
                "task_type": "time_series_forecasting",
                "forecast_config": result["config"],
                "forecast_metrics": result["metrics"],
                "training_rows": len(frame),
                "feature_schema": [request.target_column, request.datetime_column] if request.datetime_column else [request.target_column],
            },
        )
        JOBS[model_id] = {
            "job_id": model_id,
            "status": "complete",
            "best_model": f"DirectTimeSeriesForecaster({request.model_family.upper()})",
            "model_path": artifact_path,
            "metrics": result["metrics"],
            "forecast_config": result["config"],
        }
        result["forecast_id"] = model_id
        result["artifact_file"] = Path(artifact_path).name
        return result
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/forecasts/{forecast_id}/download")
def download_forecast(forecast_id: str):
    record = JOBS.get(forecast_id)
    artifact = record.get("model_path") if record and record.get("forecast_config") else None
    if not artifact:
        raise HTTPException(status_code=404, detail="Forecast model not found.")
    from fastapi.responses import FileResponse
    return FileResponse(artifact, filename=f"forecast_{forecast_id}.joblib", media_type="application/octet-stream")


@router.post("/forecasts/{forecast_id}/predict")
def predict_forecast(forecast_id: str, history_file: Optional[UploadFile] = File(None), horizon: Optional[int] = Form(None)):
    record = JOBS.get(forecast_id)
    artifact = record.get("model_path") if record and record.get("forecast_config") else None
    if not artifact:
        raise HTTPException(status_code=404, detail="Forecast model not found.")
    if horizon is not None and not 1 <= horizon <= 90:
        raise HTTPException(status_code=422, detail="horizon must be from 1 to 90.")
    try:
        forecaster = joblib.load(artifact)
        history = pd.read_csv(history_file.file) if history_file is not None else None
        predictions = forecaster.predict_future(history, horizon=horizon)
        return {"forecast_id": forecast_id, "predictions": predictions}
    except (ValueError, RuntimeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
