"""Training lifecycle endpoints backed by the existing AutoML engine."""

import threading
from uuid import uuid4
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..automl.engine import AutoMLEngine
from ..config.settings import Settings, backend_settings
from .state import DATASETS, JOBS, PROJECT_ROOT

router = APIRouter()


def _write_leaderboard_chart(job_id: str, records: list[dict], output_dir: str) -> str:
    valid_records = [record for record in records if record.get("is_valid")]
    chart_path = Path(output_dir) / f"{job_id}_leaderboard.png"
    if valid_records:
        chart_data = pd.DataFrame(sorted(valid_records, key=lambda record: record.get("cv_mean_score", 0), reverse=True))
        figure, axis = plt.subplots(figsize=(10, 4.8), dpi=150)
        sns.set_theme(style="whitegrid", palette="deep")
        sns.barplot(
            data=chart_data,
            x="cv_mean_score",
            y="model_name",
            hue="model_name",
            legend=False,
            ax=axis,
            palette="dark:#1d4c3c",
        )
        axis.set_title("Cross-validation performance by algorithm", loc="left", fontweight="bold")
        axis.set_xlabel("Mean validation score")
        axis.set_ylabel("")
        figure.tight_layout()
        figure.savefig(chart_path, transparent=False, facecolor="#f7f4ec")
        plt.close(figure)
    return str(chart_path)


class TrainingStartRequest(BaseModel):
    dataset_id: str
    target_column: str = ""
    metric: str = "auto"
    task: str = "auto"
    cv_folds: int = Field(default=5, ge=2)
    optimization_trials: int = Field(default=15, ge=1)
    keep_identifier_columns: list[str] = Field(default_factory=list)


def _run_job(job_id: str, request: TrainingStartRequest) -> None:
    job = JOBS[job_id]
    try:
        job["status"] = "running"
        settings = Settings(
            n_trials=request.optimization_trials,
            cv_folds=request.cv_folds,
            primary_metric=request.metric,
            task=request.task,
            export_dir=str(PROJECT_ROOT / backend_settings.model_directory),
        )
        engine = AutoMLEngine(settings=settings)
        export_name = f"trained_pipeline_{job_id}"
        engine.fit(
            csv_path=DATASETS[request.dataset_id]["path"],
            target_column=request.target_column or None,
            export_filename=export_name,
            keep_identifier_columns=request.keep_identifier_columns,
            task=request.task if request.task != "auto" else None,
        )
        chart_path = None
        chart_error = None
        try:
            chart_path = _write_leaderboard_chart(
                job_id,
                engine.model_records_,
                str(PROJECT_ROOT / backend_settings.model_directory),
            )
        except Exception as exc:
            chart_error = str(exc)
        job.update({
            "status": "complete",
            "best_model": engine.best_model_name_,
            "best_params": engine.best_params_,
            "task_type": engine.task_info_.task_type if engine.task_info_ else None,
            "primary_metric": engine.task_info_.primary_metric if engine.task_info_ else None,
            "metrics": engine.holdout_metrics_.metrics if engine.holdout_metrics_ else {},
            "leaderboard": [model.__dict__ for model in engine.ranked_models_],
            "model_records": engine.model_records_,
            "optimization_trials": request.optimization_trials,
            "cv_folds": min(request.cv_folds, min((record["cv_folds"] for record in engine.model_records_ if record.get("is_valid")), default=request.cv_folds)),
            "final_training_time_seconds": engine.final_training_time_seconds_,
            "fairness_audit": engine.fairness_audit_.__dict__ if engine.fairness_audit_ else None,
            "generalization_audit": engine.generalization_audit_.__dict__ if engine.generalization_audit_ else None,
            "model_path": engine.export_path_,
            "excluded_identifier_columns": engine.profile_.identifier_columns if engine.profile_ else [],
            "identifier_reasons": engine.profile_.identifier_reasons if engine.profile_ else {},
            "features_used": engine.feature_columns_,
            "prediction_series": engine.prediction_series_ or [],
            "leaderboard_chart": chart_path,
            "leaderboard_chart_error": chart_error,
        })
    except Exception as exc:
        job.update({"status": "failed", "error": str(exc)})


@router.post("/training/start", status_code=202)
def start_training(request: TrainingStartRequest) -> dict:
    if request.dataset_id not in DATASETS:
        return {"status": "failed", "error": "Dataset not found."}
    job_id = uuid4().hex
    JOBS[job_id] = {"job_id": job_id, "status": "queued", "dataset_id": request.dataset_id}
    threading.Thread(target=_run_job, args=(job_id, request), daemon=True).start()
    return JOBS[job_id]


@router.get("/training/{job_id}")
def get_training_status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        return {"status": "failed", "error": "Training job not found."}
    return job
