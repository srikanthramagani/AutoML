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


def _write_leaderboard_chart(
    job_id: str,
    records: list[dict],
    output_dir: str,
    primary_metric: str = "Score",
    best_model: str = "",
    task_type: str = "Automated Machine Learning",
) -> str:
    valid_records = [record for record in records if record.get("is_valid")]
    chart_path = Path(output_dir) / f"{job_id}_leaderboard.png"
    if not valid_records:
        return str(chart_path)

    metric_lower = primary_metric.lower().removeprefix("neg_")
    higher_better = metric_lower not in {"rmse", "mse", "mae", "root_mean_squared_error", "mean_squared_error", "mean_absolute_error"}

    # Sort so best model is at the top of the horizontal bar chart (highest y-position)
    sorted_records = sorted(
        valid_records,
        key=lambda record: record.get("cv_mean_score", 0),
        reverse=not higher_better,
    )
    chart_data = pd.DataFrame(sorted_records)

    figure, axis = plt.subplots(figsize=(10.5, max(4.5, len(chart_data) * 0.52)), dpi=180)
    sns.set_theme(style="white", font="sans-serif")

    colors = []
    for m in chart_data["model_name"]:
        if m == best_model:
            colors.append("#0d9488")  # Highlight winner in teal/emerald
        else:
            colors.append("#3b82f6")  # High-contrast modern indigo/blue

    y_positions = list(range(len(chart_data)))
    scores = chart_data["cv_mean_score"].to_numpy()
    std_scores = [float(r.get("cv_std_score", 0) or 0) for _, r in chart_data.iterrows()]

    bars = axis.barh(
        y_positions,
        scores,
        xerr=std_scores,
        color=colors,
        alpha=0.92,
        height=0.62,
        edgecolor="#1e293b",
        linewidth=0.8,
        capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": "#475569"},
    )

    axis.set_yticks(y_positions)
    model_labels = []
    for m in chart_data["model_name"]:
        if m == best_model:
            model_labels.append(f"★ {m} (Winner)")
        else:
            model_labels.append(f"  {m}")
    axis.set_yticklabels(model_labels, fontsize=10.5, fontweight="600")

    max_val = max(np.abs(scores)) if len(scores) else 1.0
    for bar, score, std in zip(bars, scores, std_scores):
        width = bar.get_width()
        label_text = f" {score:.4f} ± {std:.4f}"
        axis.text(
            width + (max_val * 0.02 if width >= 0 else -max_val * 0.02),
            bar.get_y() + bar.get_height() / 2,
            label_text,
            va="center",
            ha="left" if width >= 0 else "right",
            fontsize=9.5,
            fontweight="bold",
            color="#0f172a",
        )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#cbd5e1")
    axis.spines["bottom"].set_color("#cbd5e1")
    axis.grid(axis="x", linestyle="--", alpha=0.35, color="#94a3b8")

    axis.set_title(
        f"Cross-Validation Model Leaderboard · {task_type.replace('_', ' ').title()}",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=14,
        color="#0f172a",
    )
    axis.set_xlabel(f"Cross-Validation Mean ({primary_metric}) ± Fold Std Dev", fontsize=10.5, labelpad=8, color="#334155")

    figure.tight_layout()
    figure.savefig(chart_path, transparent=False, facecolor="#ffffff", bbox_inches="tight")
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
                primary_metric=engine.task_info_.primary_metric if engine.task_info_ else "Score",
                best_model=engine.best_model_name_ or "",
                task_type=engine.task_info_.task_type if engine.task_info_ else "Automated Machine Learning",
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
            "explainability": engine.explainability_,
            "unsupervised_visualization": engine.unsupervised_visualization_,
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
