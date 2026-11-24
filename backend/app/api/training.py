"""Training lifecycle endpoints backed by the existing AutoML engine."""
from __future__ import annotations

import threading
from uuid import uuid4
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List

from fastapi import APIRouter
from fastapi import Depends
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional

from ..automl.engine import AutoMLEngine
from ..config.settings import Settings, backend_settings
from .state import DATASETS, JOBS, PROJECT_ROOT
from .state_owner import get_owner_id, set_owner_id
from .auth import get_owner_id as authenticated_owner_id

router = APIRouter()


class ModelComparisonGraphItem(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    k: int
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    noise_ratio: float
    composite_score: float = 0.0
    is_best: bool = False


class KSelectionCurveItem(BaseModel):
    k: int
    inertia: float
    silhouette_score: float
    davies_bouldin_score: float


class PcaScatterItem(BaseModel):
    label: str
    x: float
    y: float
    cluster: int
    split: str


class ClusterSummaryItem(BaseModel):
    cluster_id: int
    count: int
    percentage: float
    feature_means: Dict[str, float]


class ComparisonGraphsSchema(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_comparison: List[ModelComparisonGraphItem] = Field(default_factory=list)
    k_selection_curve: List[KSelectionCurveItem] = Field(default_factory=list)
    pca_scatter_2d: List[PcaScatterItem] = Field(default_factory=list)
    cluster_summary: List[ClusterSummaryItem] = Field(default_factory=list)


def _write_leaderboard_chart(
    job_id: str,
    records: List[dict],
    output_dir: str,
    primary_metric: str = "Score",
    best_model: str = "",
    task_type: str = "Automated Machine Learning",
) -> str:
    valid_records = [record for record in records if record.get("is_valid")]
    chart_path = Path(output_dir) / f"{job_id}_leaderboard.png"
    if not valid_records:
        return str(chart_path)

    metric_lower = primary_metric.lower()[4:] if primary_metric.lower().startswith("neg_") else primary_metric.lower()
    higher_better = metric_lower not in {"rmse", "mse", "mae", "root_mean_squared_error", "mean_squared_error", "mean_absolute_error"}

    # Sort so best model is at the top of the horizontal bar chart (highest y-position)
    sorted_records = sorted(
        valid_records,
        key=lambda record: record.get("cv_mean_score", 0),
        reverse=not higher_better,
    )
    chart_data = pd.DataFrame(sorted_records)

    sns.set_theme(style="whitegrid", font="sans-serif")
    sns.set_palette("deep")

    chart_count = len(chart_data)
    figure_height = max(8.5, 3.8 + chart_count * 0.42)
    figure, axes = plt.subplots(2, 2, figsize=(15, figure_height), dpi=180)
    figure.patch.set_facecolor("#f8fafc")

    scores = chart_data["cv_mean_score"].to_numpy(dtype=float)
    std_scores = chart_data["cv_std_score"].fillna(0).to_numpy(dtype=float)
    model_names = chart_data["model_name"].tolist()
    y_positions = np.arange(len(model_names))

    # Panel 1: cross-validation leaderboard with fold uncertainty.
    axis = axes[0, 0]
    colors = ["#14b8a6" if name == best_model else "#94a3b8" for name in model_names]
    axis.barh(y_positions, scores, xerr=std_scores, color=colors, height=0.58,
              edgecolor="#334155", linewidth=0.55, capsize=3,
              error_kw={"elinewidth": 1.1, "ecolor": "#475569"})
    axis.set_yticks(y_positions)
    axis.set_yticklabels([("★ " if name == best_model else "") + name for name in model_names],
                         fontsize=9.5, fontweight="semibold")
    axis.invert_yaxis()
    max_score = max(np.abs(scores).max(initial=1), 1e-9)
    for position, (score, std) in enumerate(zip(scores, std_scores)):
        axis.text(score + max_score * 0.018, position, f"{score:.4f} ± {std:.4f}",
                  va="center", fontsize=8.5, fontweight="bold", color="#0f172a")
    axis.set_title("01  Cross-validation leaderboard", loc="left", fontsize=12,
                   fontweight="bold", color="#0f172a", pad=12)
    axis.set_xlabel(f"Mean {primary_metric} ± fold std dev", fontsize=9, color="#475569")
    axis.grid(axis="x", linestyle="--", alpha=0.35)
    sns.despine(ax=axis)

    # Panel 2: training score vs cross-validation score.
    axis = axes[0, 1]
    train_scores = chart_data["train_score"].fillna(0).to_numpy(dtype=float)
    comparison = pd.DataFrame({
        "Model": model_names * 2,
        "Score": np.concatenate([train_scores, scores]),
        "Evaluation": ["Training"] * len(model_names) + ["Cross-validation"] * len(model_names),
    })
    sns.barplot(data=comparison, x="Score", y="Model", hue="Evaluation",
                ax=axis, palette={"Training": "#a78bfa", "Cross-validation": "#14b8a6"},
                saturation=0.88)
    axis.set_title("02  Fit vs. validation performance", loc="left", fontsize=12,
                   fontweight="bold", color="#0f172a", pad=12)
    axis.set_xlabel("Score", fontsize=9, color="#475569")
    axis.set_ylabel("")
    axis.legend(title="", frameon=False, ncol=2, loc="lower right", fontsize=8)
    axis.tick_params(axis="y", labelsize=9)
    axis.grid(axis="x", linestyle="--", alpha=0.35)
    sns.despine(ax=axis)

    # Panel 3: generalization gap / overfit risk.
    axis = axes[1, 0]
    gaps = chart_data["generalization_gap"].fillna(0).to_numpy(dtype=float)
    gap_colors = ["#ef4444" if gap >= 0.12 else "#f59e0b" if gap >= 0.06 else "#10b981"
                  for gap in gaps]
    axis.barh(y_positions, gaps, color=gap_colors, height=0.58,
              edgecolor="#334155", linewidth=0.55)
    axis.axvline(0.06, color="#d97706", linestyle="--", linewidth=1, alpha=0.8)
    axis.axvline(0.12, color="#dc2626", linestyle="--", linewidth=1, alpha=0.8)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(model_names, fontsize=9.5)
    axis.invert_yaxis()
    for position, gap in enumerate(gaps):
        axis.text(gap + max(gaps.max(initial=0.01) * 0.025, 0.003), position,
                  f"{gap:.4f}", va="center", fontsize=8.5, fontweight="bold", color="#0f172a")
    axis.set_title("03  Generalization gap", loc="left", fontsize=12,
                   fontweight="bold", color="#0f172a", pad=12)
    axis.set_xlabel("Training score − validation score", fontsize=9, color="#475569")
    axis.grid(axis="x", linestyle="--", alpha=0.35)
    sns.despine(ax=axis)

    # Panel 4: tuning budget and fit latency.
    axis = axes[1, 1]
    training_times = chart_data["training_time_seconds"].fillna(0).to_numpy(dtype=float)
    tuning_times = chart_data["optimization_time_seconds"].fillna(0).to_numpy(dtype=float)
    latency = pd.DataFrame({
        "Model": model_names * 2,
        "Seconds": np.concatenate([training_times, tuning_times]),
        "Stage": ["Cross-validation"] * len(model_names) + ["HPO tuning"] * len(model_names),
    })
    sns.barplot(data=latency, x="Seconds", y="Model", hue="Stage",
                ax=axis, palette={"Cross-validation": "#38bdf8", "HPO tuning": "#64748b"},
                saturation=0.88)
    axis.set_title("04  Compute and tuning latency", loc="left", fontsize=12,
                   fontweight="bold", color="#0f172a", pad=12)
    axis.set_xlabel("Elapsed time (seconds)", fontsize=9, color="#475569")
    axis.set_ylabel("")
    axis.legend(title="", frameon=False, ncol=2, loc="lower right", fontsize=8)
    axis.tick_params(axis="y", labelsize=9)
    axis.grid(axis="x", linestyle="--", alpha=0.35)
    sns.despine(ax=axis)

    figure.suptitle(
        f"MODEL COMPARISON REPORT  ·  {task_type.replace('_', ' ').title()}",
        x=0.055, y=0.985, ha="left", fontsize=16, fontweight="bold", color="#0f172a",
    )
    figure.text(0.055, 0.955, f"Primary metric: {primary_metric}  ·  {chart_count} candidates",
                fontsize=9.5, color="#64748b")
    figure.subplots_adjust(left=0.18, right=0.98, top=0.91, bottom=0.07, hspace=0.42, wspace=0.30)
    figure.savefig(chart_path, transparent=False, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    return str(chart_path)


class TrainingStartRequest(BaseModel):
    dataset_id: str
    target_column: str = ""
    metric: str = "auto"
    task: str = "auto"
    cv_folds: int = Field(default=5, ge=2)
    optimization_trials: int = Field(default=15, ge=1)
    keep_identifier_columns: List[str] = Field(default_factory=list)


def _run_job(job_id: str, request: TrainingStartRequest, owner_id: str) -> None:
    set_owner_id(owner_id)
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
            "comparison_graphs": engine.comparison_graphs_,
            "leaderboard_chart": chart_path,
            "leaderboard_chart_error": chart_error,
        })
    except Exception as exc:
        job.update({"status": "failed", "error": str(exc)})


@router.post("/training/start", status_code=202)
def start_training(
    request: TrainingStartRequest,
    owner_id: str = Depends(authenticated_owner_id),
) -> dict:
    if request.dataset_id not in DATASETS:
        return {"status": "failed", "error": "Dataset not found."}
    job_id = uuid4().hex
    JOBS[job_id] = {"job_id": job_id, "status": "queued", "dataset_id": request.dataset_id}
    set_owner_id(owner_id)
    threading.Thread(target=_run_job, args=(job_id, request, owner_id), daemon=True).start()
    return JOBS[job_id]


@router.get("/training/{job_id}")
def get_training_status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        return {"status": "failed", "error": "Training job not found."}
    return job
