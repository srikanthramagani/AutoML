"""Evaluation reliability report and corrected-evaluation endpoints."""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, List

import joblib
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..evaluation.reliability_auditor import run_corrected_evaluation
from .auth import get_owner_id
from .state import DATASETS, JOBS

router = APIRouter()


class CorrectedEvaluationRequest(BaseModel):
    excluded_features: List[str] = Field(default_factory=list)


@router.get("/audit/{audit_id}")
def get_audit(audit_id: str, _owner_id: str = Depends(get_owner_id)) -> dict:
    job = JOBS.get(audit_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Audit not found.")
    if not job.get("evaluation_audit"):
        raise HTTPException(status_code=422, detail=job.get("audit_error") or "No audit report is available for this experiment.")
    return job["evaluation_audit"]


@router.get("/audit/{audit_id}/report")
def export_audit_report(
    audit_id: str,
    format: str = Query(default="json", pattern="^(json|csv)$"),
    _owner_id: str = Depends(get_owner_id),
) -> Any:
    job = JOBS.get(audit_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Audit not found.")
    report = job.get("evaluation_audit")
    if not report:
        raise HTTPException(status_code=422, detail=job.get("audit_error") or "No audit report is available for this experiment.")
    if format == "json":
        return report

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "id", "title", "severity", "evidence", "explanation",
            "potential_impact", "recommended_correction", "verified",
        ],
    )
    writer.writeheader()
    writer.writerows(report.get("findings", []))
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="evaluation-audit-{audit_id}.csv"'},
    )


@router.post("/audit/{audit_id}/corrected-evaluation")
def corrected_evaluation(
    audit_id: str,
    request: CorrectedEvaluationRequest,
    _owner_id: str = Depends(get_owner_id),
) -> dict:
    job = JOBS.get(audit_id)
    if job is None or job.get("status") != "complete":
        raise HTTPException(status_code=404, detail="Completed experiment not found.")
    dataset = DATASETS.get(job.get("dataset_id", ""))
    if dataset is None or not Path(dataset.get("path", "")).is_file():
        raise HTTPException(status_code=404, detail="The experiment dataset is no longer available.")
    model_path = job.get("model_path")
    if not model_path or not Path(model_path).is_file():
        raise HTTPException(status_code=422, detail="The trained pipeline is unavailable for corrected evaluation.")
    try:
        dataframe = pd.read_csv(dataset["path"])
        pipeline = joblib.load(model_path)
        corrected = run_corrected_evaluation(
            dataframe=dataframe,
            target_column=job.get("target_column") or "",
            fitted_pipeline=pipeline,
            task_type=job.get("task_type") or "",
            excluded_features=request.excluded_features,
            feature_columns=job.get("features_used"),
            random_seed=(job.get("random_seed") or 42) + 1,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Corrected evaluation could not be completed with this experiment.") from exc

    if corrected["status"] != "complete":
        return corrected
    original_metrics = job.get("metrics", {})
    corrected_metrics = corrected["metrics"]
    comparison = [
        {
            "metric": metric,
            "original": original_metrics.get(metric),
            "corrected": value,
            "difference": (
                round(float(value) - float(original_metrics[metric]), 6)
                if isinstance(value, (int, float))
                and isinstance(original_metrics.get(metric), (int, float))
                else None
            ),
        }
        for metric, value in corrected_metrics.items()
    ]
    return {
        **corrected,
        "original_metrics": original_metrics,
        "comparison": comparison,
        "note": "The selected pipeline was cloned and retrained on a new split; preprocessing was fitted only on its training partition.",
    }
