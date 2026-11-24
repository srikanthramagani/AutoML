"""Model metadata endpoint."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from .state import JOBS
from .auth import get_owner_id

router = APIRouter()


@router.get("/models/{model_id}")
def get_model_metadata(
    model_id: str,
    _owner_id: str = Depends(get_owner_id),
) -> dict:
    job = JOBS.get(model_id)
    if job is None or job.get("status") != "complete":
        raise HTTPException(status_code=404, detail="Model not found.")
    return {
        "model_id": model_id,
        "best_model": job.get("best_model"),
        "metrics": job.get("metrics", {}),
        "model_path": job.get("model_path"),
        "best_params": job.get("best_params", {}),
        "leaderboard_chart": f"/api/models/{model_id}/leaderboard-chart",
    }


@router.get("/models/{model_id}/leaderboard-chart")
def get_leaderboard_chart(model_id: str):
    job = JOBS.get(model_id)
    chart_path = job.get("leaderboard_chart") if job else None
    if not chart_path:
        raise HTTPException(status_code=404, detail="Leaderboard chart not found.")
    return FileResponse(chart_path, media_type="image/png", filename="leaderboard.png")


@router.get("/experiments/optimizer-comparison-chart")
def get_optimizer_comparison_chart():
    from .state import PROJECT_ROOT
    p = PROJECT_ROOT / "experiments" / "results" / "optimizer_comparison.png"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Optimizer comparison chart not generated yet.")
    return FileResponse(str(p), media_type="image/png", filename="optimizer_comparison.png")


@router.get("/experiments/all-models-chart")
def get_all_models_chart():
    from .state import PROJECT_ROOT
    p = PROJECT_ROOT / "experiments" / "results" / "all_models_real_world_metrics.png"
    if not p.exists():
        raise HTTPException(status_code=404, detail="All models comparison chart not found.")
    return FileResponse(str(p), media_type="image/png", filename="all_models_real_world_metrics.png")


@router.get("/experiments/metrics-summary")
def get_experiments_metrics_summary():
    import pandas as pd
    from .state import PROJECT_ROOT
    res_path = PROJECT_ROOT / "experiments" / "results" / "all_models_trained_real_world_metrics.csv"
    multi_path = PROJECT_ROOT / "experiments" / "results" / "multi_dataset_results.csv"
    
    records = []
    if res_path.exists():
        records = pd.read_csv(res_path).to_dict("records")
        
    optimizer_records = []
    if multi_path.exists():
        opt_df = pd.read_csv(multi_path)
        if "optimizer" in opt_df.columns:
            subset = opt_df[opt_df["optimizer"].isin(["Optuna TPE", "GridSearchCV", "RandomizedSearchCV"])]
            optimizer_records = subset.to_dict("records")
            
    return {
        "all_models_metrics": records,
        "optimizer_comparison": optimizer_records,
    }
