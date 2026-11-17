"""Model metadata endpoint."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .state import JOBS

router = APIRouter()


@router.get("/models/{model_id}")
def get_model_metadata(model_id: str) -> dict:
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
