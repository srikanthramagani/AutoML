"""Complete pipeline download endpoint."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .state import JOBS
router = APIRouter()


@router.get("/models/{model_id}/download")
def download_model(model_id: str):
    job = JOBS.get(model_id)
    model_path = job.get("model_path") if job else None
    if not model_path:
        raise HTTPException(status_code=404, detail="Model not found.")
    return FileResponse(model_path, filename="trained_pipeline.pkl", media_type="application/octet-stream")
