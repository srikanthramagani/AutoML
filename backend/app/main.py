"""FastAPI application entry point for the Phase 1 API skeleton."""
from __future__ import annotations
from typing import Dict

from fastapi import Depends, FastAPI

from .api import downloads, models, profiling, training, upload, multimodal, forecasting
from .api.auth import get_owner_id

app = FastAPI(
    title="Adaptive AutoML Platform API",
    version="0.1.0",
    description="Phase 1 API contract for CSV profiling and AutoML training.",
    dependencies=[Depends(get_owner_id)],
)

app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(profiling.router, prefix="/api", tags=["profiling"])
app.include_router(training.router, prefix="/api", tags=["training"])
app.include_router(models.router, prefix="/api", tags=["models"])
app.include_router(downloads.router, prefix="/api", tags=["downloads"])
app.include_router(multimodal.router, prefix="/api", tags=["multimodal"])
app.include_router(forecasting.router, prefix="/api", tags=["forecasting"])


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "phase": "multimodal-extension"}
