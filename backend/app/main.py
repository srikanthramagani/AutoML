"""FastAPI application entry point for the Phase 1 API skeleton."""

from fastapi import FastAPI

from .api import downloads, models, profiling, training, upload

app = FastAPI(
    title="Adaptive AutoML Platform API",
    version="0.1.0",
    description="Phase 1 API contract for CSV profiling and AutoML training.",
)

app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(profiling.router, prefix="/api", tags=["profiling"])
app.include_router(training.router, prefix="/api", tags=["training"])
app.include_router(models.router, prefix="/api", tags=["models"])
app.include_router(downloads.router, prefix="/api", tags=["downloads"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "phase": "1-skeleton"}
