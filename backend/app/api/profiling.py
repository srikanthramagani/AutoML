"""Dataset profiling endpoint."""

from fastapi import APIRouter, Depends, HTTPException

from .state import DATASETS
from .auth import get_owner_id

router = APIRouter()


@router.get("/datasets/{dataset_id}/profile")
def get_dataset_profile(
    dataset_id: str,
    _owner_id: str = Depends(get_owner_id),
) -> dict:
    dataset = DATASETS.get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return {
        **dataset.get("profile", {}),
        "encoding": dataset.get("encoding", "utf-8"),
        "delimiter": dataset.get("delimiter", ","),
    }
