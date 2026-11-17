"""Dataset profiling endpoint."""

from fastapi import APIRouter, HTTPException

from .state import DATASETS

router = APIRouter()


@router.get("/datasets/{dataset_id}/profile")
def get_dataset_profile(dataset_id: str) -> dict:
    dataset = DATASETS.get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return {
        **dataset["profile"],
        "encoding": dataset["encoding"],
        "delimiter": dataset["delimiter"],
    }
