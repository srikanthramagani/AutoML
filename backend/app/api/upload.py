"""CSV upload and lightweight profile endpoint."""

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ..config.settings import backend_settings
from ..data.loader import DataLoader
from ..data.profiler import DataProfiler
from ..data.modalities import detect_modalities
from .state import DATASETS, PROJECT_ROOT
from .auth import get_owner_id

router = APIRouter()


def _profile_response(profile, target: str) -> dict:
    columns = []
    for column in profile.columns.values():
        columns.append({
            "name": column.name,
            "dtype": column.dtype,
            "inferred_type": column.inferred_type,
            "missing": column.n_missing,
            "missing_pct": column.pct_missing,
            "unique": column.n_unique,
            "mean": column.mean,
            "std": column.std,
            "min": column.min,
            "q25": column.q25,
            "median": column.median,
            "q75": column.q75,
            "max": column.max,
            "top_values": column.top_values,
            "identifier_candidate": column.identifier_candidate,
            "identifier_excluded_by_default": column.identifier_excluded_by_default,
            "identifier_reason": column.identifier_reason,
        })
    return {
        "rows": profile.n_rows,
        "columns": profile.n_cols,
        "numeric_columns": profile.n_numeric,
        "categorical_columns": profile.n_categorical,
        "missing_pct": profile.total_missing_pct,
        "duplicates": profile.duplicate_row_count,
        "target": target,
        "identifier_columns": profile.identifier_columns,
        "identifier_reasons": profile.identifier_reasons,
        "column_details": columns,
    }


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    _owner_id: str = Depends(get_owner_id),
) -> dict:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    dataset_id = uuid4().hex
    upload_dir = PROJECT_ROOT / backend_settings.upload_directory
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{dataset_id}_{Path(file.filename).name}"
    path.write_bytes(await file.read())

    try:
        dataframe, load_result = DataLoader().load(path)
        profile = DataProfiler().profile(dataframe, target_column=load_result.target_column)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}") from exc

    response_profile = _profile_response(profile, load_result.target_column)
    modality_report = detect_modalities(dataframe, target=load_result.target_column).to_dict()
    DATASETS[dataset_id] = {
        "id": dataset_id,
        "filename": file.filename,
        "path": str(path),
        "encoding": load_result.encoding,
        "delimiter": load_result.delimiter,
        "profile": response_profile,
        "modality_report": modality_report,
    }
    return {
        "id": dataset_id,
        "filename": file.filename,
        "size": path.stat().st_size,
        "encoding": load_result.encoding,
        "delimiter": load_result.delimiter,
        **response_profile,
        "modality_report": modality_report,
    }
