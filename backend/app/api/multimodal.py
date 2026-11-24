"""Modality inspection and multimodal training/prediction endpoints."""
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
import json
from uuid import uuid4
import pandas as pd
import joblib
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from ..automl.multimodal import train_multimodal, explain_records
from ..data.modalities import detect_modalities
from ..config.settings import backend_settings
from .state import DATASETS, JOBS, PROJECT_ROOT

router=APIRouter()

class MultimodalRequest(BaseModel):
    dataset_id:str
    target_column:str
    representation:str="tfidf"
    fusion:str="early"
    trials:int=Field(6,ge=1,le=24)
    cv_folds:int=Field(3,ge=2,le=10)
    overrides:dict[str,str]=Field(default_factory=dict)
    seed:int=42

@router.get("/datasets/{dataset_id}/modalities")
def modalities(dataset_id:str):
    item=DATASETS.get(dataset_id)
    if not item: raise HTTPException(404,"Dataset not found")
    try:
        df=pd.read_csv(item["path"])
        report=detect_modalities(df,overrides=item.get("modality_overrides",{}))
        return report.to_dict()
    except Exception as exc: raise HTTPException(400,str(exc)) from exc

@router.post("/multimodal/train",status_code=202)
def multimodal_train(req:MultimodalRequest):
    item=DATASETS.get(req.dataset_id)
    if not item: raise HTTPException(404,"Dataset not found")
    if req.representation not in ("tfidf","sentence_transformer","auto"): raise HTTPException(400,"Unsupported representation")
    if req.fusion not in ("early","late","auto"): raise HTTPException(400,"Unsupported fusion strategy")
    try:
        frame=pd.read_csv(item["path"])
        report=detect_modalities(frame,req.target_column,req.overrides)
        outdir=PROJECT_ROOT/backend_settings.model_directory; outdir.mkdir(parents=True,exist_ok=True)
        path=outdir/f"multimodal_{req.dataset_id}.joblib"
        result=train_multimodal(frame,req.target_column,report,representation=req.representation,fusion=req.fusion,trials=req.trials,folds=req.cv_folds,seed=req.seed,export_path=path)
        job_id=f"multimodal-{req.dataset_id}"
        metadata={"model_id":job_id,"dataset_id":req.dataset_id,"target":req.target_column,"selected_model":result["selected_model"],"representation":result["representation"],"fusion":result["fusion"],"fusion_weight_text":result["fusion_weight"],"best_params":result["best_params"],"classes":result["classes"],"versions":{}}
        for package in ("scikit-learn","pandas","numpy","sentence-transformers","optuna"):
            try: metadata['versions'][package]=version(package)
            except PackageNotFoundError: pass
        metadata_path=path.with_suffix('.metadata.json'); metadata_path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        result.pop("model")
        JOBS[job_id]={"job_id":job_id,"status":"complete","model_path":str(path),"metadata_path":str(metadata_path),"best_model":result["selected_model"],"target_column":req.target_column,"dataset_id":req.dataset_id,**result,"modality_report":report.to_dict()}
        return JOBS[job_id]
    except Exception as exc: raise HTTPException(400,str(exc)) from exc

@router.post("/models/{model_id}/predict")
def predict(model_id:str,records:list[dict]):
    job=JOBS.get(model_id)
    if not job or not job.get("model_path"): raise HTTPException(404,"Model not found")
    try:
        estimator=joblib.load(job["model_path"]); frame=pd.DataFrame.from_records(records)
        labels=estimator.predict(frame)
        probs=estimator.predict_proba(frame)
        return {"predictions":[str(x) for x in labels],"probabilities":probs.tolist(),"classes":[str(c) for c in estimator.classes_]}
    except Exception as exc: raise HTTPException(400,str(exc)) from exc

@router.post("/models/{model_id}/explain")
def explain(model_id:str,records:list[dict]):
    job=JOBS.get(model_id)
    if not job or not job.get("model_path"): raise HTTPException(404,"Model not found")
    try:
        estimator=joblib.load(job['model_path'])
        return {"explanations":explain_records(estimator,pd.DataFrame.from_records(records)),"global_permutation_importance":job.get('feature_importance',[]),"modality_ablation":job.get('modality_ablation',{})}
    except Exception as exc: raise HTTPException(400,str(exc)) from exc

@router.post("/models/{model_id}/predict-upload")
async def predict_upload(model_id:str,file:UploadFile=File(...)):
    job=JOBS.get(model_id)
    if not job or not job.get('model_path'): raise HTTPException(404,"Model not found")
    if not file.filename or not file.filename.lower().endswith('.csv'): raise HTTPException(400,"Upload a CSV file")
    try:
        frame=pd.read_csv(file.file); estimator=joblib.load(job['model_path'])
        predictions=estimator.predict(frame); probabilities=estimator.predict_proba(frame)
        frame['prediction']=predictions
        for index,label in enumerate(estimator.classes_): frame[f'probability_{label}']=probabilities[:,index]
        if job.get('target_column') in frame:
            from sklearn.metrics import accuracy_score, f1_score
            truth=frame[job['target_column']]
            job['uploaded_evaluation']={'accuracy':float(accuracy_score(truth,predictions)),'macro_f1':float(f1_score(truth,predictions,average='macro',zero_division=0))}
        pred_dir=PROJECT_ROOT/backend_settings.experiment_directory; pred_dir.mkdir(parents=True,exist_ok=True)
        path=pred_dir/f"predictions_{uuid4().hex}.csv"; frame.to_csv(path,index=False)
        job['prediction_download']=str(path)
        return {'rows':len(frame),'predictions':[str(p) for p in predictions],'prediction_download':f'/api/models/{model_id}/prediction-download','evaluation':job.get('uploaded_evaluation')}
    except Exception as exc: raise HTTPException(400,str(exc)) from exc

@router.get("/models/{model_id}/prediction-download")
def download_predictions(model_id:str):
    job=JOBS.get(model_id); path=job.get('prediction_download') if job else None
    if not path or not Path(path).exists(): raise HTTPException(404,"Prediction file not found")
    return FileResponse(path,filename='predictions.csv',media_type='text/csv')
